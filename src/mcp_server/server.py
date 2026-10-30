"""
=============================================================================
Small Hydropower Plant (SHP) - FastAPI Model Context Protocol (MCP) Server
=============================================================================
Module: src.mcp_server.server
Purpose: FastAPI-powered MCP tool server providing standard JSON-RPC 2.0 /mcp
         endpoint and direct REST endpoints for SHP mathematical engineering
         calculations and SCADA telemetry access.
=============================================================================
"""

import json
import math
import os
import sys
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{function}</cyan>:<cyan>{line}</cyan> - {message}",
    level="INFO",
)

# ---------------------------------------------------------------------------
# FastAPI App Initialization
# ---------------------------------------------------------------------------
app = FastAPI(
    title="SHP Engineering MCP Server",
    description=(
        "Model Context Protocol (MCP) JSON-RPC 2.0 & REST API for Small Hydropower Plant "
        "engineering calculations (Power, Specific Speed, Penstock Sizing) and SCADA telemetry."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic Schemas for Tools
# ---------------------------------------------------------------------------
class HydroPowerRequest(BaseModel):
    discharge_m3s: float = Field(..., gt=0, description="Flow rate in cubic meters per second (m^3/s)")
    net_head_m: float = Field(..., gt=0, description="Net effective head in meters (m)")
    efficiency: float = Field(default=0.88, ge=0.50, le=0.98, description="Overall plant efficiency (0.50 - 0.98)")
    capacity_factor: float = Field(default=0.65, ge=0.10, le=1.0, description="Annual plant load factor")


class SpecificSpeedRequest(BaseModel):
    rotational_speed_rpm: float = Field(..., gt=0, description="Generator/turbine rotational speed in RPM")
    power_kw: float = Field(..., gt=0, description="Turbine rated capacity in kilowatts (kW)")
    net_head_m: float = Field(..., gt=0, description="Net effective head in meters (m)")


class PenstockDiameterRequest(BaseModel):
    design_discharge_m3s: float = Field(..., gt=0, description="Design flow rate in m^3/s")


class ScadaQueryRequest(BaseModel):
    plant_id: str = Field(..., description="Plant identifier, e.g. 'SHP-KHARI-01'")
    sensor_tag: str = Field(
        ...,
        description="Sensor tag name (e.g. forebay_water_level, penstock_pressure, generator_bearing_temp, turbine_vibration)",
    )


class MCPJsonRpcRequest(BaseModel):
    jsonrpc: str = Field(default="2.0", description="JSON-RPC version")
    id: int | str | None = Field(default=None, description="Request identifier")
    method: str = Field(..., description="MCP method name: initialize | tools/list | tools/call")
    params: dict[str, Any] = Field(default_factory=dict, description="Method parameters")


# ---------------------------------------------------------------------------
# Core SHP Engineering Tool Implementations
# ---------------------------------------------------------------------------
def calculate_hydro_power(
    discharge_m3s: float,
    net_head_m: float,
    efficiency: float = 0.88,
    capacity_factor: float = 0.65,
) -> dict[str, Any]:
    """
    Computes theoretical and electrical power output for a small hydro plant:
      P (kW) = 9.81 * Q (m^3/s) * H_net (m) * efficiency
    """
    if discharge_m3s <= 0 or net_head_m <= 0:
        return {"error": "Discharge and net head must be strictly positive."}
    if not (0.5 <= efficiency <= 0.98):
        return {"error": "Plant efficiency must be between 0.50 and 0.98."}

    rho_g = 9.81
    power_kw = rho_g * discharge_m3s * net_head_m * efficiency
    power_mw = power_kw / 1000.0

    annual_hours = 8760
    annual_generation_mwh = power_mw * annual_hours * capacity_factor

    return {
        "status": "success",
        "inputs": {
            "discharge_m3s": discharge_m3s,
            "net_head_m": net_head_m,
            "efficiency": efficiency,
            "capacity_factor": capacity_factor,
        },
        "results": {
            "power_kw": round(power_kw, 2),
            "power_mw": round(power_mw, 3),
            "annual_generation_mwh": round(annual_generation_mwh, 1),
            "formula_used": "P (kW) = 9.81 * Q * H_net * efficiency",
        },
    }


def calculate_specific_speed(
    rotational_speed_rpm: float,
    power_kw: float,
    net_head_m: float,
) -> dict[str, Any]:
    """
    Computes metric specific speed (Ns) and identifies recommended turbine type:
      Ns = (N * sqrt(P_kw)) / (H_net^(5/4))
    """
    if rotational_speed_rpm <= 0 or power_kw <= 0 or net_head_m <= 0:
        return {"error": "Rotational speed, power, and head must be positive."}

    ns = (rotational_speed_rpm * math.sqrt(power_kw)) / (net_head_m ** 1.25)

    if net_head_m > 250 or ns < 70:
        rec_type = "Pelton (Impulse)"
        rationale = "High head, low specific speed indicates impulse Pelton runner."
    elif 50 <= net_head_m <= 250 or (70 <= ns <= 300):
        rec_type = "Francis (Radial Reaction)"
        rationale = "Medium head and moderate specific speed suited for Francis runner."
    elif net_head_m < 50 or ns > 300:
        rec_type = "Kaplan / Propeller (Axial Reaction)"
        rationale = "Low head, high flow and high specific speed indicate axial Kaplan/propeller."
    else:
        rec_type = "Crossflow / Banki"
        rationale = "Wide operating envelope suitable for Crossflow or small Francis."

    return {
        "status": "success",
        "inputs": {
            "rotational_speed_rpm": rotational_speed_rpm,
            "power_kw": power_kw,
            "net_head_m": net_head_m,
        },
        "results": {
            "specific_speed_ns": round(ns, 2),
            "recommended_turbine": rec_type,
            "engineering_rationale": rationale,
            "formula_used": "Ns = (N * sqrt(P_kw)) / (H_net^(5/4))",
        },
    }


def calculate_penstock_diameter(
    design_discharge_m3s: float,
) -> dict[str, Any]:
    """
    Calculates economic internal diameter for penstock using Sarkaria empirical formula:
      D (m) = 0.62 * Q^(0.43)
    Also evaluates resulting flow velocity (recommended: 2.5 - 5.0 m/s).
    """
    if design_discharge_m3s <= 0:
        return {"error": "Design discharge must be positive."}

    diameter_m = 0.62 * (design_discharge_m3s ** 0.43)
    cross_sectional_area = (math.pi / 4.0) * (diameter_m ** 2)
    flow_velocity_ms = design_discharge_m3s / cross_sectional_area

    velocity_status = "OPTIMAL" if 2.5 <= flow_velocity_ms <= 5.0 else "OUT_OF_BOUNDS"
    note = (
        "Velocity within optimal 2.5-5.0 m/s range."
        if velocity_status == "OPTIMAL"
        else "Velocity outside typical 2.5-5.0 m/s guidelines. Consider adjusting diameter."
    )

    return {
        "status": "success",
        "inputs": {"design_discharge_m3s": design_discharge_m3s},
        "results": {
            "economic_diameter_m": round(diameter_m, 3),
            "cross_sectional_area_m2": round(cross_sectional_area, 3),
            "flow_velocity_ms": round(flow_velocity_ms, 2),
            "velocity_evaluation": velocity_status,
            "engineering_note": note,
            "formula_used": "D = 0.62 * Q^0.43 (Sarkaria)",
        },
    }


def query_plant_scada(
    plant_id: str,
    sensor_tag: str,
) -> dict[str, Any]:
    """
    Simulated placeholder for plant SCADA telemetry readings.
    """
    tag = sensor_tag.lower().strip()
    mock_telemetry: dict[str, dict[str, Any]] = {
        "forebay_water_level": {
            "tag": "FOREBAY_LVL_01",
            "value": 142.35,
            "unit": "meters a.s.l.",
            "status": "NORMAL",
            "threshold_max": 145.0,
        },
        "penstock_pressure": {
            "tag": "PENSTOCK_PT_01",
            "value": 14.8,
            "unit": "bar",
            "status": "NORMAL",
            "threshold_max": 18.0,
        },
        "generator_bearing_temp": {
            "tag": "GEN_BRG_TEMP_U1",
            "value": 64.2,
            "unit": "celsius",
            "status": "NORMAL",
            "threshold_warning": 75.0,
        },
        "turbine_vibration": {
            "tag": "TURB_VIB_RMS_U1",
            "value": 2.15,
            "unit": "mm/s",
            "status": "NORMAL",
            "threshold_warning": 4.5,
        },
        "tailrace_level": {
            "tag": "TAILRACE_LVL_01",
            "value": 92.1,
            "unit": "meters a.s.l.",
            "status": "NORMAL",
            "threshold_max": 96.0,
        },
    }

    matching = next((v for k, v in mock_telemetry.items() if k in tag or tag in k), None)
    if matching:
        return {
            "status": "success",
            "plant_id": plant_id,
            "sensor": matching,
            "simulated": True,
        }

    return {
        "status": "not_found",
        "plant_id": plant_id,
        "sensor_tag": sensor_tag,
        "available_sensors": list(mock_telemetry.keys()),
    }


# ---------------------------------------------------------------------------
# MCP Tool Metadata & Registry
# ---------------------------------------------------------------------------
TOOLS_REGISTRY: dict[str, dict[str, Any]] = {
    "calculate_hydro_power": {
        "name": "calculate_hydro_power",
        "description": "Calculates theoretical power output (kW, MW) and annual generation (MWh) from discharge (m^3/s), net head (m), and efficiency.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "discharge_m3s": {"type": "number", "description": "Flow rate in cubic meters per second (m^3/s)"},
                "net_head_m": {"type": "number", "description": "Net effective head in meters (m)"},
                "efficiency": {"type": "number", "default": 0.88, "description": "Overall plant efficiency (0.50 - 0.98)"},
                "capacity_factor": {"type": "number", "default": 0.65, "description": "Annual plant load factor (default: 0.65)"},
            },
            "required": ["discharge_m3s", "net_head_m"],
        },
        "handler": calculate_hydro_power,
    },
    "calculate_specific_speed": {
        "name": "calculate_specific_speed",
        "description": "Calculates metric specific speed (Ns) and recommends turbine type (Pelton, Francis, Kaplan, Crossflow) for given speed (RPM), power (kW), and net head (m).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "rotational_speed_rpm": {"type": "number", "description": "Generator/turbine rotational speed in RPM"},
                "power_kw": {"type": "number", "description": "Turbine rated capacity in kilowatts (kW)"},
                "net_head_m": {"type": "number", "description": "Net effective head in meters (m)"},
            },
            "required": ["rotational_speed_rpm", "power_kw", "net_head_m"],
        },
        "handler": calculate_specific_speed,
    },
    "calculate_penstock_diameter": {
        "name": "calculate_penstock_diameter",
        "description": "Calculates economic penstock internal diameter (m) and checks water velocity (m/s) using Sarkaria formula for given design discharge.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "design_discharge_m3s": {"type": "number", "description": "Design flow rate in m^3/s"},
            },
            "required": ["design_discharge_m3s"],
        },
        "handler": calculate_penstock_diameter,
    },
    "query_plant_scada": {
        "name": "query_plant_scada",
        "description": "Queries real-time or simulated SCADA sensor telemetry (bearing temperature, forebay water level, penstock pressure, turbine vibration) for a specified plant.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "plant_id": {"type": "string", "description": "Plant identifier, e.g. 'SHP-KHARI-01'"},
                "sensor_tag": {
                    "type": "string",
                    "description": "Sensor to inspect: forebay_water_level | penstock_pressure | generator_bearing_temp | turbine_vibration | tailrace_level",
                },
            },
            "required": ["plant_id", "sensor_tag"],
        },
        "handler": query_plant_scada,
    },
}


# ---------------------------------------------------------------------------
# JSON-RPC 2.0 Dispatcher (Core MCP Protocol)
# ---------------------------------------------------------------------------
def handle_jsonrpc_request(request: dict[str, Any]) -> dict[str, Any]:
    """Dispatches JSON-RPC 2.0 calls according to the Model Context Protocol."""
    req_id = request.get("id")
    method = request.get("method")
    params = request.get("params") or {}

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": "shp-engineering-fastapi-mcp",
                    "version": "1.0.0",
                },
            },
        }

    if method == "tools/list":
        tools_list = [
            {
                "name": meta["name"],
                "description": meta["description"],
                "inputSchema": meta["inputSchema"],
            }
            for meta in TOOLS_REGISTRY.values()
        ]
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": tools_list},
        }

    if method == "tools/call":
        tool_name = params.get("name")
        tool_args = params.get("arguments") or {}

        if tool_name not in TOOLS_REGISTRY:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32601,
                    "message": f"Tool '{tool_name}' not found on this server.",
                },
            }

        handler = TOOLS_REGISTRY[tool_name]["handler"]
        try:
            result = handler(**tool_args)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(result, indent=2),
                        }
                    ]
                },
            }
        except (TypeError, ValueError, KeyError) as err:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32602,
                    "message": f"Invalid arguments for tool '{tool_name}': {err}",
                },
            }

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {
            "code": -32601,
            "message": f"Method '{method}' not supported.",
        },
    }


# ---------------------------------------------------------------------------
# FastAPI Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", tags=["System"])
@app.get("/", tags=["System"])
def health_check() -> dict[str, Any]:
    """Returns server status, version, and count of registered MCP tools."""
    return {
        "status": "healthy",
        "server": "shp-engineering-fastapi-mcp",
        "version": "1.0.0",
        "registered_tools_count": len(TOOLS_REGISTRY),
        "docs_url": "/docs",
        "mcp_endpoint": "/mcp",
    }


@app.post("/mcp", tags=["MCP JSON-RPC"])
async def mcp_endpoint(req: MCPJsonRpcRequest) -> JSONResponse:
    """Standard Model Context Protocol (MCP) JSON-RPC 2.0 endpoint."""
    response = handle_jsonrpc_request(req.model_dump())
    return JSONResponse(content=response)


@app.get("/tools", tags=["Tools Catalog"])
def list_tools() -> dict[str, Any]:
    """REST endpoint to retrieve all available tools and their schemas."""
    return {
        "tools": [
            {
                "name": meta["name"],
                "description": meta["description"],
                "inputSchema": meta["inputSchema"],
            }
            for meta in TOOLS_REGISTRY.values()
        ]
    }


@app.post("/tools/{tool_name}", tags=["Tools Execution"])
async def execute_tool_rest(tool_name: str, request: Request) -> dict[str, Any]:
    """Direct REST invocation of any registered engineering tool."""
    if tool_name not in TOOLS_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found.")

    try:
        body = await request.json()
    except (json.JSONDecodeError, ValueError):
        body = {}

    handler = TOOLS_REGISTRY[tool_name]["handler"]
    try:
        return handler(**body)
    except (TypeError, ValueError, KeyError) as err:
        raise HTTPException(status_code=400, detail=f"Invalid arguments for tool '{tool_name}': {err}") from err


# ---------------------------------------------------------------------------
# Dedicated REST Endpoints for Interactive Swagger Docs
# ---------------------------------------------------------------------------
@app.post("/api/v1/calculate/hydro-power", tags=["Calculations"])
def api_calculate_hydro_power(req: HydroPowerRequest) -> dict[str, Any]:
    """Calculates power output (kW, MW) and annual MWh generation."""
    return calculate_hydro_power(
        discharge_m3s=req.discharge_m3s,
        net_head_m=req.net_head_m,
        efficiency=req.efficiency,
        capacity_factor=req.capacity_factor,
    )


@app.post("/api/v1/calculate/specific-speed", tags=["Calculations"])
def api_calculate_specific_speed(req: SpecificSpeedRequest) -> dict[str, Any]:
    """Calculates metric specific speed (Ns) and recommends turbine type."""
    return calculate_specific_speed(
        rotational_speed_rpm=req.rotational_speed_rpm,
        power_kw=req.power_kw,
        net_head_m=req.net_head_m,
    )


@app.post("/api/v1/calculate/penstock-diameter", tags=["Calculations"])
def api_calculate_penstock_diameter(req: PenstockDiameterRequest) -> dict[str, Any]:
    """Calculates economic penstock diameter (m) and water velocity (m/s)."""
    return calculate_penstock_diameter(
        design_discharge_m3s=req.design_discharge_m3s,
    )


@app.post("/api/v1/scada/query", tags=["SCADA Telemetry"])
def api_query_plant_scada(req: ScadaQueryRequest) -> dict[str, Any]:
    """Queries simulated real-time SCADA telemetry for plant sensors."""
    return query_plant_scada(
        plant_id=req.plant_id,
        sensor_tag=req.sensor_tag,
    )


# ---------------------------------------------------------------------------
# Server Launcher
# ---------------------------------------------------------------------------
def start_server(host: str = "0.0.0.0", port: int = 8000):
    """Launches the FastAPI server with Uvicorn."""
    logger.info(f"Starting FastAPI SHP MCP Server on http://{host}:{port} (Docs: http://{host}:{port}/docs)")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    host_env = os.getenv("MCP_HOST", "0.0.0.0")
    port_env = int(os.getenv("MCP_PORT", "8000"))
    start_server(host=host_env, port=port_env)
