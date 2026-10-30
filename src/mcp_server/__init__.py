"""
=============================================================================
Small Hydropower Plant (SHP) - FastAPI MCP Server Package
=============================================================================
Module: src.mcp_server
Purpose: Provides standalone FastAPI Model Context Protocol (MCP) tool server
         and client for engineering calculations (power, specific speed,
         penstock sizing) and plant SCADA telemetry access.
=============================================================================
"""

from src.mcp_server.client import FastAPIMCPClient, LocalMCPClient
from src.mcp_server.server import (
    app,
    calculate_hydro_power,
    calculate_penstock_diameter,
    calculate_specific_speed,
    query_plant_scada,
)

__all__ = [
    "FastAPIMCPClient",
    "LocalMCPClient",
    "app",
    "calculate_hydro_power",
    "calculate_penstock_diameter",
    "calculate_specific_speed",
    "query_plant_scada",
]
