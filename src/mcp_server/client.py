"""
=============================================================================
Small Hydropower Plant (SHP) - FastAPI MCP Client for Orchestrator Agent
=============================================================================
Module: src.mcp_server.client
Purpose: Provides an MCP Client that connects either in-process or over HTTP
         to the FastAPI MCP server, enabling the Orchestrator Agent to discover,
         list, and invoke tools.
=============================================================================
"""

import json
from typing import Any

import requests
from loguru import logger

from src.mcp_server.server import handle_jsonrpc_request


class FastAPIMCPClient:
    """
    Client for interacting with the FastAPI MCP server.

    Modes:
      - in_process: Directly invokes the FastAPI JSON-RPC dispatcher with 0 latency.
      - http: Sends JSON-RPC or REST calls over HTTP to the running FastAPI server.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        prefer_http: bool = False,
        timeout: int = 15,
    ):
        self.base_url = base_url.rstrip("/")
        self.prefer_http = prefer_http
        self.timeout = timeout
        self._request_counter = 0

    def _next_id(self) -> int:
        self._request_counter += 1
        return self._request_counter

    def is_server_online(self) -> bool:
        """Checks if the FastAPI server is running on the configured base_url."""
        try:
            r = requests.get(f"{self.base_url}/health", timeout=2)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def list_tools(self) -> list[dict[str, Any]]:
        """Retrieves list of registered tools and their schemas."""
        if self.prefer_http and self.is_server_online():
            try:
                r = requests.get(f"{self.base_url}/tools", timeout=self.timeout)
                if r.status_code == 200:
                    return r.json().get("tools", [])
            except requests.RequestException as exc:
                logger.warning(f"[MCP CLIENT] HTTP list_tools failed ({exc}), falling back to in-process.")

        # In-process MCP dispatch
        req = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/list",
            "params": {},
        }
        resp = handle_jsonrpc_request(req)
        return resp.get("result", {}).get("tools", [])

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Invokes a tool via HTTP if server is live, or in-process as fallback."""
        logger.info(f"[MCP CLIENT] Calling tool '{tool_name}' with args: {arguments}")

        if self.prefer_http and self.is_server_online():
            try:
                # Direct REST invocation on FastAPI
                url = f"{self.base_url}/tools/{tool_name}"
                r = requests.post(url, json=arguments, timeout=self.timeout)
                if r.status_code == 200:
                    return r.json()
                logger.warning(f"[MCP CLIENT] REST tool call returned {r.status_code}: {r.text}")
            except requests.RequestException as exc:
                logger.warning(f"[MCP CLIENT] HTTP call failed ({exc}), falling back to in-process.")

        # In-process JSON-RPC execution
        req = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments,
            },
        }
        resp = handle_jsonrpc_request(req)
        if "error" in resp:
            logger.error(f"[MCP CLIENT] Tool error: {resp['error']}")
            return {"status": "error", "message": resp["error"].get("message")}

        content_list = resp.get("result", {}).get("content", [])
        if content_list and content_list[0].get("type") == "text":
            try:
                return json.loads(content_list[0]["text"])
            except json.JSONDecodeError:
                return {"status": "raw", "text": content_list[0]["text"]}

        return {"status": "empty"}

    def get_prompt_catalog(self) -> str:
        """Formats tools into markdown bullet points for injection into agent prompts."""
        tools = self.list_tools()
        lines = ["### Available MCP Engineering Tools:\n"]
        for t in tools:
            name = t.get("name", "")
            desc = t.get("description", "")
            params = t.get("inputSchema", {}).get("properties", {})
            param_names = ", ".join(params.keys())
            lines.append(f"- **`{name}({param_names})`**: {desc}")
        return "\n".join(lines)


# Backwards compatibility alias
LocalMCPClient = FastAPIMCPClient


# ---------------------------------------------------------------------------
# Verification CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    client = FastAPIMCPClient()
    print("\n" + "=" * 70)
    print("FASTAPI MCP CLIENT DISCOVERY & VERIFICATION TEST")
    print("=" * 70)

    # 1. Discover tools
    tools = client.list_tools()
    print(f"\nDiscovered {len(tools)} tools on FastAPI MCP server:")
    for t in tools:
        print(f"  • {t['name']}: {t['description'][:75]}...")

    # 2. Test Hydro Power Calculation tool
    print("\n[Tool Test 1]: calculate_hydro_power(Q=4.5 m^3/s, Head=120m, efficiency=0.88)")
    p_result = client.call_tool(
        "calculate_hydro_power",
        {"discharge_m3s": 4.5, "net_head_m": 120.0, "efficiency": 0.88},
    )
    print(json.dumps(p_result, indent=2))

    # 3. Test Specific Speed tool
    print("\n[Tool Test 2]: calculate_specific_speed(N=600 rpm, P=4500 kW, H=120m)")
    ns_result = client.call_tool(
        "calculate_specific_speed",
        {"rotational_speed_rpm": 600, "power_kw": 4500.0, "net_head_m": 120.0},
    )
    print(json.dumps(ns_result, indent=2))

    # 4. Test Penstock Diameter tool
    print("\n[Tool Test 3]: calculate_penstock_diameter(Q=4.5 m^3/s)")
    d_result = client.call_tool(
        "calculate_penstock_diameter",
        {"design_discharge_m3s": 4.5},
    )
    print(json.dumps(d_result, indent=2))

    print("=" * 70 + "\n")
