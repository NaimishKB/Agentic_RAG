"""
=============================================================================
Small Hydropower Plant (SHP) Agentic RAG - Streamlit API Client
=============================================================================
Module: src.ui.api_client
Purpose: Lightweight, resilient HTTP client communicating directly with
         the FastAPI agent backend (default http://127.0.0.1:8600).
=============================================================================
"""

import os
from typing import Any
import requests

api_port = os.getenv("API_PORT", "8600")
API_BASE_URL = os.getenv("API_SERVER_URL", f"http://127.0.0.1:{api_port}").rstrip("/")
CHAT_TIMEOUT_S = int(os.getenv("API_CHAT_TIMEOUT_S", "300"))
HEALTH_TIMEOUT_S = int(os.getenv("API_HEALTH_TIMEOUT_S", "5"))


class AgentAPIClient:
    """HTTP Client interface connecting the Streamlit UI to the FastAPI server."""

    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url

    def check_health(self) -> dict[str, Any] | None:
        """
        Queries GET /api/v1/health to verify Ollama, Qdrant, and LangSmith status.
        Returns the parsed JSON response or None on failure.
        """
        url = f"{self.base_url}/api/v1/health"
        try:
            resp = requests.get(url, timeout=HEALTH_TIMEOUT_S)
            if resp.status_code == 200:
                return resp.json()
            return {
                "status": "degraded",
                "error": f"HTTP {resp.status_code}: {resp.text}",
            }
        except Exception as e:
            return {
                "status": "offline",
                "error": str(e),
            }

    def send_query(self, query: str, session_id: str | None = None) -> dict[str, Any]:
        """
        Sends an engineering question to POST /api/v1/chat.
        Returns the structured ChatResponse dictionary.
        """
        url = f"{self.base_url}/api/v1/chat"
        payload = {"query": query.strip()}
        if session_id:
            payload["session_id"] = session_id

        try:
            resp = requests.post(url, json=payload, timeout=CHAT_TIMEOUT_S)
            if resp.status_code == 200:
                return resp.json()
            error_msg = f"Server returned status {resp.status_code}: {resp.text}"
            return {
                "response": f"⚠️ **Backend Server Error**: {error_msg}",
                "circuit_breaker_tripped": False,
                "citations": [],
                "telemetry": {
                    "latency_seconds": 0.0,
                    "cycle_count": 0,
                    "routing_decision": "error",
                    "circuit_breaker_tripped": False,
                },
                "error": error_msg,
            }
        except requests.exceptions.Timeout:
            return {
                "response": "⚠️ **Request Timeout**: The agent workflow took longer than the configured timeout limit to formulate a response. Check your GPU/LLM load.",
                "circuit_breaker_tripped": True,
                "citations": [],
                "telemetry": {
                    "latency_seconds": CHAT_TIMEOUT_S,
                    "cycle_count": 3,
                    "routing_decision": "timeout",
                    "circuit_breaker_tripped": True,
                },
                "error": "Timeout",
            }
        except Exception as e:
            return {
                "response": f"⚠️ **Connection Failure**: Unable to reach FastAPI server at `{self.base_url}`. Ensure `uv run python src/api/app.py` is running.\n\n*Error details*: `{e}`",
                "circuit_breaker_tripped": False,
                "citations": [],
                "telemetry": {
                    "latency_seconds": 0.0,
                    "cycle_count": 0,
                    "routing_decision": "offline",
                    "circuit_breaker_tripped": False,
                },
                "error": str(e),
            }

    def upload_document(self, file_bytes: bytes, filename: str) -> dict[str, Any]:
        """
        Uploads a PDF or DOCX file to POST /api/v1/documents/upload.
        """
        url = f"{self.base_url}/api/v1/documents/upload"
        files = {"file": (filename, file_bytes)}
        try:
            resp = requests.post(url, files=files, timeout=60)
            if resp.status_code == 200:
                return resp.json()
            return {
                "status": "error",
                "message": f"Server rejected file with status {resp.status_code}: {resp.text}",
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed uploading file: {str(e)}",
            }
