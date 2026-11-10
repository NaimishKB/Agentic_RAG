"""
=============================================================================
Small Hydropower Plant (SHP) Agentic RAG - Environment & Tracing Setup
=============================================================================
Module: src.utils.env_setup
Purpose: Automatically loads environment variables from .env and synchronizes
         LangSmith and LangChain tracing credentials and project settings.
=============================================================================
"""

import os
from pathlib import Path

import dotenv
from loguru import logger

# Locate project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_PATH = PROJECT_ROOT / ".env"


def setup_environment():
    """Loads .env file and sets up LangSmith / LangChain tracing variables."""
    if ENV_PATH.exists():
        dotenv.load_dotenv(ENV_PATH, override=True)

    # Check LangSmith / LangChain tracing flags
    tracing_enabled = (
        os.getenv("LANGSMITH_TRACING", "").lower() in ("true", "1", "yes")
        or os.getenv("LANGCHAIN_TRACING_V2", "").lower() in ("true", "1", "yes")
    )

    if tracing_enabled:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGCHAIN_TRACING_V2"] = "true"

        # Sync API Key
        api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY", "")
        if api_key:
            os.environ["LANGSMITH_API_KEY"] = api_key
            os.environ["LANGCHAIN_API_KEY"] = api_key

        # Sync Project
        project = os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT", "agentic-rag-shp")
        os.environ["LANGSMITH_PROJECT"] = project
        os.environ["LANGCHAIN_PROJECT"] = project

        # Sync Endpoint
        endpoint = os.getenv("LANGSMITH_ENDPOINT") or os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
        os.environ["LANGSMITH_ENDPOINT"] = endpoint
        os.environ["LANGCHAIN_ENDPOINT"] = endpoint

        logger.debug(f"[ENV SETUP] LangSmith tracing ENABLED -> Project: '{project}' | Endpoint: '{endpoint}'")
    else:
        logger.debug("[ENV SETUP] LangSmith tracing is DISABLED.")


# Auto-run upon import
setup_environment()

