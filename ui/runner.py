"""
=============================================================================
Small Hydropower Plant (SHP) Agentic RAG - Streamlit UI Launcher Runner
=============================================================================
Module: ui.runner
Purpose: Direct entry point to start the Streamlit application via
         `python ui/runner.py` or `uv run ui`.
=============================================================================
"""

import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from streamlit.web import cli as stcli


def main():
    """Launches the Streamlit UI."""
    app_path = str(Path(__file__).parent / "app.py")
    port = os.getenv("STREAMLIT_PORT", os.getenv("STREAMLIT_SERVER_PORT", "8601"))
    sys.argv = ["streamlit", "run", app_path, f"--server.port={port}", "--server.headless=true"]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
