"""
=============================================================================
Small Hydropower Plant (SHP) Agentic RAG - Streamlit Web Application
=============================================================================
Module: ui.app
Purpose: Main entry point for the Streamlit UI communicating with the
         FastAPI server over asynchronous REST endpoints.
=============================================================================
"""

import sys
import uuid
from pathlib import Path

# Add project root to sys.path so modules resolve correctly if invoked from subdirectories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  # noqa: E402
from ui.api_client import AgentAPIClient  # noqa: E402
from ui.components.calculator import render_calculator  # noqa: E402
from ui.components.chat_view import render_chat_view  # noqa: E402
from ui.components.citations_view import render_citations  # noqa: E402
from ui.components.sidebar import render_sidebar  # noqa: E402

# ---------------------------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SHP Multi-Agent RAG Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.2);
        padding: 10px 14px;
        border-radius: 8px;
    }
    /* Chat message spacing */
    div[data-testid="stChatMessage"] {
        padding: 14px;
        border-radius: 8px;
        margin-bottom: 8px;
    }
    /* Tabs */
    button[data-baseweb="tab"] {
        font-size: 1.05rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_telemetry" not in st.session_state:
    st.session_state.last_telemetry = None

if "active_citations" not in st.session_state:
    st.session_state.active_citations = []

if "health_data" not in st.session_state:
    st.session_state.health_data = None

# ---------------------------------------------------------------------------
# Initialize API Client
# ---------------------------------------------------------------------------
api_client = AgentAPIClient()

# ---------------------------------------------------------------------------
# Render Components
# ---------------------------------------------------------------------------
# Sidebar (Health, Session, Telemetry summary, Document Upload)
render_sidebar(api_client)

# Main Application Tabs
tab_chat, tab_calc, tab_citations, tab_arch = st.tabs([
    "💬 Engineering Assistant",
    "⚙️ Hydro Sizing Calculator",
    "📚 Citations Inspector",
    "🏛️ System Architecture",
])

with tab_chat:
    render_chat_view(api_client)

with tab_calc:
    render_calculator()

with tab_citations:
    render_citations(st.session_state.get("active_citations", []))

with tab_arch:
    st.markdown("### 🏛️ **Agentic Workflow Architecture**")
    st.markdown(
        r"""
        The Small Hydropower Plant (SHP) Assistant utilizes an enterprise-grade **LangGraph multi-agent orchestration** 
        architecture with Corrective RAG (CRAG) and Faithfulness Guardrails:
        
        1. **Planner Agent**: Classifies query intent (`rag_retrieval`, `programmatic_tool`, or `out_of_scope_rejection`).
        2. **Orchestrator & Specialized Sub-Agents**:
           - **Civil Engineering Sub-Agent**: Penstocks, intakes, headrace tunnels, forebay tanks.
           - **Electro-Mechanical Sub-Agent**: Pelton, Francis, Kaplan turbines, generators, governors, transformers.
           - **Cost & DPR Sub-Agent**: Cost curves, civil/E&M cost models (IRENA, ESHA, Aggidis formulas).
           - **Python Code Execution / MCP Agent**: Deterministic fluid mechanics formulas ($P = \rho g Q H \eta$).
        3. **Corrective RAG (CRAG) Grader**: Evaluates retrieved chunks against quantitative query constraints.
        4. **Synthesizer Engine**: Deterministic zero-latency evidence compilation with structured references.
        5. **Faithfulness & Grounding Guardrail**: Detects hallucinations and enforces numerical verifiability.
        6. **Fallback Circuit Breaker (FR-11)**: Automatically halts execution after 3 failed correction cycles, returning 
           explicit partial documentation excerpts or missing-data alerts with clear disclaimers.
        """
    )
    st.info(
        "💡 **Decoupled Architecture**: This Streamlit UI communicates strictly via HTTP with the standalone "
        "FastAPI backend (`http://127.0.0.1:8600`), ensuring complete isolation of execution runtimes."
    )
