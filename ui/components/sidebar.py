"""
=============================================================================
Small Hydropower Plant (SHP) Agentic RAG - Sidebar Component
=============================================================================
Module: ui.components.sidebar
Purpose: Streamlit sidebar rendering system health indicators, session management,
         execution telemetry, and document upload capabilities.
=============================================================================
"""

import uuid
import streamlit as st
from ui.api_client import AgentAPIClient


def render_sidebar(client: AgentAPIClient):
    """Renders the sidebar navigation, health probes, telemetry, and uploads."""
    with st.sidebar:
        st.markdown("### ⚡ **SHP Agentic Platform**")
        st.caption("Decoupled LangGraph Multi-Agent Architecture")
        st.divider()

        # -------------------------------------------------------------------
        # 1. System Health Status
        # -------------------------------------------------------------------
        st.markdown("#### 🩺 **System Diagnostics**")
        
        col_btn, _ = st.columns([1, 1])
        with col_btn:
            if st.button("🔄 Refresh", use_container_width=True, key="btn_refresh_health"):
                st.session_state.health_data = client.check_health()

        if "health_data" not in st.session_state or st.session_state.health_data is None:
            st.session_state.health_data = client.check_health()

        health = st.session_state.health_data or {}
        overall_status = health.get("status", "offline")

        if overall_status == "healthy":
            st.success("● Core API: Online", icon="✅")
        elif overall_status == "degraded":
            st.warning("● Core API: Degraded", icon="⚠️")
        else:
            st.error("● Core API: Offline", icon="❌")

        # Detailed service breakdown in an expander
        with st.expander("Service Connectivity Details", expanded=False):
            ollama = health.get("ollama", {})
            qdrant = health.get("qdrant", {})
            langsmith = health.get("langsmith", {})

            # Ollama
            if ollama.get("connected"):
                st.markdown(f"**Ollama LLM**: 🟢 Online (`{ollama.get('details', {}).get('model', 'qwen3.5')}`)")
            else:
                st.markdown(f"**Ollama LLM**: 🔴 Offline ({ollama.get('status', 'Error')})")

            # Qdrant
            if qdrant.get("connected"):
                pts = qdrant.get("details", {}).get("points_count", "N/A")
                st.markdown(f"**Qdrant Vector DB**: 🟢 Online (`{pts}` vectors)")
            else:
                st.markdown(f"**Qdrant Vector DB**: 🔴 Offline ({qdrant.get('status', 'Error')})")

            # LangSmith
            if langsmith.get("connected"):
                proj = langsmith.get("details", {}).get("project", "default")
                st.markdown(f"**LangSmith**: 🟢 Active (`{proj}`)")
            else:
                st.markdown("**LangSmith**: ⚪ Disabled")

        st.divider()

        # -------------------------------------------------------------------
        # 2. Session Management
        # -------------------------------------------------------------------
        st.markdown("#### 💬 **Session Management**")
        curr_session = st.session_state.get("session_id", "Not set")
        st.text_input("Active Session ID", value=curr_session, disabled=True, key="input_session_id")

        if st.button("➕ New Conversation", use_container_width=True, type="secondary"):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.session_state.last_telemetry = None
            st.session_state.active_citations = []
            st.rerun()

        st.divider()

        # -------------------------------------------------------------------
        # 3. Latest Query Telemetry
        # -------------------------------------------------------------------
        st.markdown("#### 📊 **Engine Telemetry**")
        telemetry = st.session_state.get("last_telemetry")

        if telemetry:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Latency", f"{telemetry.get('latency_seconds', 0.0):.1f}s")
                st.metric("Cycles", f"{telemetry.get('cycle_count', 0)} / 3")
            with col2:
                cb_tripped = telemetry.get("circuit_breaker_tripped", False)
                if cb_tripped:
                    st.metric("Circuit Breaker", "TRIPPED", delta="Exceeded", delta_color="inverse")
                else:
                    st.metric("Circuit Breaker", "Normal", delta="OK")

                chunks = telemetry.get("total_chunks_retrieved", 0)
                st.metric("Chunks Examined", chunks)

            # Quality badges
            st.caption("Verification Checks:")
            c_suff = telemetry.get("is_retrieval_sufficient")
            c_ground = telemetry.get("is_grounded")

            badge_suff = "🟢 Sufficient" if c_suff is True else ("🔴 Insufficient" if c_suff is False else "⚪ Skipped")
            badge_ground = "🟢 Faithful" if c_ground is True else ("🔴 Unfaithful" if c_ground is False else "⚪ Skipped")

            st.markdown(f"- Retrieval: **{badge_suff}**")
            st.markdown(f"- Grounding: **{badge_ground}**")
        else:
            st.info("Submit a question to view real-time agent execution telemetry.")

        st.divider()

        # -------------------------------------------------------------------
        # 4. Document Ingestion Portal
        # -------------------------------------------------------------------
        st.markdown("#### 📄 **Ingest Plant Documentation**")
        uploaded_file = st.file_uploader(
            "Upload DPR, Manual, or Specification",
            type=["pdf", "docx"],
            help="Saves documents directly into the plant knowledge repository.",
            key="uploader_widget",
        )

        if uploaded_file is not None:
            if st.button("📤 Upload Document", use_container_width=True, type="primary"):
                with st.spinner("Uploading to backend storage..."):
                    file_bytes = uploaded_file.getvalue()
                    upload_res = client.upload_document(file_bytes, uploaded_file.name)
                    if upload_res.get("status") == "uploaded":
                        st.success(f"Uploaded `{uploaded_file.name}` successfully!")
                    else:
                        st.error(upload_res.get("message", "Upload failed."))
