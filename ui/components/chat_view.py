"""
=============================================================================
Small Hydropower Plant (SHP) Agentic RAG - Chat View Component
=============================================================================
Module: ui.components.chat_view
Purpose: Streamlit chat feed rendering user and assistant conversation turns,
         circuit breaker alert callouts, LaTeX equations, and execution spinner.
=============================================================================
"""

import streamlit as st
from ui.api_client import AgentAPIClient


def render_chat_view(client: AgentAPIClient):
    """Renders the main chat feed, input field, and triggers agent execution."""
    st.markdown("## ⚡ **Small Hydropower Engineering Assistant**")
    st.caption("Ask questions regarding small hydropower (SHP) site selection, turbine sizing, electromechanical powerhouse costs, penstocks, and DPR verification.")

    # 1. Render conversation history
    for msg in st.session_state.messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        cb_tripped = msg.get("circuit_breaker_tripped", False)
        citations = msg.get("citations", [])

        with st.chat_message(role):
            # If circuit breaker was tripped on this response, render a high-visibility warning banner
            if role == "assistant" and cb_tripped:
                st.warning(
                    "⚠️ **Circuit Breaker Active (Max Retry Cycles Exceeded)**\n\n"
                    "The multi-agent validation loops exceeded the maximum 3 correction attempts. "
                    "The response below contains best-effort partial data extracted directly from available technical chunks, "
                    "or guidance on missing documentation.",
                    icon="⚠️",
                )

            # Render response content (supports markdown and LaTeX KaTeX math)
            st.markdown(content)

            # Citations preview badge/expander if present
            if role == "assistant" and citations:
                with st.expander(f"📚 View {len(citations)} Verified Citation(s)", expanded=False):
                    for idx, c in enumerate(citations, 1):
                        st.markdown(f"**Ref #{idx}: {c.get('source_file')}** (Score: {c.get('score', 0.0):.2f})")
                        st.caption(f"Section: `{c.get('breadcrumb', 'General')}`")
                        st.code(c.get("snippet", ""), language="markdown")

    # 2. Check for pre-filled pending query (e.g., from the calculator)
    prefill = st.session_state.pop("pending_query", None)

    # 3. Chat input field
    user_prompt = st.chat_input("Ask about Pelton vs Francis turbines, penstock design, powerhouse civil costs, or DPR metrics...")
    prompt_to_run = user_prompt or prefill

    if prompt_to_run:
        # Append User Message to session
        st.session_state.messages.append({
            "role": "user",
            "content": prompt_to_run,
        })

        # Display immediately in chat
        with st.chat_message("user"):
            st.markdown(prompt_to_run)

        # Assistant thinking container
        with st.chat_message("assistant"):
            with st.spinner("⚡ Multi-agent graph executing (Planner ➔ Sub-Agents ➔ Grader ➔ Synthesizer ➔ Guardrail)..."):
                response_data = client.send_query(
                    query=prompt_to_run,
                    session_id=st.session_state.session_id,
                )

            response_text = response_data.get("response", "No response received.")
            cb_tripped = bool(response_data.get("circuit_breaker_tripped", False))
            citations = response_data.get("citations", [])
            telemetry = response_data.get("telemetry", {})

            # Update session state for telemetry and citations view
            st.session_state.last_telemetry = telemetry
            st.session_state.active_citations = citations

            if cb_tripped:
                st.warning(
                    "⚠️ **Circuit Breaker Active (Max Retry Cycles Exceeded)**\n\n"
                    "The multi-agent validation loops exceeded the maximum 3 correction attempts. "
                    "The response below contains best-effort partial data extracted directly from available technical chunks, "
                    "or guidance on missing documentation.",
                    icon="⚠️",
                )

            st.markdown(response_text)

            # Store in session state
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "circuit_breaker_tripped": cb_tripped,
                "citations": citations,
            })

            st.rerun()
