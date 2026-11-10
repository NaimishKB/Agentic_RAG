"""
=============================================================================
Small Hydropower Plant (SHP) Agentic RAG - Citations Component
=============================================================================
Module: src.ui.components.citations_view
Purpose: Streamlit component rendering verified source citations, breadcrumbs,
         similarity metrics, and document snippets.
=============================================================================
"""

import streamlit as st


def render_citations(citations: list[dict]):
    """Renders structured citations in a clean accordion interface."""
    st.markdown("### 📚 **Verified Technical Citations**")

    if not citations:
        st.info("No citations associated with this turn. Citations appear when document retrieval or tool outputs are synthesized.")
        return

    st.caption(f"Showing **{len(citations)}** verified technical source(s) referenced by the multi-agent graph:")

    for idx, item in enumerate(citations, 1):
        source_file = item.get("source_file", "Unknown Document")
        breadcrumb = item.get("breadcrumb", "General Section")
        score = item.get("score", 0.0)
        snippet = item.get("snippet", "No excerpt available.")

        # Clean display title
        expander_title = f"Ref #{idx}: {source_file} (Score: {score:.2f})"

        with st.expander(expander_title, expanded=(idx == 1)):
            st.markdown(f"**Section / Breadcrumb:** `{breadcrumb}`")
            st.markdown(f"**Relevance Similarity:** `{score:.3f}`")
            st.markdown("**Source Excerpt:**")
            st.code(snippet, language="markdown")
