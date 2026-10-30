"""
=============================================================================
Small Hydropower Plant (SHP) Agentic RAG - Synthesizer Agent Facade
=============================================================================
Module: src.agents.synthesizer.synthesizer
Purpose: Exposes clean facade APIs for the Synthesizer Agent, re-exporting the
         state definitions, nodes, and LangGraph workflow from agentnodes and agentgraph.
=============================================================================
"""

from src.agents.synthesizer.agentgraph import build_synthesizer_graph, run_synthesizer
from src.agents.synthesizer.agentnodes import (
    build_fallback_context,
    deduplicate_and_balance_chunks,
    render_synthesizer_prompts,
    synthesize_context,
    synthesizer_node,
)
from src.states.synthesizeragent_state import (
    RetrievedChunk,
    SynthesizerAgentState,
    SynthesizerOutput,
)

__all__ = [
    "RetrievedChunk",
    "SynthesizerAgentState",
    "SynthesizerOutput",
    "build_fallback_context",
    "build_synthesizer_graph",
    "deduplicate_and_balance_chunks",
    "render_synthesizer_prompts",
    "run_synthesizer",
    "synthesize_context",
    "synthesizer_node",
]

