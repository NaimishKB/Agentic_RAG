"""
=============================================================================
Small Hydropower Plant (SHP) Agentic RAG - Generation Guardrail Facade
=============================================================================
Module: src.agents.generation_guardrail.generation_guardrail
Purpose: Exposes clean facade APIs for the Post-Generation Faithfulness Guardrail,
         re-exporting state definitions, nodes, and the compiled LangGraph workflow.
=============================================================================
"""

from src.agents.generation_guardrail.agentgraph import (
    build_generation_guardrail_graph,
    run_generation_guardrail,
)
from src.agents.generation_guardrail.agentnodes import (
    evaluate_generation_faithfulness,
    generation_guardrail_node,
)
from src.states.generation_guardrail_state import (
    GenerationGuardrailOutput,
    GenerationGuardrailState,
)

__all__ = [
    "GenerationGuardrailOutput",
    "GenerationGuardrailState",
    "build_generation_guardrail_graph",
    "evaluate_generation_faithfulness",
    "generation_guardrail_node",
    "run_generation_guardrail",
]

