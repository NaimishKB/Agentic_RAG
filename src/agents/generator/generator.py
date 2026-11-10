"""
=============================================================================
Small Hydropower Plant (SHP) Agentic RAG - Generator Agent Facade
=============================================================================
Module: src.agents.generator.generator
Purpose: Exposes clean facade APIs for the Answer Generator Agent,
         re-exporting state definitions, nodes, and the compiled LangGraph workflow.
=============================================================================
"""

from src.agents.generator.agentgraph import (
    build_generator_graph,
    run_generator,
)
from src.agents.generator.agentnodes import (
    extract_citations,
    generate_answer,
    generator_node,
    render_generator_prompts,
)
from src.states.generatoragent_state import (
    GeneratorAgentState,
    GeneratorOutput,
)

__all__ = [
    "GeneratorAgentState",
    "GeneratorOutput",
    "build_generator_graph",
    "extract_citations",
    "generate_answer",
    "generator_node",
    "render_generator_prompts",
    "run_generator",
]

