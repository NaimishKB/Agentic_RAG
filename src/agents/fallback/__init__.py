"""
Fallback Package - Circuit Breaker & Graceful Refusal Node (FR-11).
"""

from src.agents.fallback.agentnodes import (
    fallback_node,
    format_best_effort_generation_response,
    format_circuit_breaker_response,
    format_out_of_scope_response,
    format_partial_retrieval_response,
    format_unavailable_in_docs_response,
)

__all__ = [
    "fallback_node",
    "format_best_effort_generation_response",
    "format_circuit_breaker_response",
    "format_out_of_scope_response",
    "format_partial_retrieval_response",
    "format_unavailable_in_docs_response",
]
