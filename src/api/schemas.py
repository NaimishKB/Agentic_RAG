"""
=============================================================================
Small Hydropower Plant (SHP) Agentic RAG - FastAPI Pydantic Schemas
=============================================================================
Module: src.api.schemas
Purpose: Request and Response Pydantic models for the FastAPI agent server.
=============================================================================
"""

from typing import Any
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Incoming user chat query request."""
    query: str = Field(..., min_length=1, description="Engineering question from user", example="What is the difference between Pelton and Francis turbines for 45m head?")
    session_id: str | None = Field(default=None, description="Optional conversation UUID for multi-turn session memory")


class CitationItem(BaseModel):
    """Structured citation metadata for verified sources."""
    source_file: str = Field(..., description="Source document filename or tool name")
    breadcrumb: str = Field(default="General", description="Section or slide breadcrumb")
    score: float = Field(default=0.0, description="Cosine relevance similarity score")
    snippet: str = Field(default="", description="Relevant document excerpt")


class TelemetryData(BaseModel):
    """Runtime execution telemetry for the multi-agent graph lifecycle."""
    latency_seconds: float = Field(..., description="Total end-to-end execution latency in seconds")
    cycle_count: int = Field(default=0, description="Self-correction cycles executed")
    routing_decision: str = Field(default="rag_retrieval", description="Routing decision from Planner")
    is_retrieval_sufficient: bool | None = Field(default=None, description="CRAG Grader sufficiency verdict")
    is_grounded: bool | None = Field(default=None, description="Guardrail faithfulness verdict")
    circuit_breaker_tripped: bool = Field(default=False, description="True if max retry cycles exceeded")
    total_chunks_retrieved: int = Field(default=0, description="Count of chunks retrieved across sub-agents")


class ChatResponse(BaseModel):
    """Complete structured response returned to the frontend."""
    query: str = Field(..., description="Original user query")
    session_id: str = Field(..., description="Conversation session UUID")
    response: str = Field(..., description="Final markdown answer or circuit breaker notice")
    routing_decision: str = Field(default="rag_retrieval", description="Planner classification")
    circuit_breaker_tripped: bool = Field(default=False, description="True if fallback circuit breaker activated")
    citations: list[CitationItem] = Field(default_factory=list, description="Verified citations")
    telemetry: TelemetryData = Field(..., description="Performance telemetry")
    retrieved_chunks: list[dict[str, Any]] = Field(default_factory=list, description="All retrieved document chunks with content and metadata")


class ServiceStatus(BaseModel):
    """Health check status for an individual backend service."""
    connected: bool
    status: str
    details: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """System-wide health check response."""
    status: str = Field(..., description="'healthy' or 'degraded'")
    ollama: ServiceStatus
    qdrant: ServiceStatus
    langsmith: ServiceStatus
    timestamp: str


class DocumentUploadResponse(BaseModel):
    """Response returned upon document file upload."""
    filename: str
    size_bytes: int
    status: str
    message: str

