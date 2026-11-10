"""
=============================================================================
Small Hydropower Plant (SHP) Agentic RAG - FastAPI Application Server
=============================================================================
Module: src.api.app
Purpose: Production-grade asynchronous REST API exposing the compiled
         LangGraph multi-agent workflow, system health probes, and
         document ingestion capabilities to Streamlit and external clients.
=============================================================================
"""

import datetime
import os
import shutil
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path when running as `python src/api/app.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import requests

from src.agents.main_agent import build_main_agent_graph
from src.api.schemas import (
    ChatRequest,
    ChatResponse,
    CitationItem,
    DocumentUploadResponse,
    HealthResponse,
    ServiceStatus,
    TelemetryData,
)
from src.states.agent_state import AgentState
import src.utils.env_setup  # Synchronize environment and LangSmith variables

# Configure logger
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)
os.makedirs("logs", exist_ok=True)
logger.add(
    "logs/fastapi_server.log",
    rotation="20 MB",
    retention="14 days",
    level="DEBUG",
    encoding="utf-8",
)

# Configuration Constants
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_LLM_MODEL", "qwen3.5:9b")

raw_qdrant_host = os.getenv("QDRANT_HOST", "127.0.0.1")
qdrant_port = os.getenv("QDRANT_PORT", "6333")
if not raw_qdrant_host.startswith("http://") and not raw_qdrant_host.startswith(
    "https://"
):
    QDRANT_BASE_URL = f"http://{raw_qdrant_host}:{qdrant_port}"
else:
    QDRANT_BASE_URL = raw_qdrant_host

QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "shp_knowledge")
DATA_DIR = Path("data")


# ---------------------------------------------------------------------------
# Lifespan Application Manager (Compiles LangGraph Singleton Once)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes and pre-compiles the LangGraph workflow on server startup."""
    logger.info("[STARTUP] Initializing SHP Agentic RAG FastAPI Server...")
    t0 = time.perf_counter()
    try:
        app.state.graph = build_main_agent_graph()
        duration = time.perf_counter() - t0
        logger.success(
            f"[STARTUP] LangGraph multi-agent workflow compiled in {duration:.2f}s."
        )
    except Exception as e:
        logger.error(f"[STARTUP ERROR] Failed to compile LangGraph workflow: {e}")
        app.state.graph = None

    yield

    logger.info("[SHUTDOWN] SHP Agentic RAG FastAPI Server shutting down.")


# ---------------------------------------------------------------------------
# FastAPI Application Declaration
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Small Hydropower Plant (SHP) Agentic RAG API",
    description="Multi-Agent Engineering Assistant with Self-Correction, CRAG Graders, and Fallback Circuit Breakers",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable Cross-Origin Resource Sharing (CORS) for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health Check & Dependency Probing
# ---------------------------------------------------------------------------
@app.get("/api/v1/health", response_model=HealthResponse, tags=["Health"])
def get_system_health() -> HealthResponse:
    """Verifies connectivity to Ollama, Qdrant, and LangSmith tracing."""
    now_str = datetime.datetime.now(datetime.timezone.utc).isoformat()
    overall_healthy = True

    # 1. Probe Ollama
    ollama_status = ServiceStatus(
        connected=False, status="unreachable", details={"url": OLLAMA_BASE_URL}
    )
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3.0)
        if resp.status_code == 200:
            models = [m.get("name") for m in resp.json().get("models", [])]
            model_present = any(OLLAMA_MODEL in m for m in models)
            ollama_status = ServiceStatus(
                connected=True,
                status="online" if model_present else f"model_{OLLAMA_MODEL}_missing",
                details={"model": OLLAMA_MODEL, "available_models": models},
            )
            if not model_present:
                overall_healthy = False
        else:
            overall_healthy = False
    except Exception as e:
        ollama_status.details["error"] = str(e)
        overall_healthy = False

    # 2. Probe Qdrant
    qdrant_status = ServiceStatus(
        connected=False, status="unreachable", details={"url": QDRANT_BASE_URL}
    )
    try:
        resp = requests.get(
            f"{QDRANT_BASE_URL}/collections/{QDRANT_COLLECTION}", timeout=3.0
        )
        if resp.status_code == 200:
            data = resp.json().get("result", {})
            points_count = data.get("points_count", 0)
            qdrant_status = ServiceStatus(
                connected=True,
                status="online",
                details={
                    "collection": QDRANT_COLLECTION,
                    "points_count": points_count,
                    "vectors_count": data.get("vectors_count", 0),
                },
            )
        else:
            overall_healthy = False
            qdrant_status.status = f"collection_{QDRANT_COLLECTION}_not_found"
    except Exception as e:
        qdrant_status.details["error"] = str(e)
        overall_healthy = False

    # 3. Probe LangSmith
    langsmith_key = os.getenv("LANGSMITH_API_KEY", "")
    langsmith_project = os.getenv("LANGSMITH_PROJECT", "agentic-rag-shp")
    langsmith_endpoint = os.getenv(
        "LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"
    )

    ls_connected = bool(langsmith_key and len(langsmith_key) > 10)
    langsmith_status = ServiceStatus(
        connected=ls_connected,
        status="configured" if ls_connected else "disabled",
        details={
            "project": langsmith_project,
            "endpoint": langsmith_endpoint,
            "tracing_enabled": os.getenv("LANGSMITH_TRACING", "").lower() == "true",
        },
    )

    return HealthResponse(
        status="healthy" if overall_healthy else "degraded",
        ollama=ollama_status,
        qdrant=qdrant_status,
        langsmith=langsmith_status,
        timestamp=now_str,
    )


# ---------------------------------------------------------------------------
# Core Chat Execution Route
# ---------------------------------------------------------------------------
@app.post("/api/v1/chat", response_model=ChatResponse, tags=["Chat"])
def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """
    Executes an engineering query through the compiled multi-agent LangGraph workflow.
    """
    if not getattr(app.state, "graph", None):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LangGraph multi-agent workflow is not compiled. Check server logs.",
        )

    session_id = request.session_id or str(uuid.uuid4())
    query = request.query.strip()
    logger.info(f"[API: chat] Executing query: '{query}' | session_id={session_id}")

    initial_state: AgentState = {
        "query": query,
        "user_query": query,
        "original_query": query,
        "messages": [],
        "cycle_count": 0,
        "sub_agent_results": {},
        "all_retrieved_chunks": [],
        "documents": [],
    }

    config = {
        "configurable": {
            "thread_id": session_id,
        }
    }

    t0 = time.perf_counter()
    try:
        result = app.state.graph.invoke(initial_state, config=config)
    except Exception as e:
        logger.error(f"[API: chat ERROR] Execution failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent workflow execution error: {str(e)}",
        )
    duration_s = time.perf_counter() - t0

    response_text = (
        result.get("final_response") or "No response could be formulated by the system."
    )
    routing_decision = result.get("routing_decision", "rag_retrieval")
    cb_tripped = bool(result.get("circuit_breaker_tripped", False))

    # Parse and format citations
    raw_citations = result.get("citations") or []
    documents = result.get("documents") or result.get("all_retrieved_chunks") or []

    citation_items: list[CitationItem] = []
    seen_sources: set[str] = set()

    # 1. Add structured citation items from document chunks
    for doc in documents:
        src = doc.get("source_file", "Unknown Document")
        if src not in seen_sources:
            seen_sources.add(src)
            snippet = doc.get("content", "")[:280].strip().replace("\n", " ")
            citation_items.append(
                CitationItem(
                    source_file=src,
                    breadcrumb=doc.get("breadcrumb", "General Section"),
                    score=round(float(doc.get("score", 0.0)), 3),
                    snippet=snippet,
                )
            )

    # 2. Add any remaining raw citation strings
    for c in raw_citations:
        if isinstance(c, str) and c not in seen_sources:
            seen_sources.add(c)
            citation_items.append(
                CitationItem(
                    source_file=c,
                    breadcrumb="Reference",
                    score=1.0,
                    snippet=f"Referenced by synthesis engine: {c}",
                )
            )

    telemetry = TelemetryData(
        latency_seconds=round(duration_s, 2),
        cycle_count=result.get("cycle_count", 0),
        routing_decision=routing_decision,
        is_retrieval_sufficient=result.get("is_retrieval_sufficient"),
        is_grounded=result.get("is_grounded"),
        circuit_breaker_tripped=cb_tripped,
        total_chunks_retrieved=len(result.get("all_retrieved_chunks", [])),
    )

    logger.info(
        f"[API: chat SUCCESS] Finished in {duration_s:.2f}s | Decision={routing_decision} | "
        f"Citations={len(citation_items)} | CB_Tripped={cb_tripped}"
    )

    # Format full retrieved chunks for evaluation & transparency
    formatted_chunks: list[dict[str, Any]] = []
    for doc in documents:
        formatted_chunks.append({
            "chunk_id": doc.get("chunk_id", ""),
            "source_file": doc.get("source_file", "Unknown"),
            "breadcrumb": doc.get("breadcrumb", "General"),
            "score": round(float(doc.get("score", 0.0)), 4),
            "content": doc.get("content", ""),
        })

    return ChatResponse(
        query=query,
        session_id=session_id,
        response=response_text,
        routing_decision=routing_decision,
        circuit_breaker_tripped=cb_tripped,
        citations=citation_items,
        telemetry=telemetry,
        retrieved_chunks=formatted_chunks,
    )


# ---------------------------------------------------------------------------
# Document Upload Endpoint
# ---------------------------------------------------------------------------
@app.post(
    "/api/v1/documents/upload",
    response_model=DocumentUploadResponse,
    tags=["Documents"],
)
async def upload_document(file: UploadFile = File(...)) -> DocumentUploadResponse:
    """
    Accepts plant manuals, specifications, and DPR PDFs/DOCX and saves them
    into the `data/` directory for subsequent vector ingestion.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    clean_filename = Path(file.filename or "uploaded_doc.pdf").name
    dest_path = DATA_DIR / clean_filename

    logger.info(f"[API: upload] Receiving file: '{clean_filename}'")
    try:
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        size_bytes = dest_path.stat().st_size
    except Exception as e:
        logger.error(f"[API: upload ERROR] Failed saving file: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save uploaded file: {str(e)}",
        )

    logger.success(
        f"[API: upload SUCCESS] Saved '{clean_filename}' ({size_bytes} bytes)."
    )
    return DocumentUploadResponse(
        filename=clean_filename,
        size_bytes=size_bytes,
        status="uploaded",
        message=f"File '{clean_filename}' successfully stored in data/. Run ingestion to index into Qdrant.",
    )


# ---------------------------------------------------------------------------
# Root Welcome Route
# ---------------------------------------------------------------------------
@app.get("/", tags=["Root"])
def root():
    """Root metadata route."""
    return {
        "service": "Small Hydropower Plant (SHP) Agentic RAG API Server",
        "status": "online",
        "docs_url": "/docs",
        "health_url": "/api/v1/health",
        "chat_url": "/api/v1/chat",
    }


# ---------------------------------------------------------------------------
# Direct Execution Runner
# ---------------------------------------------------------------------------
def main():
    """Starts the FastAPI Uvicorn server with hot-reloading."""
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", os.getenv("PORT", "8600")))

    logger.info(
        f"Starting SHP Agentic RAG FastAPI server on http://{host}:{port} (Docs: http://{host}:{port}/docs)"
    )
    uvicorn.run("src.api.app:app", host=host, port=port, reload=True)


if __name__ == "__main__":
    main()
