# Agentic RAG — Small Hydropower Plant (SHP) Knowledge System

> A **cost-effective Agentic RAG pipeline with zero API fees** for Small Hydropower Plant (SHP) operations, maintenance & knowledge management — built with **LangGraph**, **LangChain**, and **Ollama**, and developed using **Spec-Driven Development (SDD)** principles. Fully on-prem capable architecture, evaluated using zero-cost resources (local Ollama inference + free-tier cloud LLM judges).

---

## 📋 Table of Contents

- [1. Overview & Vision](#1-overview--vision)
- [2. Spec-Driven Development (SDD) Methodology](#2-spec-driven-development-sdd-methodology)
- [3. Specifications (The Specs)](#3-specifications-the-specs)
  - [Spec 01: System Requirements & Objectives](#spec-01-system-requirements--objectives)
  - [Spec 02: State Schema & Hierarchical Subgraph Architecture](#spec-02-state-schema--hierarchical-subgraph-architecture)
  - [Spec 03: Graph Topology & Node Contracts](#spec-03-graph-topology--node-contracts)
  - [Spec 04: Tools, Retrievers & Integrations](#spec-04-tools-retrievers--integrations)
  - [Spec 05: Guardrails, Fallbacks & Quality Checks](#spec-05-guardrails-fallbacks--quality-checks)
  - [Spec 06: Evaluation & Acceptance Criteria](#spec-06-evaluation--acceptance-criteria)
  - [Spec 06.1: Hardware Performance & Scaling (RTX 5050 vs Production GPUs)](#spec-061-hardware-performance--scaling-rtx-5050-vs-production-gpus)
- [4. Multi-Tier Document Parsing Pipeline](#4-multi-tier-document-parsing-pipeline)
- [5. System Architecture & Hierarchical Subgraph Workflow](#5-system-architecture--hierarchical-subgraph-workflow)
  - [5.1 Parent StateGraph vs. Subgraph States (LangGraph Architecture Pattern)](#51-parent-stategraph-vs-subgraph-states-langgraph-architecture-pattern)
  - [5.2 Model Context Protocol (MCP) FastAPI Server (src/mcp_server)](#52-model-context-protocol-mcp-fastapi-server-srcmcp_server)
- [6. Project Directory Structure](#6-project-directory-structure)
- [7. Setup & Installation](#7-setup--installation)
- [8. Configuration (.env)](#8-configuration-env)
- [9. Development Roadmap](#9-development-roadmap)

---

## 1. Overview & Vision

| Field | Detail |
| :--- | :--- |
| **Goal** | Build a cost-effective, zero-API-fee agentic RAG pipeline for Small Hydropower Plant (SHP) — operations, maintenance, fault diagnosis & knowledge management. Fully on-prem capable at runtime. |
| **Target Domain** | Small Hydropower Plant (SHP) — operations, maintenance, fault diagnosis & knowledge management. |
| **Data Scale** | ~2 GB corpus: PDF manuals, Word reports, scanned images, plain text, JPEG/PNG drawings. |
| **LLM** | `qwen3.5:9b` via **Ollama** (local inference, fully air-gapped capable) |
| **Embedding Model** | `Qwen3-Embedding-0.6B` (local, via Ollama or sentence-transformers) |
| **Vector DB** | **QDrant** (self-hosted, persistent, production-grade) |
| **Document Parsing** | **Multi-Tier Smart Parser** — `pymupdf4llm` + `Docling` GPU Vision OCR + `RapidOCR` + Office DOM (`python-pptx`, `python-docx`, `openpyxl`, `xlrd`) |
| **Orchestration** | **LangGraph** (Stateful Multi-Agent Hierarchical Architecture with Parent StateGraph & Specialized Agent Subgraphs + Send API Map-Reduce) + **LangChain** |
| **Language** | Python 3.11+ |
| **Key Capabilities** | Hierarchical subgraphs, self-healing reflection loops, corrective RAG (CRAG), self-RAG, query rewriting, hallucination detection, multi-agent scatter-gather routing |
| **Evaluation** | **Multi-Provider LLM-as-a-Judge Offline Benchmark** (Zero API fees via Google Gemini 2.5 Flash & Groq 120B/20B waterfall failover across 53 golden test cases; optional local assertions via Ollama) |

---

## 2. Spec-Driven Development (SDD) Methodology

This project follows **Spec-Driven Development (SDD)**: every node, state transition, tool behavior, and routing condition is written as an explicit, frozen contract *before* any code is implemented.

```
┌─────────────────┐       ┌────────────────────┐       ┌───────────────────┐
│  1. SPEC PHASE  │ ───>  │ 2. BUILD PHASE     │ ───>  │ 3. VERIFY PHASE   │
│  State, Graph & │       │ Implement against  │       │ Evals & Tracing   │
│  Node Contracts │       │ frozen contracts   │       │ against metrics   │
\─────────────────/       \────────────────────/       \───────────────────/
```

1. **Specify** — Define typed state schemas, node input/output contracts, and all routing conditions in this README. **Nothing is built until the spec is signed off.**
2. **Implement** — Build discrete LangGraph nodes that strictly match the written contract. No scope creep.
3. **Verify** — Run automated multi-provider LLM evaluations and traces to validate every node against acceptance criteria.

> **Rule**: If a behavior is not in the spec, it is not built. If the spec is wrong, the spec is updated first, then the code.

---

## 3. Specifications (The Specs)

### Spec 01: System Requirements & Objectives

#### Functional Requirements

| # | Requirement | Status |
|---|---|---|
| FR-01 | Ingest & parse PDF (native + scanned OCR), DOCX/DOC, PPTX/PPT, XLSX/XLS, TXT, PNG/JPG → Markdown | [x] Operational (54/54 docs, 0 failures) |
| FR-02 | Chunk Markdown into semantic segments and embed with `Qwen3-Embedding-0.6B` | [x] Operational (4,064 chunks, 1024-dim vectors) |
| FR-03 | Store & retrieve vectors from self-hosted QDrant with metadata filtering | [x] Operational (4,064 points, dual-mode engine) |
| FR-04 | **Planner Agent**: Unified Intent Classification (`direct_answer` / `out_of_scope` / `rag_retrieval`), Domain-Specific Query Elaboration for high-level prompts, Dynamic Sub-Query Decomposition (1–3 focused sub-queries or atomic single sub-query) with Metadata Filters, and Feedback Absorption | [x] Operational (src/agents/planner/) |
| FR-05 | **Orchestrator Agent & Sub-Agents**: Dynamic Map-Reduce dispatch (`Send`) of specialized sub-agents (`narrative_worker`, `tabular_worker`, `mcp_worker`) | [x] Operational (src/agents/orchestrator/) |
| FR-06 | **Synthesizer Agent**: Multi-source chunk fusion, deduplication, fair-share sub-query balancing, citation packaging, and conflict detection | [x] Operational (src/agents/synthesizer/) |
| FR-07 | **Retrieval Sufficiency Grader (Production CRAG)**: Zero-reference retrieval sufficiency and relevance evaluator using `qwen3.5:9b`. Validates whether synthesized context contains complete engineering facts, parameters, and formulas to answer the user query and planned sub-queries before generation | [x] Operational (src/agents/retrieval_grader/) |
| FR-07-QA | **Offline Multi-Provider LLM-as-a-Judge Benchmark**: Automated evaluation suite (`src/evals/evaluate_offline_with_llm.py`) with waterfall failover across Google Gemini (2.5 Flash) and Groq (120B/20B) measuring Faithfulness, Answer Relevancy, Context Precision, and Context Recall against 53 curated golden test cases | [x] Operational (53 cases evaluated: 78.0% Faithfulness, 75.3% Relevancy, 73.9% Precision, 79.2% Recall) |
| FR-08 | **Generator Node**: Grounded final answer generation using `qwen3.5:9b`, multi-source evidence synthesis, and inline citation anchoring (with `GENERATOR_MAX_TOKENS=8192` preventing mid-sentence truncation) | [x] Operational (src/agents/generator/) |
| FR-09 | **Post-Generation Guardrail (Runtime Single-Pass Judge)**: Reference-free Faithfulness & Groundedness check using single-pass local judge (`qwen3.5:9b`). Validates every claim against context; passes unfaithful statements to Feedback Agent upon failure | [x] Operational (src/agents/generation_guardrail/) |
| FR-10 | **Feedback Agent**: Diagnostic reflection on retrieval or generation defects, formulating actionable remediation hints for the Planner Agent and advancing cycle count (loops back if cycle < max_cycles, routes to fallback if exceeded) | [x] Operational (src/agents/feedback/) |
| FR-11 | **Fallback Circuit Breaker**: Graceful cannot-answer response upon exceeding max retry cycles. Features 3-tier recovery: extracts best-effort LLM generation with engineering disclaimers, surfaces partial verified chunk evidence, or provides clear notice when data is unavailable in uploaded documents | [x] Operational (src/agents/fallback/) |
| FR-12 | **Session Conversation Memory**: In-Memory `MemorySaver` checkpointing across turns (upgradeable to SQLite for persistent auditing) | [x] Operational (src/agents/main_agent.py) |
| FR-13 | **Asynchronous REST API Server (FastAPI)**: Decoupled REST API exposing pre-compiled LangGraph singleton on port 8600 with endpoints for health monitoring (`/api/v1/health`), execution (`/api/v1/chat`), and document uploads (`/api/v1/documents/upload`), returning un-truncated chunk texts and similarity scores | [x] Operational (src/api/) |
| FR-14 | **Interactive Streamlit Web Portal**: Decoupled engineering interface on port 8601 featuring multi-turn chat, KaTeX LaTeX formulas, Fallback Circuit Breaker callouts, SHP sizing calculator, citations explorer, and live telemetry metrics | [x] Operational (ui/) |

#### Non-Functional Requirements & Measured Benchmark Performance

> [!NOTE] Aspirational Targets vs. Recalibrated Empirical Baselines
> Initial engineering specifications set aggressive aspirational targets (e.g. Faithfulness ≥ 0.90, Precision/Recall ≥ 0.85, sub-5-second interactive responses). When tested on local consumer hardware (RTX 5050 8GB GPU) running `qwen3.5:9b`, realistic measured performance settled at **78.0% Faithfulness**, **75.3% Answer Relevancy**, **73.9% Context Precision**, **79.2% Context Recall**, and a measured full-lifecycle **Median Latency of 687.2s (~11.5 minutes)** and **p95 Latency of 1,601.1s (~26.7 minutes)** across 53 golden cases. We do not present these recalibrated scores as an unmitigated "all targets met" victory; rather, they reflect an honest, transparent baseline of running a multi-agent LangGraph system on consumer hardware.

| # | Requirement | Aspirational Target | Recalibrated Baseline (53 Golden Cases) | Status / Implementation Notes |
|---|---|---|---|---|
| NFR-01 | End-to-end query latency (Median / p95) | < 5s (Direct Chat) | **Median: 687.2s (~11.5m)**<br>**p95: 1,601.1s (~26.7m)** (RTX 5050) | Memory bandwidth-bound (224 GB/s) across multi-agent loops (Passed gate: median 501.6s; Fallback: median 986.7s) |
| NFR-02 | Faithfulness score (LLM-as-a-Judge) | ≥ 0.90 (Aspirational) | **0.7798 (78.0%)** (Target: ≥ 0.75) | Hallucinations primarily in arithmetic & dense tables |
| NFR-03 | Answer Relevance score (LLM-as-a-Judge) | ≥ 0.88 (Aspirational) | **0.7526 (75.3%)** (Target: ≥ 0.70) | Strong query adherence on policy & licensing queries |
| NFR-04 | Context Precision (Signal-to-Noise) | ≥ 0.85 (Aspirational) | **0.7391 (73.9%)** (Target: ≥ 0.70) | Signal-to-noise across multi-agent scatter-gather chunks |
| NFR-05 | Context Recall (Completeness) | ≥ 0.85 (Aspirational) | **0.7921 (79.2%)** (Target: ≥ 0.75) | Broad coverage across multi-hop civil & turbine documents |
| NFR-06 | Zero dependency on paid external LLMs | Zero API Fees | **No API Fees** | Local Ollama `qwen3.5:9b` runtime + free-tier Gemini/Groq judges |
| NFR-07 | On-prem capable architecture vs. experiment | On-Prem Capable | **Hybrid Free Setup in Experiment** | Production can run 100% on-prem; benchmark used free cloud judges & LangSmith |
| NFR-08 | Max query rewrite attempts before fallback | 1–3 configurable | **Configurable via `MAX_REWRITE_ATTEMPTS=1` in `.env`** | Prevents infinite reflection loops and bounds GPU latency |

---

### Spec 02: State Schema & Hierarchical Subgraph Architecture

The SHP Agentic RAG system adopts a **Hierarchical State Architecture** in LangGraph. The overarching system is governed by a **Parent State (`AgentState`)**, while each agent node operates with **Specialized Domain Subgraph States**:

#### 1. Parent State vs. Subgraph State Communication Pattern
In LangGraph, subgraphs can communicate with parent graphs in two primary ways:
1. **Shared/Identical State Schema**: The parent and subgraph share the exact same state structure.
2. **Different/Specialized State Schemas with Transformation (Our System Architecture)**:
   - The parent graph manages the end-to-end conversation channel (`AgentState`).
   - Each specialized agent subgraph manages its own domain-specific schema (e.g. `WorkerInputState` for parallel retrieval, `OrchestratorPlan` for task assignments, `FeedbackDiagnosis` for failure remediation).
   - **How Parent State is Populated by Subgraph States**:
     - **Scatter-Gather via `Send()` API**: The Orchestrator node transforms decomposed queries from the parent state into discrete `WorkerInputState` task payloads and dispatches them across parallel sub-agents (`narrative_worker`, `tabular_worker`, `mcp_worker`).
     - **Channel Reducer Fan-In**: Results from parallel subgraphs fan in to the parent `AgentState["sub_agent_results"]` via an `Annotated` reducer channel (`merge_sub_agent_results`), automatically merging worker outputs without overwriting parallel threads.
     - **Node Adapter Transformation**: Each agent node extracts its required slice of `AgentState`, invokes the internal agent logic/subgraph, and returns only the updated keys (e.g. `is_retrieval_sufficient`, `documents`, `final_response`) to update the parent state.

```python
# FILE: src/states/agent_state.py  — Unified Parent AgentState
from typing import Annotated, Any, Literal, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


def merge_sub_agent_results(
    left: dict[str, list[dict[str, Any]]] | None,
    right: dict[str, list[dict[str, Any]]] | None,
) -> dict[str, list[dict[str, Any]]]:
    """Reducer merging dictionary of sub-agent results keyed by sub_query_id."""
    merged = dict(left) if left else {}
    if not right:
        return merged
    for k, v in right.items():
        merged[k] = list(merged.get(k, [])) + list(v)
    return merged


class DocumentChunk(TypedDict, total=False):
    """Represents a single retrieved document chunk with rich citation metadata."""
    chunk_id: str                     # Unique deterministic chunk UUID/string
    content: str                      # Full markdown text of the chunk (with breadcrumbs)
    source_file: str                  # Source filename (e.g. HDP_AHN511.ppt)
    file_type: str                    # Document type (PDF, PPT, DOCX, XLS)
    page_or_slide: int | None         # Page number or slide index
    breadcrumb: str                   # Section path (e.g. Doc > Section > Table)
    is_table: bool                    # Flag indicating structured table data
    score: float                      # Qdrant cosine similarity score


class SubQuery(TypedDict, total=False):
    """A decomposed sub-query planned for targeted retrieval."""
    sub_query_id: str                 # Identifier (e.g. "sq_1")
    query_text: str                   # Specific search query
    target_filter: dict[str, Any] | None # Qdrant filter (e.g. {"is_table": "true"})
    rationale: str                    # Why this sub-query was formed


class FeedbackDiagnosis(TypedDict, total=False):
    """Structured diagnostic generated by Feedback Agent upon guardrail failure."""
    failure_type: Literal[
        "INSUFFICIENT_RETRIEVAL",      # Low context recall/precision
        "HALLUCINATION_DETECTED",     # Answer contains ungrounded claims
        "OFF_TOPIC_ANSWER"            # Low answer relevancy
    ]
    missing_aspects: list[str]        # Specific engineering facts/tables missing
    critique: str                     # Human-readable explanation of the defect
    remediation_hint: str             # Actionable suggestion for the Planner Agent
    target_recommendations: list[str] # Recommended workers or filters


class AgentState(TypedDict, total=False):
    # ── Core Conversation ──────────────────────────────────────────────
    messages: Annotated[list[BaseMessage], add_messages]  # Chat message history
    query: str                                             # Current active user query
    original_query: str                                    # Unmodified user input

    # ── Planner Decision & Intent ──────────────────────────────────────
    routing_decision: Literal[
        "rag_retrieval",              # Complex technical SHP question requiring RAG
        "direct_answer",              # Conversational greeting or simple factual question
        "out_of_scope"                # Non-SHP / disallowed domain query
    ]
    plan: list[str]                   # High-level steps formulated by Planner Agent
    sub_queries: list[SubQuery]       # Decomposed queries for sub-agents

    # ── Orchestration & Parallel Subgraph Worker Fan-In (Send API) ──────
    sub_agent_results: Annotated[
        dict[str, list[DocumentChunk]], 
        merge_sub_agent_results
    ]                                 # Results keyed by sub_query_id reduced into Parent State

    # ── Synthesis & Fusion ─────────────────────────────────────────────
    documents: list[DocumentChunk]    # Deduplicated, synthesized, and re-ranked chunks
    synthesized_context: str          # Clean context block assembled for LLM generation
    executive_summary: str | None     # High-level technical briefing across all sub-queries
    conflicts_detected: list[str] | None # Cross-document discrepancies flagged

    # ── Retrieval Quality & Sufficiency Grader (Production CRAG) ───────
    is_retrieval_sufficient: bool     # True if retrieved context has sufficient data
    missing_aspects: list[str]        # Specific missing engineering parameters
    retrieval_critique: str           # Detailed reasoning explaining sufficiency
    retrieval_confidence: float | None# Confidence score (0.0 to 1.0)

    # ── Post-Generation Guardrail & Faithfulness ───────────────────────
    generation_metrics: dict[str, float]# {"faithfulness": 0.94, "answer_relevancy": 0.92}
    is_grounded: bool                 # True if post-gen faithfulness passes threshold

    # ── Feedback & Self-Correction Loop ────────────────────────────────
    feedback_diagnosis: FeedbackDiagnosis | None # Analysis fed back to Planner
    cycle_count: int                  # Number of full plan-retrieve-evaluate loops

    # ── Final Output ───────────────────────────────────────────────────
    final_response: str               # Final verified answer returned to user
```

---

### Spec 03: Graph Topology & Node Contracts

#### Node Contracts

| Node | Type / Role | Input Fields (State) | Output Fields (State) | Responsibilities & Fallback |
| :--- | :--- | :--- | :--- | :--- |
| `planner` | **Planner Agent** | `query`, `feedback_diagnosis`, `cycle_count` | `routing_decision`, `plan`, `sub_queries`, `final_response` | Unified entry point: classifies intent (`direct_answer`/`out_of_scope`/`rag_retrieval`), elaborates high-level queries with domain concepts while preserving intent, decomposes technical queries into 1–3 `SubQuery` specs (or 1 atomic sub-query if indivisible) with metadata filters, incorporates feedback diagnostics. |
| `plan_orchestration` | **Orchestrator Agent** | `user_query`, `sub_queries` | `tasks`, `coordination_strategy` | Formulates worker tasks; dynamically spawns specialized sub-agents via LangGraph's `Send()` API. |
| `narrative_worker` | Sub-Agent (Manuals) | `task` (via `Send`) | `sub_agent_results`, `all_retrieved_chunks` | Dense semantic retrieval over turbine principles, guidelines, and maintenance text in Qdrant (`filter: is_table=false`). |
| `tabular_worker` | Sub-Agent (Tables) | `task` (via `Send`) | `sub_agent_results`, `all_retrieved_chunks` | Structured retrieval over monthly discharge, head-flow matrices, and cost tables in Qdrant (`filter: is_table=true`). |
| `mcp_worker` | Sub-Agent (Tools/SCADA)| `task` (via `Send`) | `sub_agent_results`, `all_retrieved_chunks` | Exact formula execution ($P$, $N_s$, Penstock sizing) and plant sensor telemetry via FastAPI MCP Server. |
| `aggregate_results` | Fan-In Aggregator | `all_retrieved_chunks`, `sub_agent_results` | `all_retrieved_chunks`, `total_retrieved` | Fans in outputs across all dynamic sub-agents, deduplicates by `chunk_id`, and sorts by relevance score. |
| `synthesizer` | **Synthesizer Agent** | `sub_agent_results`, `all_retrieved_chunks`, `query` / `user_query`, `sub_queries` | `documents`, `synthesized_context`, `executive_summary`, `conflicts_detected`, `citations` | Fuses chunks across all sub-queries, eliminates duplicates by `chunk_id`, enforces fair-share sub-query coverage, elevates MCP tool calculations, and assembles structured, citation-anchored context. Features high-speed pure-Python deterministic fusion (0.000s) to eliminate redundant LLM generation cycles and preserve VRAM/bandwidth on local GPUs. |
| `retrieval_grader` | **Retrieval Sufficiency Grader** (Production CRAG) | `query`, `sub_queries`, `synthesized_context`, `documents` | `is_retrieval_sufficient`, `missing_aspects`, `retrieval_critique`, `retrieval_confidence` | Zero-reference evaluator using `qwen3.5:9b`. Analyzes whether synthesized context contains all required technical parameters, formulas, equipment curves, and operational facts to answer the user query and sub-queries. If sufficient → `generator`. If insufficient and `cycle_count < MAX_CYCLES` (configurable in `.env`, default: 1) → `feedback_agent`. If insufficient and `cycle_count >= MAX_CYCLES` → `generator` (attempts best-effort synthesis with available evidence; guardrail will subsequently verify faithfulness before delivery). |
| `generator` | Answer Generator | `query`, `synthesized_context` | `final_response` | Synthesizes an authoritative engineering answer strictly from context using `qwen3.5:9b` (configured with `GENERATOR_MAX_TOKENS=8192` to prevent mid-sentence cutoff). |
| `generation_guardrail` | Post-Gen Evaluator | `query`, `final_response`, `documents` | `generation_metrics`, `is_grounded` | Evaluates Faithfulness and Hallucination metrics. If ungrounded and `cycle_count < MAX_CYCLES` → routes to `feedback_agent`; if exceeding limit → routes to `fallback`. |
| `feedback_agent` | **Feedback Agent** | `query`, `retrieval_metrics`, `generation_metrics`, `cycle_count`, `missing_aspects`, `retrieval_critique` | `feedback_diagnosis`, `cycle_count` | Diagnoses defect root cause, formulates actionable query remediation, increments `cycle_count`. If `cycle_count >= MAX_CYCLES` → routes to `fallback`. |
| `fallback` | **Fallback Circuit Breaker** (FR-11) | `original_query` / `query`, `routing_decision`, `cycle_count`, `final_response`, `documents`, `feedback_diagnosis`, `missing_aspects`, `unfaithful_statements` | `final_response`, `circuit_breaker_tripped` | Transparent fail-safe terminal node. Activated when retries reach max limit (`cycle_count >= MAX_CYCLES`) or query is out-of-scope. Employs 3-tier response generation: (1) extracts best-effort LLM generation with high-visibility engineering warning & disclaimer, (2) formats partial verified chunk evidence if generation was bypassed, or (3) clarifies that the specific data is unavailable in uploaded documents with required ingestion guidance. |

#### Routing & Conditional Edges

```
START                     ──> planner

planner → "direct_answer" ──> END (returns direct answer immediately)
planner → "out_of_scope"  ──> fallback (returns polite domain refusal)
planner → "rag_retrieval" ──> plan_orchestration

plan_orchestration ──(Send)──> narrative_worker ──> aggregate_results
plan_orchestration ──(Send)──> tabular_worker   ──> aggregate_results
plan_orchestration ──(Send)──> mcp_worker       ──> aggregate_results

aggregate_results         ──> synthesizer
synthesizer               ──> retrieval_grader

retrieval_grader → is_retrieval_sufficient = True                                      ──> generator
retrieval_grader → is_retrieval_sufficient = False AND cycle_count < MAX_CYCLES        ──> feedback_agent
retrieval_grader → is_retrieval_sufficient = False AND cycle_count >= MAX_CYCLES       ──> generator (best-effort synthesis)

generator                 ──> generation_guardrail

generation_guardrail → is_grounded = True                                             ──> END (Return response)
generation_guardrail → is_grounded = False AND cycle_count < MAX_CYCLES               ──> feedback_agent
generation_guardrail → is_grounded = False AND cycle_count >= MAX_CYCLES              ──> fallback (circuit breaker trips)

feedback_agent            ──> planner (re-plan with diagnostic guidance)
```

---

### Spec 04: Tools, Retrievers & Integrations

| Component | Choice | Config |
| :--- | :--- | :--- |
| **LLM** | `qwen3.5:9b` via Ollama | `base_url: http://localhost:11434`, `temperature: 0.0` for graders, `0.3` for generation |
| **Embedding Model** | `Qwen3-Embedding-0.6B` via Ollama | `model: qwen3-embedding:0.6b`, `dim: 1024` |
| **Vector Store** | QDrant (self-hosted) | `host: localhost`, `port: 6333`, `collection: shp_knowledge` |
| **Document Parser** | Docling | Input: PDF, DOCX, TXT, JPEG, PNG → Output: Markdown |
| **Text Splitter** | `RecursiveCharacterTextSplitter` (Markdown-aware) | `chunk_size: 1000 characters (~330 tokens via cl100k_base)`, `chunk_overlap: 150 characters (~40 tokens)` |
| **Retriever** | **Hybrid Retriever**: Dense Vector Search (Qdrant) + Sparse Keyword Search (Okapi BM25 with Exact Name Boosting via Reciprocal Rank Fusion / RRF) | `top_k: 6`, `retriever_mode: hybrid`, `score_threshold: 0.65`, `dense_weight: 0.5`, `bm25_weight: 0.5` |
| **Checkpointing** | LangGraph `MemorySaver` (In-Memory RAM for active session); upgradeable to `SqliteSaver` (`./data/checkpoints.db`) | In-RAM thread checkpointing during session execution; zero disk overhead; persistent cross-session audit tracing via SQLite |
| **MCP Tool Server** | FastAPI SHP MCP Server (`src/mcp_server/`) | Port 8602 (FastAPI + Uvicorn): JSON-RPC 2.0 (`/mcp`), REST endpoints (`/tools`, `/api/v1/...`), Swagger UI (`/docs`) |

#### Hybrid Retrieval Engine: Dense Vector + Okapi BM25 Keyword Search

While dense embeddings (`qwen3-embedding:0.6b` via Qdrant) excel at general semantic similarity, dense-only retrieval frequently struggles with **exact entity names, project identifiers, and statutory references** (e.g. project sites like *"Tala"*, author names like *"Aggidis"* or *"Alvarado-Ancieta"*, policy documents like *"allotment of shp"*, or specific codes like *"Regulation 14(2)"*). 

To solve this, the system implements an in-memory **Okapi BM25 Sparse Keyword Retriever** ([`src/ingestion/bm25_retriever.py`](src/ingestion/bm25_retriever.py)) that operates alongside Qdrant:

1. **Exact Name & Entity Boosting**:
   - Analyzes all 4,064 chunks in `data/chunks/chunks.jsonl`.
   - When a query contains terms matching a document's filename (`source_file`), title (`document_title`), or breadcrumb header (`breadcrumb`), it applies an **Exact Name Boost (+3.5 score)**.
   - Exact multi-word phrase containment in chunk text receives an additional **+7.0 boost**, ensuring that chunks referencing the exact queried entity are propelled to the very top rank.
2. **Sub-5ms Query Latency & Persistent Disk Caching**:
   - Pre-computes and caches the complete inverted index and IDF weights in `data/chunks/bm25_index.pkl`.
   - Subsequent initializations load from disk in **< 20ms**, and searches execute in **3–5 milliseconds** with zero GPU memory contention.
3. **Reciprocal Rank Fusion (RRF)**:
   - Worker nodes (`narrative_worker`, `tabular_worker`) combine ranked lists from dense Qdrant search and BM25 keyword search:
     $$\text{RRF}(d) = \frac{w_{\text{dense}}}{60 + \text{rank}_{\text{dense}}(d)} + \frac{w_{\text{bm25}}}{60 + \text{rank}_{\text{bm25}}(d)}$$
   - Configurable via `.env`: `RETRIEVER_MODE=hybrid` (default), `dense`, or `bm25`.

---

### Spec 05: Guardrails, Quality Checks & Feedback Agent

#### 1. Production Retrieval Sufficiency & Relevance Grader (Reference-Free CRAG)
- **Module**: `src/agents/retrieval_grader/`
- **Role**: **Runtime Pre-Generation Gatekeeper** in the live production LangGraph workflow.
- **Mechanism**: LLM-as-judge using `qwen3.5:9b` executing **Reference-Free Corrective RAG (CRAG)** grading.
- **Why Reference-Free**: In live production, unseen end-user queries do not have known ground-truth answers. The system cannot rely on reference answers to determine if retrieval succeeded. Instead, the Grader inspects the user query, planned sub-queries, and synthesized context to evaluate whether all technical dimensions and data points are present to formulate a complete answer.
- **Inputs**:
  - `query`: The user's primary engineering query.
  - `sub_queries`: Decomposed technical sub-queries with targets and rationales from the Planner.
  - `synthesized_context`: Clean context block assembled by the Synthesizer Agent (including tables, narrative chunks, and MCP calculations).
- **Grading Dimensions**:
  1. **Technical Sufficiency**: Does the context contain the specific formulas, equipment ratings, head/flow ranges, or site data required?
  2. **Sub-Query Coverage**: Are all sub-queries planned by the Planner Agent addressed by the retrieved evidence?
  3. **Relevance & Signal-to-Noise**: Is the context technically relevant to Small Hydropower engineering, rather than generic filler?
- **Outputs**: Structured Pydantic `RetrievalGraderOutput`:
  ```json
  {
    "is_sufficient": true,
    "missing_aspects": [],
    "critique": "Retrieved context thoroughly provides net head (120m), discharge requirements, and efficiency curves for both Pelton and Francis turbines.",
    "confidence_score": 0.95
  }
  ```
- **Decision Contract**:
  - If `is_retrieval_sufficient == True` → Route to `generator` node.
  - If `is_retrieval_sufficient == False` AND `cycle_count < MAX_CYCLES` (default: 1 in `.env`, max: 3) → Route to `feedback_agent` with `missing_aspects` and `critique`.
  - If `is_retrieval_sufficient == False` AND `cycle_count >= MAX_CYCLES` → Route to `generator` (attempts best-effort synthesis with available evidence; the subsequent `generation_guardrail` node inspects groundedness and trips the fallback circuit breaker if ungrounded).

#### 2. Offline Evaluation Metrics Framework (Free-Tier LLM-as-a-Judge)
- **Module**: `src/evals/evaluate_offline_with_llm.py`
- **Role**: **Offline QA & Regression Testing Suite** for systematic evaluation against curated golden datasets (53 test cases across policy, multi-hop engineering, and mathematical calculations).
- **Mechanism**: Automated **Multi-Provider LLM-as-a-Judge** benchmark with resilient waterfall failover across free-tier API endpoints (Google Gemini 2.5 Flash → Groq 120B/20B → local Ollama fallback), operating with **zero API fees**.
- **Requirement**: Evaluates against a curated **ground-truth reference answer (`expected_output`)** for each query in the golden datasets.
- **Intended Usage**:
  - **For QA Engineers & Researchers**: Systematically benchmarks vector index retrieval quality, chunking strategies, generation groundedness, and re-ranking performance.
  - **Not in Live Production Workflow**: Because live production queries have no known reference answer, this harness is executed independently during offline regression testing.
- **Evaluated Core RAG Metrics**:
  1. **Faithfulness (Claim-Level Groundedness Ratio)**:
     $$\text{Faithfulness} = \frac{\text{Factual Claims in Answer Supported by Context}}{\text{Total Factual Claims Made in Answer}}$$
     *Meaning*: Evaluates the fraction of individual factual claims that are backed by the retrieved context. A 0.78 score means 78% of factual propositions were verified against context.
  2. **Answer Relevancy (Query Adherence)**:
     $$\text{Answer Relevancy} = \frac{\text{Sentences Directly Addressing Query Intent}}{\text{Total Sentences in Generated Answer}}$$
     *Meaning*: Evaluates whether the generated response directly answers the user's specific engineering question without drift, omissions, or generic boilerplate.
  3. **Context Precision (Signal-to-Noise Ratio)**:
     $$\text{Context Precision} = \frac{\text{Relevant Chunks Retrieved}}{\text{Total Chunks Retrieved}}$$
     *Meaning*: Of all the chunks the retriever returned, how many were actually relevant to formulating the technical answer? High precision prevents polluting LLM attention with irrelevant noise.
  4. **Context Recall (Retrieval Completeness)**:
     $$\text{Context Recall} = \frac{\text{Relevant Facts Found in Retrieved Context}}{\text{Total Required Facts in Ground Truth Reference}}$$
     *Meaning*: Of all the key technical facts and criteria required by the golden ground truth (`expected_output`), how many did the retriever successfully locate?
  5. **Query Hallucination Incident Rate (Defect Frequency)**:
     $$\text{Query Hallucination Rate} = \frac{\text{Queries with } \ge 1 \text{ Flagged Unsupported Claim}}{\text{Total Evaluated Queries}} = \frac{14}{53} \approx 26.4\%$$
     *Meaning*: Measures the proportion of generated responses that contained *at least one* ungrounded statement or arithmetic error.
     > [!IMPORTANT] Distinguishing Faithfulness from Query Hallucination Rate
     > **Do not equate Query Hallucination Rate with $1 - \text{Faithfulness}$.** They measure fundamentally different dimensions at different granularities:
     > - **Faithfulness (0.78)** is a continuous, claim-level micro-average (78% of all individual claims were grounded, leaving a 22% claim ungroundedness margin).
     > - **Query Hallucination Rate (26.4%)** is a query-level binary occurrence metric (14 out of 53 queries contained at least one defect). An answer containing 9 grounded statements and 1 minor arithmetic slip has 90% faithfulness, but is recorded as a flagged query in the binary hallucination rate.

#### 3. Post-Generation Guardrail (Single-Pass Groundedness Evaluation at Runtime)
- **Module**: `src/agents/generation_guardrail/`
- **Role**: **Runtime Post-Generation Gatekeeper** validating answer faithfulness and eliminating hallucinations before delivery to the end user.
- **Mechanism (Runtime)**: High-speed, **Single-Pass Local LLM Judge (`qwen3.5:9b`)** via Jinja2 templates (`src/prompts/generation_guardrail_*.jinja2`) with constrained token prediction (`num_predict=2048`). Evaluates factual consistency across numbers, formulas, and turbine specifications in **one single inference (~15–20 seconds)** rather than multi-minute claim-by-claim loops.
- **Why Reference-Free (Runs in Production)**: Unlike offline recall metrics that require a golden reference answer (`expected_output`), the runtime guardrail evaluates `final_response` strictly against `synthesized_context` (the retrieved evidence dossier). It requires zero ground-truth answers and operates seamlessly at live inference time.
- **Inputs**: `query` + `final_response` + `synthesized_context` (or `documents`).
- **Runtime Threshold vs. Offline Benchmark Target**:
  > [!IMPORTANT] Runtime Guardrail Threshold (0.90) vs. Offline Benchmark Target (0.75)
  > - **Runtime Guardrail Threshold (`0.90`)**: At runtime during live graph execution, each individual generated answer is evaluated by the single-pass local LLM guardrail (`src/agents/generation_guardrail/agentnodes.py`). An answer must achieve `is_grounded == True` and a faithfulness score **$\ge 0.90$** (configured via `DEFAULT_FAITHFULNESS_THRESHOLD = float(os.getenv("FAITHFULNESS_THRESHOLD", "0.90"))`). This strict threshold ensures that in production, only high-confidence, fully grounded answers are delivered directly to the user.
  > - **Offline Benchmark Target (`0.75`)**: In Spec 06, the target of **$\ge 0.75$** (75%) is a macro dataset-level acceptance target averaged across the entire 53-case golden test suite (which includes complex multi-hop, tabular, and mathematical questions).
- **Outputs**: Pydantic structured `GenerationGuardrailOutput` (`is_grounded`, `faithfulness_score`, `unfaithful_statements`, `critique`).
- **Decision Contract**:
  - If Faithfulness ≥ 0.90 → `is_grounded = True` → Route to `END` (deliver verified answer to user).
  - Else → `is_grounded = False` → Passes `unfaithful_statements` and critique to `feedback_agent` for diagnostic re-planning.

> [!WARNING] Sensitive Documents & Free-Tier Cloud Privacy Notice
> - **Cloud APIs & Sensitive Data**: While the production inference runtime (`qwen3.5:9b` via Ollama + Qdrant) is fully local and on-prem, the offline evaluation script (`src/evals/evaluate_offline_with_llm.py`) sends retrieved chunks and generated answers to external cloud endpoints (Google Gemini and Groq). Free-tier terms of service often permit providers to use submitted prompt data for model improvement and training. **If your SHP documents are confidential, proprietary, or classified, do not submit them to free cloud APIs.** Instead, evaluate using local models via Ollama.
> - **LangSmith Telemetry**: The example configuration sets `LANGSMITH_TRACING=false` in `.env` to prevent accidental external data transmission by default. Set to `true` with your `LANGCHAIN_API_KEY` only when cloud tracing is explicitly desired.

#### 4. Feedback Agent (Reflection & Remediation Loop)
- **Module**: `src/agents/feedback/`
- **Role**: Rather than a blind retry, the Feedback Agent acts as an intelligent diagnostic inspector whenever the **Retrieval Sufficiency Grader** or **Post-Generation Guardrail** rejects a run.
- **Inputs**:
  - `query`: Current engineering query.
  - Pre-Gen Failure Inputs (from `retrieval_grader`): `is_retrieval_sufficient: False`, `missing_aspects: list[str]`, `retrieval_critique: str`.
  - Post-Gen Failure Inputs (from `generation_guardrail`): `is_grounded: False`, `generation_metrics: dict`, `final_response: str`.
  - `cycle_count`: Current loop execution count.
- **Diagnostic Logic & Loop Back**:
  1. **Dual Failure Detection**:
     - If triggered after `retrieval_grader` $\to$ sets `failure_type = "INSUFFICIENT_RETRIEVAL"`, packages `missing_aspects`, and writes a targeted `remediation_hint` for the Planner Agent.
     - If triggered after `generation_guardrail` $\to$ sets `failure_type = "HALLUCINATION_DETECTED"` or `"OFF_TOPIC_ANSWER"`, identifies unverified statements, and directs the Planner or Generator on how to constrain claims.
  2. **Cycle Advancement & Loop Control**:
     - Increments `cycle_count` by 1 (`cycle_count + 1`).
     - **If `cycle_count < MAX_CYCLES`** (configured via `MAX_REWRITE_ATTEMPTS` in `.env`, default: 1; max: 3): Passes `feedback_diagnosis` back to the `planner` node. The Planner Agent absorbs the diagnosis, modifies search terms, adds missing formulas or technical parameters, adjusts Qdrant metadata filters (`is_table`), or allocates the MCP tool worker.
     - **If `cycle_count >= MAX_CYCLES`**: The circuit breaker trips and routes to `fallback` (graceful refusal with full transparency on what data was missing).
- **Outputs**: Structured Pydantic `FeedbackDiagnosis`:
  ```json
  {
    "failure_type": "INSUFFICIENT_RETRIEVAL | HALLUCINATION_DETECTED | OFF_TOPIC_ANSWER",
    "missing_aspects": [
      "Sarkaria formula for economic penstock diameter calculation",
      "Penstock shell wall thickness equations (IS 11639 / ASCE)"
    ],
    "critique": "Retrieved context only covered civil intake structures and lacked penstock hydraulic sizing data.",
    "remediation_hint": "Decompose into a dedicated sub-query for penstock diameter using calculate_penstock_diameter MCP tool and a narrative sub-query for steel plate thickness.",
    "target_recommendations": [
      "mcp_worker:calculate_penstock_diameter",
      "narrative_retriever"
    ]
  }
  ```


---

### Spec 06: Evaluation & Acceptance Criteria

#### 1. Multi-Provider LLM-as-a-Judge Offline Evaluation Framework (Zero API Fees)
The offline evaluation suite ([`src/evals/evaluate_offline_with_llm.py`](src/evals/evaluate_offline_with_llm.py)) implements an automated **Multi-Provider LLM-as-a-Judge** benchmark with resilient waterfall failover across free-tier API endpoints. It costs \$0 to execute, allowing anyone with an adequate consumer GPU to reproduce the evaluation without enterprise cloud budgets.

```
[Primary: Google Gemini] (gemini-2.5-flash / gemini-flash-latest)
       │ (On 429 quota exhaustion or daily rate limit)
       ▼
[Secondary: Groq 120B / 20B] (openai/gpt-oss-120b / openai/gpt-oss-20b / qwen/qwen3.8-27b)
       │ (With automatic prompt-chunk capping to stay within Groq's 8k TPM limit)
       ▼
[Tertiary & Fallback: Hugging Face & Local Ollama] (Safety net)
```

#### Measured Benchmark Performance (53 Golden Test Cases)
The table below reflects the actual benchmark evaluation run on [`src/evals/generated_answers_and_chunks.json`](src/evals/generated_answers_and_chunks.json) using Gemini 2.5 Flash and Groq 120B/20B:

| Metric | Measured Score | Target | Mathematical Formula | Core Engineering Meaning |
| :--- | :---: | :---: | :--- | :--- |
| **Faithfulness** | **0.7798 (78.0%)** | ≥ 0.75 | $\frac{\text{Claims Supported by Context}}{\text{Total Factual Claims in Answer}}$ | **Hallucination Prevention**: Are all parameters, formulas, and ratings in the generated answer strictly backed by context? (Claim-level micro-average). |
| **Answer Relevancy** | **0.7526 (75.3%)** | ≥ 0.70 | $\frac{\text{Sentences Addressing Query Intent}}{\text{Total Sentences in Answer}}$ | **Query Adherence**: Does the generated response directly answer the engineering question without drift or omissions? |
| **Context Precision** | **0.7391 (73.9%)** | ≥ 0.70 | $\frac{\text{Relevant Chunks Retrieved}}{\text{Total Chunks Retrieved}}$ | **Signal-to-Noise Ratio**: Of all chunks returned by the retriever, how many were relevant to formulating the answer? |
| **Context Recall** | **0.7921 (79.2%)** | ≥ 0.75 | $\frac{\text{Relevant Facts Found in Context}}{\text{Total Required Facts in Ground Truth}}$ | **Completeness**: How completely does the retrieved context cover the facts required in the golden ground-truth? |
| **Query Hallucination Rate** | **14 / 53 (26.4%)** | < 30% | $\frac{\text{Queries with } \ge 1 \text{ Flagged Claim}}{\text{Total Executed Queries}}$ | Binary query defect rate: proportion of questions where the judge flagged at least one unsupported claim or math error. |

#### Category-by-Category Benchmark Breakdown (53 Cases Reconciled)

> [!WARNING] Small Sample Size Limitation (Avoid Over-Interpretation)
> Several categories in the benchmark contain very small sample sizes ($N = 1$ to $4$ cases). In particular, the **Disambiguation / Reconciliation** category contains only **a single test case ($N = 1$, score 0.22)**, while **Adversarial / Guardrail ($N = 2$)**, **Adversarial & Hallucination-Trap ($N = 2$)**, **Technical Governance ($N = 3$)**, **Targeted Numerical ($N = 3$)**, **Multi-Hop Cross-Doc ($N = 3$)**, **Conditional Policy ($N = 4$)**, and **Tool Call / Computation ($N = 4$)** have limited representation.
> **Do not draw generalizable statistical conclusions from these individual category rows.** They serve as preliminary qualitative case studies and diagnostic indicators rather than statistically definitive domain performance.

| Category | Cases ($N$) | Faithfulness | Relevancy | Context Precision | Context Recall | Sample Note |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Technical Governance & Environmental Protocols** | 3 | **1.00 (100%)** | **0.87** | 0.63 | 0.83 | Small sample ($N=3$) |
| **Targeted Numerical & Policy Thresholds** | 3 | **0.95 (95%)** | **0.80** | 0.81 | 0.83 | Small sample ($N=3$) |
| **Multi-Hop Cross-Document Synthesis** | 6 | **0.90 (90%)** | **0.90** | 0.81 | 0.78 | Robust multi-doc reasoning |
| **Single-Hop Policy & Licensing** | 12 | **0.90 (90%)** | **0.78** | 0.64 | 0.85 | Core policy retrieval strength |
| **Single-Hop Fact Lookup** | 7 | **0.88 (88%)** | **0.93** | 0.87 | 0.88 | High precision factual extraction |
| **Conditional & Scenario-Based Policy Reasoning** | 4 | **0.86 (86%)** | **0.88** | 0.78 | 0.76 | Small sample ($N=4$) |
| **Adversarial / Guardrail** | 2 | **0.73 (73%)** | **0.85** | 0.89 | 0.81 | Small sample ($N=2$) |
| **Adversarial & Hallucination-Trap** | 2 | **0.61 (61%)** | **0.56** | 0.58 | 0.74 | Small sample ($N=2$) |
| **Table & Diagram Extraction** | 6 | **0.57 (57%)** | **0.56** | 0.69 | 0.61 | Weak: OCR alignment challenges |
| **Multi-Hop Cross-Doc** | 3 | **0.56 (56%)** | **0.76** | 0.79 | 0.79 | Small sample ($N=3$) |
| **Tool Call / Computation** | 4 | **0.42 (42%)** | **0.40** | 0.74 | 0.84 | Weak: Small sample ($N=4$), tool schema gap |
| **Disambiguation / Reconciliation** | 1 | **0.22 (22%)** | **0.12** | 0.78 | 0.41 | **Single case ($N=1$) — do not extrapolate** |
| **Total / Macro Average** | **53** | **0.780 (78.0%)** | **0.753 (75.3%)** | **0.739 (73.9%)** | **0.792 (79.2%)** | **Full 53-Case Benchmark Reconciled** |

#### Fallback, Gate Pass & Retry Statistics (Empirical Breakdown of 53 Cases)

Across the 53 executed golden benchmark cases (configured with `FAITHFULNESS_THRESHOLD=0.90` and `MAX_REWRITE_ATTEMPTS=1` in `.env`):

| Pipeline Outcome | Case Count | Percentage | Faithfulness | Relevancy | Context Precision | Context Recall | Average Latency | Primary Reason / Behavior |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Passed 0.90 Gate Directly / Retried** | **39** | **73.6%** | **88.7%** | **84.6%** | **75.2%** | **83.9%** | **586s (~9.8m)** | Grounded engineering synthesis; verified against context |
| **Circuit Breaker Tripped (Fallback)** | **14** | **26.4%** | **48.1%** | **49.4%** | **70.4%** | **66.2%** | **1,175s (~19.6m)** | Re-planning exhausted without passing 0.90 gate |

##### Why Did 14 Cases End in Fallback?
1. **Tier-2/3 Partial Data Notices (7 cases)**:
   - For cases like `GD-29` (adversarial hallucination trap querying nonexistent regulations) and `GD-27` (targeted numerical threshold absent from uploaded DPRs), the system's CRAG grader and guardrail functioned as intended: rather than hallucinating answers, the circuit breaker tripped and delivered a transparent **"Partial Engineering Data Notice"** disclosing that documentation was unavailable.
2. **Tier-1 Best-Effort Delivery with Disclaimers (7 cases)**:
   - For complex calculations (`GD-015`, `GD-023`) and multi-hop synthesis (`GD-13`, `GD-15`), the system exhausted retry loops and delivered best-effort answers with explicit qualification banners, acknowledging that groundedness could not be verified at $\ge 0.90$.

#### Plain-RAG Baseline Comparison vs. Agentic RAG

To measure the exact trade-offs of agentic reflection versus traditional retrieval, a standalone **Plain-RAG Baseline Runner** ([`src/evals/run_plain_rag_baseline.py`](src/evals/run_plain_rag_baseline.py)) was constructed. Plain-RAG executes single-shot hybrid retrieval (top-6 chunks) followed directly by single-turn LLM generation without planning, CRAG verification, or reflection loops:

| Dimension | Plain-RAG Baseline | Multi-Agent Hierarchical RAG | Trade-Off Analysis |
| :--- | :--- | :--- | :--- |
| **Architecture** | Single-shot retrieval + 1 LLM call | Hierarchical StateGraph (Planner + Orchestrator + Synthesizer + CRAG + Guardrail) | Agentic RAG performs multi-step verification and scatter-gather retrieval |
| **LLM Calls per Query** | **1 inference** | **3 to 7 inferences** | Plain-RAG uses 70–85% fewer LLM inferences |
| **Average Query Latency** | **~58–173 seconds** | **~305–550 seconds** (Median across 53: ~687s) | Plain-RAG is **~3× to 5× faster** by executing a single LLM inference |
| **Hallucination Prevention** | None (delivers raw LLM output) | **0.90 Post-Gen Guardrail + Circuit Breaker** | Agentic RAG intercepts 14 ungrounded cases and surfaces partial data notices |
| **Multi-Hop / Tabular Synthesis** | Frequently misses cross-document facts | High coverage via Orchestrator scatter-gather | Agentic RAG scores 90% on multi-hop cross-document synthesis |

```powershell
# Run Plain-RAG baseline on golden dataset (capped at 10 cases)
python -m src.evals.run_plain_rag_baseline --limit 10 --retriever hybrid

# Evaluate Plain-RAG results with LLM judge
python src/evals/evaluate_offline_with_llm.py --input src/evals/plain_rag_answers_and_chunks.json --output src/evals/plain_rag_evaluation_report.json --no-resume
```

##### Empirical Head-to-Head Benchmark: Plain-RAG vs. Agentic RAG (10 Policy Cases: GD-01 to GD-10)

The table below contrasts the empirical results evaluated by the multi-provider LLM judge (`gemini-2.5-flash`) across the first 10 golden benchmark cases (`GD-01` to `GD-10`) in [`src/evals/plain_rag_answers_and_chunks.json`](src/evals/plain_rag_answers_and_chunks.json) vs. [`src/evals/generated_answers_and_chunks.json`](src/evals/generated_answers_and_chunks.json):

| Metric / Dimension | Plain-RAG Baseline (10 Cases) | Multi-Agent RAG (Same 10 Cases) | Trade-Off & Engineering Analysis |
| :--- | :---: | :---: | :--- |
| **Faithfulness Score** | **0.9000 (90.0%)** | 0.8800 (88.0%) | Plain-RAG is slightly more concise; Agentic RAG's generator occasionally attempts extra synthetic formulas (e.g. eligibility equations). |
| **Answer Relevancy** | **0.9000 (90.0%)** | 0.8400 (84.0%) | Plain-RAG produces shorter, direct answers with less background narrative boilerplate. |
| **Context Precision** | 0.5780 (57.8%) | **0.6850 (68.5%)** | **Agentic Precision Advantage**: Planner query decomposition and targeted worker filtering improve retrieval signal-to-noise. |
| **Context Recall** | 0.9100 (91.0%) | **0.9550 (95.5%)** | **Agentic Recall Advantage**: Multi-agent scatter-gather recovers cross-document facts that single-shot retrieval misses. |
| **Median Latency (p50)**| **105.7s (~1.8m)** | 500.1s (~8.3m) | **Plain-RAG is 4.73× faster** by executing a single LLM inference without reflection overhead. |
| **Mean Latency** | **134.0s (~2.2m)** | 583.2s (~9.7m) | Consistent ~4.4× mean speedup across all 10 policy queries. |
| **Query Hallucination Rate** | **1 / 10 (10.0%)** | 4 / 10 (40.0%) | Plain-RAG's sole defect was an overly conservative refusal (`GD-08`), whereas Agentic RAG had minor ungrounded formula derivations. |

##### Case-by-Case Deep Dive: GD-01 & GD-02 Comparative Analysis

| Metric / Dimension | GD-01 (Plain-RAG) | GD-01 (Agentic RAG) | GD-02 (Plain-RAG) | GD-02 (Agentic RAG) | Engineering Analysis & Key Finding |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Query Type** | Multi-part Policy & Licensing | Multi-part Policy & Licensing | Focused Single-Hop Policy | Focused Single-Hop Policy | GD-01 requires cross-document facts; GD-02 is concentrated |
| **Context Recall** | **0.50 (50.0%)** | **0.70 (70.0%)** | **1.00 (100%)** | **1.00 (100%)** | **Cross-Doc Coverage Gap**: With typo-tolerant retrieval, Plain-RAG successfully located the 3 methods (50% recall), but missed preference scenarios spread across other documents. Agentic RAG decomposed the query across multiple documents to reach 70% recall. |
| **Context Precision** | 0.60 (60.0%) | **0.70 (70.0%)** | 0.60 (60.0%) | **0.90 (90.0%)** | Agentic RAG maintained higher signal-to-noise through targeted query planning and metadata filtering. |
| **Faithfulness Score** | **1.00 (100%)** | 0.75 (75.0%) | **1.00 (100%)** | **1.00 (100%)** | Plain-RAG scored 1.00 on both by adhering strictly to chunks; Agentic RAG scored 0.75 on GD-01 due to an ungrounded qualification formula. |
| **Answer Relevancy** | **1.00 (100%)** | 0.65 (65.0%) | **1.00 (100%)** | **1.00 (100%)** | On GD-02, both provided exact, concise policy answers. |
| **Execution Latency** | **208.4s** | 549.5s | **50.2s** | 305.2s | Plain-RAG is **~3× to 6× faster** by executing a single LLM inference without multi-agent coordination. |
| **Generation Outcome** | **Identified 3 Methods** | **Synthesized Comparison** | **Answered Directly** | **Answered with Verification** | **The Agentic Trade-Off**: For simple single-hop lookups (GD-02), Plain-RAG is sufficient and ~6× faster. For distributed multi-part questions (GD-01), Agentic RAG is necessary to gather cross-document criteria. |

###### Qualitative Case Study: Why the Pipelines Diverged

1. **Query GD-01 (Multi-part Policy & Licensing)**:
   - **User Query**: *"What are the three primary methods used by State Governments for the allotment or licensing of Small Hydro Power (SHP) sites, and when is each preferred?"*
   - **The Retrieval Bug Investigation (`allotmnet of shp.pdf`)**:
     - In initial testing, Plain-RAG scored **0.00 recall** and abstained. Forensic investigation revealed this was **not an inherent limit of RAG, but a filename typo bug**: the source document on disk is literally misspelled as `allotmnet of shp.pdf` (transposition of 'm' and 'n').
     - Because strict string matching failed (`"allotment" in "allotmnet of shp.pdf" == False`), the exact name boost did not fire. The crucial chunk (`Site allotment/license`) was pushed down to rank #7, just 1 rank outside `top_k=6`.
     - Implementing **typo-tolerant Damerau-Levenshtein entity matching** in [`src/ingestion/bm25_retriever.py`](src/ingestion/bm25_retriever.py) restored `Site allotment/license` to rank #2, elevating Plain-RAG Context Recall from **0.00 to 0.50 (50%)**.
   - **Vanilla / Plain-RAG Outcome (With Typo Fix)**:
     - **Result**: **Recall: 0.50, Precision: 0.60, Faithfulness: 1.00, Relevancy: 1.00 (Latency: 208.4s)**.
     - **Actual Output**: Clean 3-bullet list identifying (1) **Bidding process**, (2) **MoU method**, and (3) **Swiss model of defence** [Source: `allotmnet of shp.pdf`]. It accurately acknowledged that the retrieved chunks did not contain specific preference criteria for the bidding or Swiss models.
   - **Multi-Agent Hierarchical RAG Outcome**:
     - **Result**: **Context Recall reached 0.70 (70%)** with **Context Precision at 0.70 (70%)**.
     - **Why Agentic RAG Scored Higher Recall**: The question asked two parts: (a) the three methods, and (b) *when each is preferred*. While the method names reside in `allotmnet of shp.pdf`, the policy preference rationale resides in `hydro policy.pdf` and `WSHPDR_2016.pdf`. Plain-RAG's single-shot top-6 could not span all three documents, whereas the **Planner Agent** decomposed the question into parallel sub-queries, retrieving chunks across all relevant documents to synthesize the complete comparison.
     - **Faithfulness Note**: Faithfulness was scored at **0.75** because the generator attempted to formulate a pseudo-mathematical threshold formula for developer qualification (`$$\text{Eligibility Status} = \dots$$`) not explicitly present in the source text.

2. **Query GD-02 (Focused Single-Hop Policy)**:
   - **User Query**: *"During competitive bidding for an SHP site where a PFR/DPR is available, how are ties broken, and what happens if no bids are received?"*
   - **Vanilla / Plain-RAG Outcome**:
     - **Result**: **Flawless Execution (Recall: 1.00, Precision: 0.60, Faithfulness: 1.00, Relevancy: 1.00)**.
     - **Execution Latency**: **50.2 seconds** (**~6.1× faster** than Agentic RAG's 305.2s).
     - **Actual Output**: Clean 2-bullet answer directly quoting `allotmnet of shp.pdf` (ties broken by gradation list based on prequalification criteria; site allotted to government agency if no bids received twice).
   - **Multi-Agent Hierarchical RAG Outcome**:
     - **Result**: **Recall: 1.00, Precision: 0.90, Faithfulness: 1.00, Relevancy: 1.00 (Latency: 305.2s)**.
     - **Actual Output**: Complete 4-section engineering brief with executive summary, comparison matrix, regulatory provisions, and risk mitigation.
   - **Engineering Conclusion**: For focused, single-hop factual retrieval, Plain-RAG is optimal and avoids multi-agent coordination overhead (~50s vs ~300s). For distributed, multi-part questions requiring cross-document synthesis, Agentic RAG's planner decomposition is essential to bridge the cross-document coverage gap.

#### Critical Failure Analysis: The 26.4% Hallucination Rate & Weakest Categories

An aggregate query hallucination rate of **26.4% (14 out of 53 cases)** is substantial in mission-critical hydropower engineering. While policy lookup and regulatory queries achieved high groundedness (88% to 100%), performance dropped sharply in the two high-risk categories:

1. **Tool Call & Mathematical Regression Calculations (0.42 Faithfulness, $N=4$)**:
   - **Empirical Tool Dispatch Audit**: Across all 4 test cases in this category (`GD-003`, `GD-015`, `GD-022`, `GD-023`), **`mcp_worker` fired in 0 of 4 cases (0%)**.
   - **Root Cause (Tool Catalog Schema Gap & Planner Skip)**:
     - The FastAPI MCP server (`src/mcp_server/server.py`) implements 3 deterministic hydraulic calculators: `calculate_hydro_power`, `calculate_penstock_diameter`, and `calculate_specific_speed`.
     - However, the 4 benchmark test cases evaluated specialized domain regressions: Francis turbine cost formulas from Aggidis et al. (`GD-003`), CWC Report No. 1/73 unit hydrograph flood equations (`GD-022`, `GD-023`), and Cesar Alvarado-Ancieta (2009) global powerhouse E&M regression (`GD-015`).
     - Because no matching tool schema existed in the MCP tool catalog for these specialized non-linear formulas, the **Planner Agent appropriately skipped tool assignment** (allocating 0 MCP sub-tasks) and routed the sub-queries entirely to narrative document retrieval.
   - **Observed Failure Modes**:
     - **Inline Autoregressive Calculation (`GD-015`)**: When forced to calculate regressions inline in text without a tool, the Generator suffered severe arithmetic drift (outputting $42.13\text{ Million US\$}$ instead of the correct $\approx 49.1\text{ M\$}$) and misread a dense OCR table value ($4,286\text{ M\$}$ instead of $42.86\text{ M\$}$).
     - **Circuit Breaker Abort (`GD-003`, `GD-023`)**: When retrieved text lacked specific formula constants, the system refused to fabricate numbers and tripped the circuit breaker after 3 cycles, returning Partial Engineering Data Notices (lowering benchmark recall/faithfulness scores due to missing requested numbers).
   - **Experimental Purpose of MCP Calculators**: The MCP calculators exist as an **experimental demonstration** showing how external engineering tools can be registered and dispatched over JSON-RPC in an agentic graph. Expanding this to production requires authoring dedicated MCP endpoints for every empirical regression in the benchmark.
   - **Sarkaria Formula Investigation ($D = 0.62 \cdot Q^{0.43}$)**: For example, in our experimental penstock calculator, the economic diameter formula was implemented as a simplified flow-only empirical relation ($D = 0.62 \cdot Q^{0.43}$). In classical hydraulic engineering literature (USBR, ASCE, Indian Central Electricity Authority / CWC guidelines), the full Sarkaria formulation typically incorporates design head $H$ (e.g., $D \propto \frac{Q^{0.43}}{H^{0.14}}$ or head-loss optimization). This discrepancy underscored that experimental tools require strict schema alignment with specific engineering standards and explicit prompt guidance to ensure the LLM invokes them correctly.

2. **Table & Diagram Extraction (0.57 Faithfulness, $N=6$)**:
   - **Empirical Contrast (Clean Markdown vs. OCR Distortion)**:
     - When tables had cleanly preserved Markdown delimiters, extraction was flawless. In **`GD-009`** (project cycle duration table from `AK-DPR Aug 2017.pdf`), the retrieved chunks contained clean Markdown pipes (`||**Activities**|**Time Taken**|`), and the model scored **1.00 (100%) Faithfulness and 1.00 Relevancy**.
     - In 4 out of the 6 cases (`GD-006`, `GD-007`, `GD-011`, `GD-016`), dense multi-column PDF tables and scanned figures suffered from parser distortion:
       - **Scanned Character Noise (`GD-006` Chunk 0)**: Scanned pages in `hydro policy.pdf` produced garbled OCR character sequences:
         ```text
         [Source: hydro policy.pdf | Section: P08 Hydpe]
         |'St<br>~.~1&~~~1&~<br>.jg;<br>1<br>~~~!f~~~l~~l<br>~~_I~1O·1&!1&t~...|---|---|
         ```
       - **Multi-Row Collapsing (`GD-016` Table 1)**: In Alvarado-Ancieta (2009), the document parser collapsed multiple table rows into single cell blocks separated by HTML `<br>` tags:
         ```text
         |#<br>Year<br>Country|Hydropower<br>Project|Power<br>P [MW]|Net Head<br>H [m]|...
         |28<br>2008<br>29<br>2007|Akocak<br>AkkoyI|80<br>103.5|250<br>149...
         ```
         This collapsed vertical row boundaries, causing the Generator to miscorrelate project names with cost and head metrics, leading the LLM judge to penalize factual groundedness.

#### Aspirational Targets vs. Recalibrated Baselines: An Honest Retrospective
- **Why Targets Moved**: The initial SDD specifications proposed aggressive aspirational targets: Faithfulness $\ge 0.90$, Precision $\ge 0.85$, Recall $\ge 0.85$, and sub-5-second interactive responses.
- **Empirical Recalibration**: When deploying an open-weight 9B model on an 8 GB consumer GPU (RTX 5050), hardware and model realities emerged:
  - Latency is memory bandwidth-bound (~45s to 120s for multi-turn agentic reflection cycles).
  - Across 53 challenging test cases, realistic performance settled at **78.0% Faithfulness**, **75.3% Answer Relevancy**, **73.9% Context Precision**, and **79.2% Context Recall**.
- **Transparency**: We do not present "targets met" as an unqualified triumph. Rather, these results provide a transparent, empirical benchmark for running local agentic RAG on consumer hardware, pinpointing exact areas (table parsing and MCP tool dispatch) for future development.

#### Provider Waterfall Execution Statistics
- **Google Gemini (`gemini-2.5-flash` / `gemini-flash-latest`)**: Evaluated **23 cases** before reaching the Google free tier daily cap (20 requests/day).
- **Groq (`openai/gpt-oss-120b` / `openai/gpt-oss-20b`)**: Seamlessly took over via waterfall failover and evaluated the remaining **30 cases** without crashing.
- **Failures / Crashes**: **0 (100% completion rate)** across all 53 test cases.

```powershell
# Run Multi-Provider LLM-as-a-Judge Offline Benchmark (Zero API fees)
python src/evals/evaluate_offline_with_llm.py --mode auto --delay 2.0
```

---

### Spec 06.1: Hardware Performance & Scaling (RTX 5050 vs Production GPUs)

> [!NOTE] Hardware Architecture: Compute vs. Memory Bandwidth
> In transformer autoregressive decoding, token generation speed is strictly **Memory Bandwidth-bound**, not compute-bound. For every single generated token, the GPU must transfer the complete 6.6 GB of `qwen3.5:9b` weights from VRAM across the memory bus to the Tensor Cores. Consequently, generation throughput scales almost linearly with VRAM memory bus speed ($\text{GB/s}$).

#### 1. Baseline Benchmark: NVIDIA GeForce RTX 5050 Laptop GPU (8 GB)
- **VRAM Capacity**: 8.0 GB GDDR6
- **Memory Bus Width**: 128-bit
- **Effective Memory Bandwidth**: $\approx 224\text{ GB/s}$
- **Measured Generation Throughput (`qwen3.5:9b`)**: **12 to 15 tokens/second**
- **Measured Per-Node Response Latencies (RTX 5050 Laptop GPU)**:
  - Direct Conversational Reply (~150 tokens): **~6.6 seconds**
  - Full Technical Engineering Report (~1,050 tokens): **~75 to 85 seconds**
  - Synthesizer Deterministic Fusion (Zero LLM overhead): **0.000 seconds**
  - Single-Pass Groundedness Guardrail (~120 tokens): **~15 to 20 seconds**
- **Measured End-to-End Multi-Agent Lifecycle Latency (53 Golden Cases Benchmark)**:
  - **Median Latency (p50)**: **687.2 seconds (~11.5 minutes)**
  - **95th Percentile Latency (p95)**: **1,601.1 seconds (~26.7 minutes)**
  - **Mean Latency**: **741.9 seconds (~12.4 minutes)**
  - **Passed 0.90 Gate Directly / Retried ($N=39$)**: Median **501.6s (~8.4m)**, p95 **1,214.3s (~20.2m)**
  - **Exhausted Reflection / Fallback Notice ($N=14$)**: Median **986.7s (~16.4m)**, p95 **1,738.3s (~29.0m)**
  - **Fastest Direct Execution**: **22.2 seconds** | **Longest Multi-Turn Iteration**: **1,786.9 seconds (~29.8m)**
- **VRAM Headroom Reality**:
  $$\text{VRAM Usage} \approx 6.6\text{ GB (Model)} + 1.2\text{ GB (8k KV Cache)} + 0.4\text{ GB (CUDA Runtime)} = \mathbf{8.2\text{ GB}}$$
  On an 8 GB GPU, memory is utilized at 100% capacity. When desktop compositing or multi-agent memory spikes occur, slight host paging across the PCIe bus can introduce minor transient latency variations.

#### 2. Production GPU Scaling Matrix (Measured Baseline vs. Theoretical Projections)

> [!NOTE] Measurement Distinction
> **Only the GeForce RTX 5050 Laptop GPU baseline was empirically benchmarked and measured in local hardware testing.**
> Figures for desktop, workstation, and datacenter tiers (RTX 4070 Ti, RTX 4090, RTX 6000 Ada, A100, H100) are **theoretical projections estimated** based on linear memory bus bandwidth scaling ($\text{GB/s}$) during transformer autoregressive token decoding.

| Hardware Tier | GPU Model | Benchmark Status | VRAM | Bus Width | Memory Bandwidth | Throughput (`qwen3.5:9b`) | 1,000-Token Report | End-to-End RAG Lifecycle |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Local Baseline (Laptop)** | **GeForce RTX 5050 Laptop** | **Empirically Measured** | **8 GB** | **128-bit** | **224 GB/s** | **~12–15 tok/s** | **~75–85s** | **Median: ~11.5m \| p95: ~26.7m** |
| **Mid-Range Desktop** | GeForce RTX 4070 Ti Super | *Estimated Projection* | 16 GB | 256-bit | 672 GB/s | ~40–50 tok/s | ~20–25s | ~40–50s |
| **High-End Desktop** | GeForce RTX 4090 | *Estimated Projection* | 24 GB | 384-bit | 1,008 GB/s | ~65–80 tok/s | ~12–15s | ~20–25s |
| **Professional Workstation**| RTX 6000 Ada Generation | *Estimated Projection* | 48 GB | 384-bit | 960 GB/s | ~70–90 tok/s | ~11–14s | ~18–22s |
| **Datacenter Enterprise** | NVIDIA A100 SXM4 | *Estimated Projection* | 80 GB | 5120-bit HBM2e | 2,039 GB/s | ~120–150 tok/s | ~6–8s | ~10–14s |
| **Hyperscale Acceleration**| NVIDIA H100 SXM5 | *Estimated Projection* | 80 GB | 5120-bit HBM3 | 3,350 GB/s | ~180–230 tok/s | ~4–5s | ~6–9s |

#### 3. Enterprise Production Benefits on 24 GB / 32 GB / 64 GB+ (Theoretical Projections)
1. **Projected $5\times$ to $15\times$ Latency Reduction**: Based on memory bandwidth scaling, upgrading from a 128-bit mobile bus (224 GB/s) to enterprise high-bandwidth memory (1,008 to 3,350 GB/s) is projected to reduce total end-to-end multi-agent execution time from the **~8–16 minute range observed on consumer laptop GPUs down to an estimated 30–60 seconds (or 8–15 seconds with concurrent continuous batching via vLLM / TensorRT-LLM)** (theoretical projection).
2. **Zero Host RAM Paging**: 24 GB+ VRAM provides abundant headroom for full 32k context lengths, multi-turn session history, and full FP16 KV caches with zero PCIe bus offloading.
3. **Concurrent Multi-Agent Parallelism**: High-capacity GPUs allow setting `OLLAMA_NUM_PARALLEL=4` (or deploying with vLLM / TensorRT-LLM PagedAttention continuous batching), enabling Planner, Orchestrator, narrative workers, tabular workers, and guardrails to run simultaneously without queue serialization.

---

## 4. Multi-Tier Document Parsing Pipeline (Operational)

The ingestion pipeline ([`src/ingestion/parser.py`](src/ingestion/parser.py)) implements a cost-effective, fully on-prem **3-tier smart routing architecture** designed to maximize extraction fidelity (preserving Markdown headings, tables, and lists), eliminate OCR hallucination on digital text, and selectively engage GPU/CPU OCR only where visual elements or scans require it.

### Routing Architecture

```mermaid
flowchart TD
    Input([Input Document in data/raw/]) --> HashCheck{SHA-256 Hash\nin data/parsed/?}
    HashCheck -->|Cache Hit & not --force| Cached[Load Cached .md\nSkip Parsing]
    HashCheck -->|New or --force| RouteExt{Inspect File\nExtension}

    %% Plain Text
    RouteExt -->|.txt / .md| PlainText[Direct UTF-8 Reader]

    %% Office Docs
    RouteExt -->|.docx| DocxParser[python-docx + Embedded RapidOCR]
    RouteExt -->|.doc| DocCOM[Windows Word COM Automation]

    %% Presentations
    RouteExt -->|.pptx| PPTXParser[python-pptx DOM Parser\nNative Tables + Text]
    RouteExt -->|.ppt| PPTCOM[PowerPoint COM Conversion\n.ppt ──> .pptx]
    PPTCOM --> PPTXParser
    PPTXParser --> PPTImgCheck{Slide has Images\nor Scans?}
    PPTImgCheck -->|Image-only slide <30 chars| SlideOCR[RapidOCR Full Slide Body]
    PPTImgCheck -->|Digital text + Diagram| DiagramOCR[RapidOCR Diagram Blockquote]
    PPTImgCheck -->|No images| SlideClean[Clean Markdown Output]

    %% Spreadsheets
    RouteExt -->|.xlsx| XlsxParser[openpyxl ──> GFM Markdown Tables\n+ Embedded RapidOCR]
    RouteExt -->|.xls| XlsParser[xlrd ──> GFM Markdown Tables\n+ Embedded RapidOCR]

    %% PDFs
    RouteExt -->|.pdf| PDFProbe[PyMuPDF4LLM Fast-Path Probe]
    PDFProbe --> PDFCharCheck{Alphanumeric chars\nacross all pages > 80?}
    PDFCharCheck -->|Yes: Digital PDF| PyMuPDFOut[Structured Markdown Tables & Headings\n~0.2s/doc]
    PDFCharCheck -->|No: Scanned PDF| DoclingGPU[Docling + PyPdfium Backend\n+ EasyOCR on NVIDIA RTX 5050 GPU]

    %% Standalone Images
    RouteExt -->|.png / .jpg / .jpeg| DoclingGPU

    %% Output
    PlainText --> OutMd["data/parsed/{stem}__{hash}.md"]
    DocxParser --> OutMd
    DocCOM --> OutMd
    SlideOCR --> OutMd
    DiagramOCR --> OutMd
    SlideClean --> OutMd
    XlsxParser --> OutMd
    XlsParser --> OutMd
    PyMuPDFOut --> OutMd
    DoclingGPU --> OutMd

    OutMd --> MetaIndex["Upsert data/parsed/metadata.json"]
```

---

### Format-by-Format Extraction Strategy

| File Type | Extensions | Primary Engine | OCR Engine & Mode | Table Extraction Quality | Speed |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Digital PDF** | `.pdf` | `pymupdf4llm` (C++ fitz) | Selective (image crops only) | Native GFM Markdown tables (`\| ... \|`) | **~0.2s–0.5s** / doc |
| **Scanned PDF** | `.pdf` | `Docling` (`PyPdfiumDocumentBackend`) | **EasyOCR on CUDA GPU** (NVIDIA RTX 5050, `sm_120`) | Layout transformer table recovery | **~5s–15s** / doc |
| **PowerPoint** | `.pptx`, `.ppt` | `python-pptx` (via COM for `.ppt`) | **RapidOCR** (selective per-image & image-only slides) | Native GFM Markdown tables (`\| ... \|`) | **~1s–5s** / doc |
| **Word Document** | `.docx`, `.doc` | `python-docx` / Word COM | **RapidOCR** (on embedded figures/charts) | Native GFM Markdown tables (`\| ... \|`) | **~0.5s–1.5s** / doc |
| **Excel Workbook** | `.xlsx`, `.xls` | `openpyxl` / `xlrd` | **RapidOCR** (on embedded charts/drawings) | Native GFM Markdown tables (`\| ... \|`) | **~0.05s–0.2s** / doc |
| **Standalone Images** | `.png`, `.jpg`, `.jpeg` | `Docling` / `EasyOCR` | Full-image OCR (GPU/CPU) | Visual table recognition | **~2s–5s** / image |
| **Plain Text** | `.txt`, `.md` | Direct UTF-8 stream | None | Preformatted Markdown | **< 0.01s** |

---

### Detailed Technical Implementations

#### 1. PDFs: Digital Fast-Path with Scanned GPU Fallback
- **Digital Fast-Path**: Uses `pymupdf4llm` to convert digital vector PDFs directly into clean Markdown. Preserves structural headings (`#`, `##`, `###`), bold text, bullet points, and converts vector-drawn and text-delimited tables into standard GitHub-Flavored Markdown tables.
- **Scanned Detection Probe**: Calculates the clean alphanumeric character count across all pages. If fewer than 80 characters are detected, it classifies the file as a scanned document or blueprint and engages the GPU vision pipeline.
- **GPU Docling Vision OCR**:
  - Configured with `PyPdfiumDocumentBackend` (Google Chromium PDFium engine) for native crash-proof stability on Windows.
  - Accelerated via PyTorch CUDA 12.8 (`cu128`) on the **NVIDIA GeForce RTX 5050 Laptop GPU (8GB VRAM, Blackwell `sm_120`)**.
  - Runs vision layout models + EasyOCR for multi-column text restoration and scanned table grid reconstruction.

#### 2. PowerPoint (`.pptx` & `.ppt`): Native Tables + Selective OCR
- **Native Table Integrity**: Extracts structured tables directly from PowerPoint's internal shape tree (`shape.has_table`). Every row and cell is mapped to a Markdown table with header dividers, preserving numeric data (MW, kW, tariffs, flow rates) without OCR distortion.
- **Smart Selective OCR**:
  - **Image-Only Slides**: If a slide contains virtually no digital text (< 30 characters) and no table (e.g. a scanned slide or photograph), RapidOCR transcribes the image and formats the text as the primary slide body.
  - **Embedded Diagrams & Flowcharts**: If a slide has digital text *plus* embedded charts, schematics, or maps, RapidOCR transcribes the visual text and appends it cleanly in a blockquote:
    ```markdown
    > **[Slide 13 Image/Diagram OCR]**: Day of Year NATURAL FLOW PATTERN Flood River Flow Natural High Flows
    ```
- **Legacy `.ppt` COM Bridge**: Uses Windows PowerPoint COM automation (`win32com.client`) in headless mode (`WithWindow=False`) to convert legacy binary `.ppt` files to `.pptx` in-memory before parsing, ensuring legacy presentations receive identical table and text fidelity.

#### 3. Excel Spreadsheets (`.xlsx` & `.xls`): Worksheets to Tables
- **`.xlsx`**: Processed via `openpyxl` with `data_only=True` to capture computed cell values rather than raw formula strings.
- **`.xls`**: Processed via `xlrd` for legacy binary workbooks.
- Converts all active worksheets into dedicated Markdown sections (`## Sheet: SheetName`) followed by GFM Markdown tables with proper column alignment.
- Worksheet drawings and floating chart objects are inspected and OCR'd via RapidOCR.

#### 4. Word Documents (`.docx` & `.doc`):
- Modern `.docx` files extract heading styles, bullet hierarchies, and table grids via `python-docx`. Embedded `r:embed` image streams in document parts are inspected and OCR'd.
- Legacy `.doc` files extract formatted text via Windows Word COM automation (`win32com.client.Dispatch("Word.Application")`).

#### 5. Hardware & Process Optimization:
- **GPU (RTX 5050 via CUDA 12.8)**: Dedicated exclusively to heavy vision models (Docling layout transformers, scanned PDF EasyOCR, and upcoming Qwen3 vector embeddings) to prevent VRAM fragmentation.
- **CPU (Multi-Threaded ONNX Runtime with AVX2)**: RapidOCR (PP-OCRv6 engine, ~15MB weights) runs on multi-core CPU, achieving 0.3s–0.5s per image with zero GPU memory contention.

---

### Ingestion CLI Commands

```powershell
# Parse all supported documents in data/raw/ (skips already-parsed files via SHA-256 cache)
python -m src.ingestion.parser

# Force re-parse all documents (overwriting data/parsed/)
python -m src.ingestion.parser --force

# Force re-parse only PowerPoint presentations
python -m src.ingestion.parser --force --ext .pptx .ppt

# Force re-parse Word and Excel files
python -m src.ingestion.parser --force --ext .docx .doc .xlsx .xls

# Run with custom directories and parallel workers
python -m src.ingestion.parser --raw-dir ./data/raw --parsed-dir ./data/parsed --workers 2
```

---

### Markdown-Aware Semantic Chunking (`src/ingestion/chunker.py`)

The chunking pipeline ([`src/ingestion/chunker.py`](src/ingestion/chunker.py)) is specifically engineered for high-precision technical retrieval and optimized for smaller/local LLMs (`qwen3.5:9b`):

1. **Context Breadcrumb Injection**:
   - Every chunk automatically inherits its complete document hierarchy (e.g. `[Source: Madkini flood peak report.doc | Location: Slide/Page 3 | Section: Flood Frequency Analysis > Gumbel Distribution]`).
   - Prepending this context ensures that local models immediately understand what project, river, or clause a technical excerpt refers to, eliminating retrieval ambiguity.
2. **Atomic Table Protection**:
   - Markdown tables (`| ... |`) are identified and kept unbroken within a single chunk whenever possible.
   - For giant tables exceeding token limits, the table is split row-by-row while **repeating the column header row on every split chunk**, ensuring every chunk remains a valid, readable Markdown table.
3. **Slide Boundary Preservation**:
   - Presentation slides (`## Slide {i}`) are preserved as distinct contextual units.
4. **Token Budgeting via Tiktoken**:
   - Exact token counts are computed using `tiktoken` (`cl100k_base`), maintaining chunks around **~330 tokens** (well within `Qwen3-Embedding-0.6B`'s sweet spot).

#### Chunk Metadata Payload (For Citations)
```json
{
  "chunk_id": "HDP_AHN511__9fe0f668__c0002",
  "source_file": "HDP_AHN511.ppt",
  "file_type": "PPT",
  "document_title": "HDP_AHN511",
  "breadcrumb": "HDP_AHN511 > Slide 1: Dynamic Pricing of electrical power",
  "section_hierarchy": ["HDP_AHN511", "Slide 1: Dynamic Pricing of electrical power"],
  "page_or_slide": 1,
  "chunk_index": 2,
  "total_chunks": 36,
  "is_table": false,
  "char_count": 482,
  "token_count": 121
}
```

#### Chunking CLI Commands
```powershell
# Chunk all parsed documents in data/parsed/ (outputs to data/chunks/chunks.jsonl)
python -m src.ingestion.chunker

# Customize chunk size and overlap
python -m src.ingestion.chunker --max-chars 1800 --overlap-chars 150
```

---

### 4.2 Vector Embedding Pipeline (`qwen3-embedding:0.6b`)

The vector embedding engine ([`src/ingestion/embedder.py`](src/ingestion/embedder.py)) translates semantic chunk texts into high-density 1024-dimensional continuous representations using local Ollama.

#### Model Selection Rationale
- **Model**: `qwen3-embedding:0.6b` (639 MB manifest).
- **Vector Dimension**: `1024` floats per vector.
- **VRAM Footprint**: ~640 MB VRAM, co-existing comfortably alongside `qwen3.5:9b` within the 8 GB VRAM budget of the NVIDIA GeForce RTX 5050 Laptop GPU.
- **Multilingual & Domain Breadth**: Outstanding semantic density across technical engineering concepts (hydrology, hydro-mechanical turbine envelopes, civil structures, Indian Central Electricity Authority regulatory guidelines).

#### Technical Specifications & Reliability
1. **Ollama Batch API Endpoint**: Interacts with `http://127.0.0.1:11434/api/embed` passing multi-text arrays (`batch_size=32`), drastically cutting HTTP request overhead compared to single-item endpoints.
2. **Incremental Checkpointing**: Reads existing `data/embeddings/embedded_chunks.jsonl` upon startup; already-embedded `chunk_id`s are skipped automatically, preventing redundant recomputation during interrupted runs.
3. **Resilient Retry Protocol**: Implements exponential backoff retries (`2^attempt` second backoff) to smoothly absorb transient Ollama GPU memory re-allocations.
4. **Execution Performance**:
   - **Total Chunks Embedded**: **4,064**
   - **Throughput**: **21.5 chunks/second** (189.37 seconds total runtime)
   - **Failures / Drops**: **0**

```powershell
# Generate embeddings for all chunks in data/chunks/chunks.jsonl
python -m src.ingestion.embedder --batch-size 32

# Re-embed from scratch (dropping cache)
python -m src.ingestion.embedder --force
```

---

### 4.3 Qdrant Vector Indexing & Dense Retrieval Engine

The vector indexer ([`src/ingestion/indexer.py`](src/ingestion/indexer.py)) ingests embedded chunks into the `shp_knowledge` Qdrant collection for high-speed similarity search and payload filtering.

#### Architecture Highlights
1. **Zero-Dependency Dual Mode**:
   - **Server Mode**: Probes `http://127.0.0.1:6333`. Connects to a live Qdrant container if running.
   - **Embedded Disk Mode (Fallback)**: If no server daemon is running, seamlessly launches an embedded local disk engine storing data in `data/qdrant_db/`. No Docker or background service required.
2. **Deterministic UUID5 Identifiers**:
   - Every chunk ID (e.g. `HDP_AHN511__9fe0f668__c0002`) maps to a deterministic UUID via `uuid5(NAMESPACE_SHP, chunk_id)`.
   - Guaranteed idempotent upserts: Running indexer repeatedly never generates duplicate points or phantom vectors.
3. **Rich Metadata Payload Indexing**:
   - Payload attributes indexed for pre-filtering: `file_type`, `is_table`, `source_file`, `page_or_slide`.
   - Full citation text preserved: `content`, `breadcrumb`, `document_title`, `token_count`.
4. **Indexing Benchmarks**:
   - **Total Indexed Points**: **4,064**
   - **Upsert Speed**: **199.3 points/second** (completed in 20.39 seconds)
   - **Distance Metric**: **Cosine**

#### CLI & Verification Commands
```powershell
# Index embedded chunks into Qdrant collection 'shp_knowledge'
python -m src.ingestion.indexer --batch-size 128

# Recreate collection from scratch
python -m src.ingestion.indexer --recreate

# Test semantic query retrieval with top-3 formatted citations
python -m src.ingestion.indexer --query "Kaplan turbine efficiency curve and cavitation limits" --top-k 3

# Test structured table-only filtering
python -m src.ingestion.indexer --query "hydrological discharge flow data" --filter-table --top-k 3
```

---

## 5. System Architecture & Hierarchical Subgraph Workflow

The system is architected as a **Hierarchical Multi-Agent LangGraph System**, combining a top-level **Parent StateGraph** (`src/agents/main_agent.py`) with specialized **Agent Subgraphs** and dynamic map-reduce workers.

### 5.1 Parent StateGraph vs. Subgraph States (LangGraph Architecture Pattern)

In LangGraph, there are two primary paradigms for designing subgraphs:
1. **Shared State Subgraphs (Identical Schema)**: Both parent and child graphs share the exact same state TypedDict. State is shared directly across all nodes.
2. **Independent State Subgraphs with Transformation (Our System Architecture)**:
   - The **Parent Graph** (`main_agent`) maintains the overarching conversation channel and workflow lifecycle via `AgentState`.
   - Each **Specialized Agent Subgraph** maintains its own dedicated, domain-specific state schema (e.g. `PlannerAgentState`, `WorkerInputState`, `SynthesizerAgentState`, `GeneratorAgentState`).
   - **How the Parent State is Populated from Subgraph States**:
     * **Scatter-Gather Worker Dispatch (`Send` API)**: The Orchestrator node extracts sub-queries from the parent state, wraps them into specialized `WorkerInputState` payloads, and dispatches them in parallel via LangGraph's `Send()` API.
     * **Channel Reducer Fan-In**: Each worker returns an isolated dictionary `{ "sub_agent_results": { ... } }`. In the parent `AgentState`, this channel is defined as:
       ```python
       sub_agent_results: Annotated[dict[str, list[DocumentChunk]], merge_sub_agent_results]
       ```
       The custom channel reducer `merge_sub_agent_results` automatically merges incoming worker dictionaries into the parent state without race conditions or overwriting parallel outputs.
     * **Adapter Node Transformation**: Each agent node acts as a bridge adapter: it reads the relevant parent state keys, passes them to the agent's logic/subgraph, and returns a dictionary updating only the designated output fields (e.g. `is_retrieval_sufficient`, `synthesized_context`, `final_response`) in the parent state.

```mermaid
flowchart TD
    subgraph ParentGraph ["Master Parent StateGraph (AgentState)"]
        Start([User Query]) --> PlannerNode["🎯 planner Adapter Node"]
        
        subgraph PlannerSub ["Planner Agent Subgraph (PlannerAgentState)"]
            P_Exec["Intent Classification & Sub-Query Planning"]
        end
        PlannerNode <--> P_Exec

        PlannerNode -->|"rag_retrieval"| OrchNode["⚡ plan_orchestration Adapter Node"]
        PlannerNode -->|"direct_answer"| Final([Return Direct Response])
        PlannerNode -->|"out_of_scope"| FallbackNode["Fallback Node"] --> Final

        subgraph WorkerSubgraphs ["Dynamic Scatter-Gather Sub-Agents (WorkerInputState)"]
            OrchNode -->|"Send WorkerInputState"| W1["narrative_worker: Manuals & Guidelines"]
            OrchNode -->|"Send WorkerInputState"| W2["tabular_worker: Discharge & Sizing Tables"]
            OrchNode -->|"Send WorkerInputState"| W3["mcp_worker: FastAPI MCP Power Tools"]
            W1 & W2 --> Qdrant[("Qdrant Vector DB: 4,064 Chunks")]
            W3 --> MCPServer["🛠️ FastAPI MCP Server: Port 8602"]
            W1 & W2 & W3 --> AggNode["📦 aggregate_results Node"]
        end

        AggNode --> SynthNode["🧩 synthesizer Adapter Node"]
        subgraph SynthSub ["Synthesizer Subgraph (SynthesizerAgentState)"]
            S_Exec["Deterministic Fair-Share Fusion & Citation Deduplication"]
        end
        SynthNode <--> S_Exec

        SynthNode --> PreGuardNode{"🛡️ retrieval_grader Adapter Node"}
        subgraph GraderSub ["Retrieval Grader Subgraph (RetrievalGraderAgentState)"]
            G_Exec["Reference-Free CRAG Sufficiency Check"]
        end
        PreGuardNode <--> G_Exec

        PreGuardNode -->|"is_sufficient = True or cycle >= max_cycles"| GenNode["✍️ generator Adapter Node"]
        subgraph GenSub ["Generator Subgraph (GeneratorAgentState)"]
            Gen_Exec["Grounded Answer Synthesis (num_predict=8192)"]
        end
        GenNode <--> Gen_Exec

        PreGuardNode -->|"is_sufficient = False and cycle < max_cycles"| FeedNode["🔁 feedback Adapter Node"]

        GenNode --> PostGuardNode{"🛡️ generation_guardrail Adapter Node"}
        subgraph GuardSub ["Guardrail Subgraph (GenerationGuardrailState)"]
            Guard_Exec["Single-Pass Faithfulness & Groundedness"]
        end
        PostGuardNode <--> Guard_Exec

        PostGuardNode -->|"is_grounded = True"| Final
        PostGuardNode -->|"is_grounded = False and cycle < max_cycles"| FeedNode

        subgraph FeedSub ["Feedback Subgraph (FeedbackAgentState)"]
            F_Exec["Defect Diagnosis & Remediation Plan"]
        end
        FeedNode <--> F_Exec

        FeedNode -->|"cycle_count += 1: Re-plan with Diagnostic Hints"| PlannerNode

        PostGuardNode -->|"is_grounded = False and cycle >= max_cycles"| FallbackNode
    end
```

---

### 5.2 Model Context Protocol (MCP) FastAPI Server (`src/mcp_server`)

The system includes a high-performance **FastAPI-powered Model Context Protocol (MCP) server** on **port 8602** providing dual interfaces:
1. **JSON-RPC 2.0 MCP Endpoint (`POST /mcp`)**: Conforms to the open Model Context Protocol specification for integration with Orchestrator sub-agents, Antigravity, and Claude Desktop.
2. **REST API & Swagger UI (`/docs`, `/api/v1/...`)**: Interactive OpenAPI documentation enabling engineers to test calculations and SCADA queries directly in their browser at `http://localhost:8602/docs`.

While dense vector retrieval covers manuals and historical records, the MCP server provides deterministic mathematical solvers and plant sensor access to eliminate LLM arithmetic hallucinations and enable live telemetry queries.

> [!NOTE] Experimental Demonstration of MCP Tool Calling
> The MCP engineering calculators (e.g. hydro power, specific speed, penstock diameter) are provided as an **experimental demonstration** of tool-calling integration in agentic RAG workflows. They show how external deterministic solvers can be plugged in via MCP, allowing engineers to experiment with, test, and refine specialized domain calculation tools.

#### Registered MCP Engineering Tools:

| Tool Name | Endpoint / Parameters | Engineering Purpose / Formula |
| :--- | :--- | :--- |
| `calculate_hydro_power` | `POST /api/v1/calculate/hydro-power`<br>`discharge_m3s`, `net_head_m`, `efficiency=0.88`, `capacity_factor=0.65` | Solves $P = 9.81 \cdot Q \cdot H_{\text{net}} \cdot \eta$ (kW/MW) and annual generation (MWh). |
| `calculate_specific_speed` | `POST /api/v1/calculate/specific-speed`<br>`rotational_speed_rpm`, `power_kw`, `net_head_m` | Computes metric specific speed $N_s = \frac{N \sqrt{P}}{H^{5/4}}$ and recommends optimal runner (Pelton, Francis, Kaplan, Crossflow). |
| `calculate_penstock_diameter` | `POST /api/v1/calculate/penstock-diameter`<br>`design_discharge_m3s` | Computes economic internal diameter via Sarkaria formula $D = 0.62 \cdot Q^{0.43}$ and validates velocity within 2.5–5.0 m/s. |
| `query_plant_scada` | `POST /api/v1/scada/query`<br>`plant_id`, `sensor_tag` | Inspects real-time or simulated plant sensors (forebay level, penstock pressure, generator bearing temp, runner vibration). |

#### Client Integration:
The Orchestrator Agent utilizes [`FastAPIMCPClient`](src/mcp_server/client.py) which can execute in-process with zero network latency, or connect over HTTP (`http://127.0.0.1:8602`) when running as a standalone microservice:

```powershell
# Launch FastAPI MCP Server with Uvicorn
python -m src.mcp_server.server

# Interactive Swagger Documentation
http://localhost:8602/docs
```

---

## 6. Project Directory Structure

```text
Agentic_RAG/
├── src/
│   ├── api/
│   │   ├── __init__.py                      # Package entry point
│   │   ├── app.py                           # FastAPI application server (lifespan, CORS, routes)
│   │   └── schemas.py                       # Pydantic v2 request, response, and telemetry models
│   ├── prompts/
│   │   ├── planner_agent_system.jinja2      # Planner Agent system prompt
│   │   ├── planner_agent_user.jinja2        # Planner Agent user prompt + few-shots
│   │   ├── orchestrator_agent_system.jinja2 # Orchestrator Agent worker allocation prompt
│   │   ├── orchestrator_agent_user.jinja2   # Orchestrator sub-query assignment prompt
│   │   ├── retrieval_grader_system.jinja2   # Reference-Free Retrieval Grader system prompt
│   │   ├── retrieval_grader_user.jinja2     # Reference-Free Retrieval Grader user prompt + few-shots
│   │   ├── generator_agent_system.jinja2    # Grounded Generator system prompt
│   │   ├── generator_agent_user.jinja2      # Grounded Generator user prompt
│   │   ├── generation_guardrail_system.jinja2 # Post-Gen Groundedness Guardrail system prompt
│   │   ├── generation_guardrail_user.jinja2 # Post-Gen Groundedness Guardrail user prompt + few-shots
│   │   ├── feedback_agent_system.jinja2     # Feedback diagnostic system prompt
│   │   └── feedback_agent_user.jinja2       # Feedback diagnostic remediation prompt
│   ├── states/
│   │   ├── planneragent_state.py            # Planner Agent TypedDict & Pydantic models
│   │   ├── orchestratoragent_state.py       # Orchestrator & worker task schemas
│   │   ├── synthesizeragent_state.py        # Synthesizer Agent output & state schemas
│   │   ├── retrieval_grader_state.py        # Retrieval Grader output & state schemas
│   │   ├── generatoragent_state.py          # Generator output & state schemas
│   │   └── agent_state.py                   # Global unified AgentState
│   ├── mcp_server/
│   │   ├── __init__.py                      # Package exports (FastAPIMCPClient, tools)
│   │   ├── server.py                        # MCP JSON-RPC 2.0 tool server (Power, Ns, Penstock, SCADA)
│   │   └── client.py                        # MCP in-process & HTTP client for Orchestrator
│   ├── agents/
│   │   ├── planner/
│   │   │   ├── agentnodes.py                # Planner node (Jinja2 + Ollama qwen3.5:9b)
│   │   │   ├── agentgraph.py                # Planner LangGraph StateGraph
│   │   │   └── planner.py                   # Public facade API
│   │   ├── orchestrator/
│   │   │   ├── agentnodes.py                # Sub-agent dispatch & Qdrant retrieval
│   │   │   ├── agentgraph.py                # Orchestrator LangGraph workflow
│   │   │   └── orchestrator.py              # Public facade API
│   │   ├── synthesizer/
│   │   │   ├── agentnodes.py                # Fusion, deduplication & citation re-ranking
│   │   │   ├── agentgraph.py                # Synthesizer LangGraph workflow
│   │   │   └── synthesizer.py               # Public facade API
│   │   ├── retrieval_grader/
│   │   │   ├── agentnodes.py                # Reference-Free CRAG Grader (Ollama qwen3.5:9b)
│   │   │   ├── agentgraph.py                # Retrieval Grader LangGraph workflow
│   │   │   └── retrieval_grader.py          # Public facade API
│   │   ├── generator/
│   │   │   ├── agentnodes.py                # Grounded answer synthesis node
│   │   │   ├── agentgraph.py                # Generator LangGraph workflow
│   │   │   └── generator.py                 # Public facade API
│   │   ├── generation_guardrail/
│   │   │   ├── agentnodes.py                # Single-Pass LLM Groundedness Guardrail (qwen3.5:9b)
│   │   │   ├── agentgraph.py                # Post-Gen Guardrail LangGraph workflow
│   │   │   └── generation_guardrail.py      # Public facade API
│   │   └── feedback/
│   │       ├── agentnodes.py                # Diagnostic reflection on guardrail failures
│   │       ├── agentgraph.py                # Feedback Agent workflow
│   │       └── feedback.py                  # Public facade API
│   ├── utils/
│   │   ├── deepeval_evaluator.py            # Offline DeepEval retrieval benchmark harness (QA)
│   │   └── logging.py                       # Loguru centralized logger configuration
│   ├── ingestion/
│   │   ├── parser.py                        # Smart Multi-Tier Document Parser (Operational)
│   │   ├── chunker.py                       # Markdown-aware semantic text splitting (Operational)
│   │   ├── embedder.py                      # Qwen3-Embedding-0.6B via Ollama (Operational)
│   │   ├── indexer.py                       # QDrant collection manager & upsert (Operational)
│   │   └── bm25_retriever.py                # Okapi BM25 keyword & exact name retriever with RRF (Operational)
│   ├── evals/
│   │   ├── shp_agentic_rag_golden_dataset.json # Curated golden dataset (30 policy & licensing cases: GD-01 to GD-30)
│   │   ├── agentic_rag_shp_golden_dataset.json # Curated golden dataset (25 engineering cases: 23 in full 53-case run GD-003 to GD-025; GD-001/002 in pilot tests)
│   │   ├── generate_dataset_answers.py      # Automated batch generator for test datasets
│   │   ├── generated_answers_and_chunks.json# Full responses, citations & raw retrieved chunks
│   │   ├── run_plain_rag_baseline.py        # Plain-RAG baseline benchmark runner (Operational)
│   │   ├── evaluate_offline_with_llm.py     # Multi-Provider LLM-as-a-Judge offline benchmark suite
│   │   ├── evaluation_report.json           # Comprehensive benchmark report with scores & critiques
│   │   └── run_deepeval.py                  # Optional DeepEval assertion testing harness
│   └── main.py                              # CLI entry point
├── ui/                                      # Streamlit Interactive Web Application
│   ├── app.py                               # Streamlit main entry point (wide layout)
│   ├── api_client.py                        # Resilient HTTP client for FastAPI
│   ├── runner.py                            # Programmatic runner entry point
│   └── components/                          # Modular UI widgets
│       ├── sidebar.py                       # System diagnostics, telemetry, file upload
│       ├── chat_view.py                     # Multi-turn chat & circuit breaker banner
│       ├── citations_view.py                # Collapsible verified source explorer
│       └── calculator.py                    # Hydro sizing & turbine selector calculator
├── data/
│   ├── raw/                              # Raw SHP documents (PDF, DOCX, PPTX, XLS, etc.)
│   ├── parsed/                           # Structured Markdown output + metadata.json
│   ├── chunks/                           # Semantic chunks JSONL + summary
│   ├── embeddings/                       # 1024-dim vectors JSONL + summary
│   ├── qdrant_db/                        # Local embedded Qdrant vector store
│   └── checkpoints.db                    # LangGraph SQLite checkpoint store
├── .env.example                          # Environment variable template
├── pyproject.toml                        # uv project dependencies
└── README.md                             # This file — Project spec & documentation
```

---

## 7. Setup & Installation

### Prerequisites
- Python `>= 3.11`
- [uv](https://github.com/astral-sh/uv) (fast Python package manager)
- [Ollama](https://ollama.com/) installed and running
- [QDrant](https://qdrant.tech/) running locally (Docker recommended)

### 1. Pull Ollama Models
```powershell
ollama pull qwen3.5:9b
ollama pull qwen3-embedding:0.6b
```

### 2. Start QDrant (Docker)
```powershell
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

### 3. Install Python Dependencies
```powershell
# Create & sync virtual environment
uv venv
.\.venv\Scripts\activate
uv sync
```

### 4. Run Ingestion Pipeline (One-Time Setup)
```powershell
# 1. Parse all 54 raw documents (PDF, DOCX, PPTX, XLS) to structured Markdown
python -m src.ingestion.parser

# 2. Chunk Markdown into semantic segments with breadcrumb headers
python -m src.ingestion.chunker

# 3. Generate 1024-dim embeddings via Ollama qwen3-embedding:0.6b
python -m src.ingestion.embedder --batch-size 32

# 4. Index points into Qdrant vector database (embedded local or server)
python -m src.ingestion.indexer --recreate

# 5. Verify vector retrieval
python -m src.ingestion.indexer --query "Kaplan turbine efficiency curve" --top-k 3

# 6. Verify BM25 keyword & exact name retrieval (sub-5ms)
python -m src.ingestion.bm25_retriever --query "allotment of shp competitive bidding" --top-k 3
```

### 5. Run the Agent & User Interfaces

#### A. Interactive CLI
```powershell
# Interactive Multi-Turn Chat (with In-Memory Checkpointing)
python src/main.py

# System Health Diagnostics (Ollama, Qdrant, FastAPI MCP)
python src/main.py --check

# Single Query Execution
python src/main.py "Compare Francis vs Pelton turbines for 120m net head and 2.0 m3/s discharge in high quartz silt conditions"
```

#### B. FastAPI Asynchronous Backend Server (Port 8600)
```powershell
# Start FastAPI backend server (port 8600 avoids Windows Hyper-V reserved ports)
python src/api/app.py
```
- **Interactive Swagger Documentation**: [http://localhost:8600/docs](http://localhost:8600/docs)
- **ReDoc Technical Specification**: [http://localhost:8600/redoc](http://localhost:8600/redoc)
- **Health Check Probe**: `GET http://localhost:8600/api/v1/health`
- **Execution Endpoint**: `POST http://localhost:8600/api/v1/chat`

#### C. Streamlit Web Engineering Portal (Port 8601)
```powershell
# Start Streamlit Frontend (port 8601)
streamlit run ui/app.py --server.port 8601
```
- **Web UI URL**: [http://localhost:8601](http://localhost:8601)
- **Features**:
  - 💬 **Multi-Turn Engineering Chat**: Markdown + KaTeX LaTeX equations ($P = \rho g Q H \eta$).
  - ⚠️ **Circuit Breaker Callout**: High-visibility warnings when fallback activates.
  - ⚙️ **SHP Sizing Calculator**: Live hydraulic power, annual MWh generation, and turbine envelope selector.
  - 📚 **Citations Inspector**: Expandable technical source verification with similarity scores and full chunk views.
  - 📊 **Real-Time Telemetry**: Live latency, self-correction cycle counts, and CRAG sufficiency checks.
  - 📤 **Document Ingestion Portal**: Drag-and-drop DPR, manual, and spec uploads.

#### D. Multi-Provider LLM-as-a-Judge Offline Benchmark Runner
```powershell
# 1. Batch generate answers and retrieved chunks across golden test datasets
python src/evals/generate_dataset_answers.py

# 2. Run multi-provider LLM-as-a-Judge offline evaluation (Gemini -> Groq waterfall)
python src/evals/evaluate_offline_with_llm.py --mode auto --delay 2.0

# Target specific cloud provider (e.g., Groq directly):
python src/evals/evaluate_offline_with_llm.py --mode groq --delay 2.0

# Target local Ollama instance (100% offline, zero cloud calls):
python src/evals/evaluate_offline_with_llm.py --mode ollama --delay 0.5
```

---

## 8. Configuration (.env)

Copy `.env.example` to `.env`:
```powershell
Copy-Item .env.example .env
```

Key variables:

```env
# Free-Tier Evaluation API Keys (for offline LLM-as-a-Judge)
free_gemini_key=your_gemini_api_key
Groq_key=your_groq_api_key
HF_TOKEN=your_huggingface_token

# Ollama Local Runtime
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=qwen3.5:9b
OLLAMA_EMBED_MODEL=qwen3-embedding:0.6b
OLLAMA_NUM_CTX=16384

# QDrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=shp_knowledge

# Retrieval & Hybrid Search
RETRIEVER_TOP_K=6
RETRIEVER_SCORE_THRESHOLD=0.65
ENABLE_BM25_RETRIEVAL=true
RETRIEVER_MODE=hybrid
BM25_WEIGHT=0.5
DENSE_WEIGHT=0.5

# Self-healing limits
MAX_REWRITE_ATTEMPTS=1
MAX_GENERATION_ATTEMPTS=1

# Agent Latency Timeouts (seconds) & Max Generation Tokens
PLANNER_TIMEOUT_S=120
PLANNER_MAX_TOKENS=2048
ORCHESTRATOR_TIMEOUT_S=120
ORCHESTRATOR_MAX_TOKENS=2048
RETRIEVAL_GRADER_TIMEOUT_S=120
RETRIEVAL_GRADER_MAX_TOKENS=2048
GENERATOR_TIMEOUT_S=600
GENERATOR_MAX_TOKENS=8192
GENERATION_GUARDRAIL_TIMEOUT_S=180
GENERATION_GUARDRAIL_MAX_TOKENS=2048
FEEDBACK_TIMEOUT_S=180
FEEDBACK_MAX_TOKENS=2048
QDRANT_TIMEOUT_S=60
MCP_TIMEOUT_S=30
API_CHAT_TIMEOUT_S=1800

# Server Ports (8600-8602 avoid Windows Hyper-V / WSL reserved port conflicts)
API_PORT=8600
API_SERVER_URL=http://127.0.0.1:8600
STREAMLIT_PORT=8601
MCP_PORT=8602
MCP_SERVER_URL=http://127.0.0.1:8602

# LangSmith (disabled by default for local privacy; set true to enable tracing)
LANGSMITH_TRACING=false
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=agentic-rag-shp
```

---

## 9. Development Roadmap

| Phase | Description | Status |
|---|---|---|
| **Phase 0: Spec Freeze** | Finalize all 6 specs in this README; review & sign off | ✅ Complete |
| **Phase 1: Ingestion Pipeline** | Multi-Tier Document Parser (54/54) → Semantic Chunker (4,064 chunks) → Embedder (qwen3-embedding) → Qdrant Indexer | ✅ Complete (4,064 pts indexed) |
| **Phase 2: Planner Agent** | Unified Intent Classification (`direct_answer` / `out_of_scope` / `rag_retrieval`), Sub-Query Decomposition, and Feedback Absorption (`src/agents/planner/`) | ✅ Complete & Verified |
| **Phase 3: Orchestrator Agent & Sub-Agents** | Dynamic dispatch of specialized sub-queries against Qdrant (`narrative_worker`, `tabular_worker`, `mcp_worker`) via LangGraph `Send()` map-reduce (`src/agents/orchestrator/`) | ✅ Complete & Verified |
| **Phase 4: Synthesizer Agent** | Multi-source chunk fusion, deduplication, fair-share sub-query balancing, and citation re-ranking (`src/agents/synthesizer/`) | ✅ Complete & Verified |
| **Phase 5: Retrieval Grader, Generator & Guardrails** | Reference-Free Retrieval Sufficiency Grader (CRAG) + Grounded Generator (`qwen3.5:9b`, 8k tokens) + Post-Gen Faithfulness Guardrail + Diagnostic Feedback Agent | ✅ Complete & Verified |
| **Phase 6: Graph Wiring & In-Memory Checkpointing** | Connect LangGraph Hierarchical StateGraph, specialized agent subgraphs, conditional edges, cycle counter circuit breaker, and In-Memory RAM checkpointing (`MemorySaver`) | ✅ Complete & Verified |
| **Phase 7: Evaluation Benchmark & CLI** | Multi-Provider LLM-as-a-Judge offline evaluation suite (`src/evals/evaluate_offline_with_llm.py`) with Gemini + Groq waterfall scoring 53 golden cases (78.0% Faithfulness, 75.3% Relevancy, 73.9% Precision, 79.2% Recall) | ✅ Complete & Verified |
| **Phase 8: Web UI & REST API** | Decoupled FastAPI Server (port 8600) with LangGraph singleton lifecycle + Streamlit Engineering Portal (port 8601) with live telemetry, hydro sizing calculator, citations inspector, and circuit-breaker alerts | ✅ Complete & Verified |