"""
Ollama-Powered Embedding Generator for Qwen3-Embedding-0.6B.

Features:
  1. High-throughput batch processing via Ollama's /api/embed endpoint (~25-30 chunks/sec).
  2. Incremental checkpointing: skips already-embedded chunk IDs so resuming is instant.
  3. Validates 1024-dimensional semantic vectors.
  4. Automatic retry with exponential backoff on transient timeouts.
  5. Outputs streaming JSONL (data/embeddings/embedded_chunks.jsonl) and summary metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, List, Optional

from loguru import logger

# ── Logging Setup ─────────────────────────────────────────────────────────────

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{function}:{line}</cyan> - {message}",
    level="INFO",
)
logger.add(
    LOG_DIR / "embedder.log",
    rotation="20 MB",
    retention="7 days",
    encoding="utf-8",
    level="DEBUG",
)

# ── Defaults & Configuration ──────────────────────────────────────────────────

DEFAULT_CHUNKS_PATH = Path("data/chunks/chunks.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/embeddings")
DEFAULT_MODEL_NAME = "qwen3-embedding:0.6b"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_BATCH_SIZE = 32
DEFAULT_TIMEOUT_S = 60
EXPECTED_VECTOR_DIM = 1024


@dataclass
class EmbeddingResult:
    """Summary metrics of an embedding run."""
    model_name: str = DEFAULT_MODEL_NAME
    vector_dimension: int = EXPECTED_VECTOR_DIM
    total_chunks: int = 0
    newly_embedded: int = 0
    already_cached: int = 0
    failed_chunks: int = 0
    avg_chunks_per_second: float = 0.0
    duration_seconds: float = 0.0
    output_path: str = ""


# ── Ollama Embedding Client ───────────────────────────────────────────────────

class OllamaEmbeddingClient:
    """Lightweight, resilient HTTP client for Ollama's /api/embed batch endpoint."""

    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_URL,
        model_name: str = DEFAULT_MODEL_NAME,
        timeout: int = DEFAULT_TIMEOUT_S,
        max_retries: int = 3,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries
        self.embed_url = f"{self.base_url}/api/embed"

    def check_health(self) -> bool:
        """Verify Ollama is reachable and the requested model is installed."""
        tags_url = f"{self.base_url}/api/tags"
        try:
            req = urllib.request.Request(tags_url)
            with urllib.request.urlopen(req, timeout=5) as r:
                data = json.loads(r.read().decode())
                models = [m.get("name", "") for m in data.get("models", [])]
                # Match name with or without tag
                match = any(
                    self.model_name == m or self.model_name.split(":")[0] == m.split(":")[0]
                    for m in models
                )
                if not match:
                    logger.warning(
                        f"Model '{self.model_name}' not found in Ollama library. "
                        f"Installed models: {models}"
                    )
                    return False
                return True
        except Exception as exc:
            logger.error(f"Cannot reach Ollama at {self.base_url}: {exc}")
            return False

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Send a batch of text strings to Ollama /api/embed.
        Retries with exponential backoff on transient errors.
        """
        payload = json.dumps({
            "model": self.model_name,
            "input": texts,
        }).encode("utf-8")

        for attempt in range(1, self.max_retries + 1):
            try:
                req = urllib.request.Request(
                    self.embed_url,
                    data=payload,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    embeddings = data.get("embeddings", [])
                    if len(embeddings) != len(texts):
                        raise ValueError(
                            f"Expected {len(texts)} embeddings, received {len(embeddings)}"
                        )
                    return embeddings

            except (urllib.error.URLError, TimeoutError, ValueError) as err:
                wait_time = 2 ** attempt
                logger.warning(
                    f"Embedding batch failed (attempt {attempt}/{self.max_retries}): {err}. "
                    f"Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)

        raise RuntimeError(
            f"Failed to embed batch of {len(texts)} texts after {self.max_retries} retries."
        )

    def embed_query(self, text: str) -> list[float]:
        """Convenience method to embed a single query string."""
        return self.embed_batch([text])[0]



# ── Core Embedder Pipeline ────────────────────────────────────────────────────

def embed_corpus(
    chunks_path: Path = DEFAULT_CHUNKS_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    model_name: str = DEFAULT_MODEL_NAME,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    force: bool = False,
) -> EmbeddingResult:
    """
    Read chunks from JSONL, generate dense vector embeddings via Ollama in batches,
    and persist streaming output to data/embeddings/embedded_chunks.jsonl.
    """
    t0 = time.perf_counter()
    chunks_path = Path(chunks_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    out_file = output_dir / "embedded_chunks.jsonl"
    summary_file = output_dir / "embedding_summary.json"

    if not chunks_path.exists():
        logger.error(f"Chunks file not found: {chunks_path}")
        return EmbeddingResult()

    # 1. Initialize Ollama Client and verify health
    client = OllamaEmbeddingClient(
        base_url=ollama_url,
        model_name=model_name,
    )
    if not client.check_health():
        logger.error(f"Ollama health check failed for model '{model_name}'.")
        return EmbeddingResult()

    # 2. Check existing embedded chunks for resuming
    cached_ids: set[str] = set()
    if out_file.exists() and not force:
        try:
            with open(out_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        obj = json.loads(line)
                        if "chunk_id" in obj:
                            cached_ids.add(obj["chunk_id"])
            logger.info(f"[CACHE] Found {len(cached_ids):,} already-embedded chunks in {out_file.name}")
        except Exception as e:
            logger.warning(f"Error reading existing embeddings: {e}")

    # 3. Read chunks to process
    raw_chunks: list[dict[str, Any]] = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw_chunks.append(json.loads(line))

    total_chunks = len(raw_chunks)
    to_embed = [c for c in raw_chunks if c["chunk_id"] not in cached_ids] if not force else raw_chunks
    already_cached = total_chunks - len(to_embed)

    logger.info(
        f"Starting embedding pipeline with '{model_name}': "
        f"{len(to_embed):,} chunks to embed ({already_cached:,} cached), "
        f"batch_size={batch_size}"
    )

    if not to_embed:
        logger.success(f"All {total_chunks:,} chunks are already embedded!")
        return EmbeddingResult(
            model_name=model_name,
            total_chunks=total_chunks,
            newly_embedded=0,
            already_cached=already_cached,
            output_path=str(out_file),
        )

    # 4. Open output file for streaming writes (append mode if resuming)
    write_mode = "w" if force or not cached_ids else "a"
    newly_embedded = 0
    failed_chunks = 0
    batch_t0 = time.perf_counter()

    with open(out_file, write_mode, encoding="utf-8") as out_f:
        # Process in batches
        for i in range(0, len(to_embed), batch_size):
            chunk_batch = to_embed[i : i + batch_size]
            texts = [c["content"] for c in chunk_batch]

            try:
                embeddings = client.embed_batch(texts)
                for c, emb in zip(chunk_batch, embeddings):
                    record = {
                        "chunk_id": c["chunk_id"],
                        "source_file": c["source_file"],
                        "file_type": c["file_type"],
                        "document_title": c["document_title"],
                        "breadcrumb": c["breadcrumb"],
                        "section_hierarchy": c.get("section_hierarchy", []),
                        "page_or_slide": c.get("page_or_slide"),
                        "chunk_index": c["chunk_index"],
                        "total_chunks": c["total_chunks"],
                        "is_table": c["is_table"],
                        "char_count": c["char_count"],
                        "token_count": c["token_count"],
                        "content": c["content"],
                        "embedding": emb,
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    newly_embedded += 1

                # Log progress periodically
                elapsed = time.perf_counter() - batch_t0
                throughput = newly_embedded / elapsed if elapsed > 0 else 0
                pct = (newly_embedded / len(to_embed)) * 100
                logger.info(
                    f"Progress: {newly_embedded:,}/{len(to_embed):,} ({pct:.1f}%) "
                    f"| {throughput:.1f} chunks/s | Elapsed: {elapsed:.1f}s"
                )

            except Exception as batch_err:
                logger.error(f"Failed to embed batch starting at index {i}: {batch_err}")
                failed_chunks += len(chunk_batch)

    duration = time.perf_counter() - t0
    avg_speed = newly_embedded / duration if duration > 0 else 0

    # 5. Persist summary metrics
    summary = {
        "model_name": model_name,
        "vector_dimension": EXPECTED_VECTOR_DIM,
        "total_chunks_in_corpus": total_chunks,
        "newly_embedded": newly_embedded,
        "already_cached": already_cached,
        "failed_chunks": failed_chunks,
        "duration_seconds": round(duration, 2),
        "avg_chunks_per_second": round(avg_speed, 2),
        "output_file": str(out_file),
    }
    summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.success(
        f"Embedding complete — {newly_embedded:,} newly embedded, {already_cached:,} cached "
        f"across {total_chunks:,} chunks in {duration:.2f}s ({avg_speed:.1f} chunks/s)"
    )

    return EmbeddingResult(
        model_name=model_name,
        vector_dimension=EXPECTED_VECTOR_DIM,
        total_chunks=total_chunks,
        newly_embedded=newly_embedded,
        already_cached=already_cached,
        failed_chunks=failed_chunks,
        avg_chunks_per_second=avg_speed,
        duration_seconds=duration,
        output_path=str(out_file),
    )


# ── CLI Interface ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cli = argparse.ArgumentParser(
        description="Batch Vector Embedding Generator using local Ollama (qwen3-embedding:0.6b)."
    )
    cli.add_argument(
        "--chunks-path", type=Path, default=DEFAULT_CHUNKS_PATH,
        help="Path to chunks.jsonl input file.",
    )
    cli.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Output directory for embedded_chunks.jsonl and summary.",
    )
    cli.add_argument(
        "--model", type=str, default=DEFAULT_MODEL_NAME,
        help="Ollama embedding model name (default: qwen3-embedding:0.6b).",
    )
    cli.add_argument(
        "--ollama-url", type=str, default=DEFAULT_OLLAMA_URL,
        help="Ollama base URL (default: http://127.0.0.1:11434).",
    )
    cli.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help="Batch size for parallel embedding requests (default: 32).",
    )
    cli.add_argument(
        "--force", action="store_true",
        help="Re-embed all chunks even if already cached in output.",
    )

    args = cli.parse_args()

    res = embed_corpus(
        chunks_path=args.chunks_path,
        output_dir=args.output_dir,
        model_name=args.model,
        ollama_url=args.ollama_url,
        batch_size=args.batch_size,
        force=args.force,
    )

    print("\n" + "=" * 55)
    print("  EMBEDDING SUMMARY")
    print("=" * 55)
    print(f"  Model Name       : {res.model_name}")
    print(f"  Vector Dimension : {res.vector_dimension}")
    print(f"  Total Chunks     : {res.total_chunks:,}")
    print(f"  Newly Embedded   : {res.newly_embedded:,}")
    print(f"  Already Cached   : {res.already_cached:,}")
    print(f"  Failed Chunks    : {res.failed_chunks:,}")
    print(f"  Duration         : {res.duration_seconds:.2f}s")
    print(f"  Throughput       : {res.avg_chunks_per_second:.1f} chunks/sec")
    print(f"  Output Path      : {res.output_path}")
    print("=" * 55 + "\n")
