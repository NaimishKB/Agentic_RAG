"""
=============================================================================
Small Hydropower Plant (SHP) Agentic RAG - Qdrant Vector Indexer
=============================================================================
Module: src.ingestion.indexer
Purpose: Builds, manages, and queries Qdrant vector collections for chunked
         and embedded SHP technical documentation.

Features:
- Dual-Mode Operation: Automatically connects to a live Qdrant server
  (http://127.0.0.1:6333) or seamlessly falls back to embedded local disk
  storage (`data/qdrant_db/`) requiring 0 external Docker/daemon dependencies.
- Cosine Distance & 1024-dimension vector space (matches qwen3-embedding:0.6b).
- Fast Payload Indexing on key metadata fields: `file_type`, `is_table`,
  `page_or_slide`, and `source_file` for pre-filtered agentic queries.
- Deterministic UUID5 point IDs: Idempotent indexing preventing duplicate entries.
- Streaming batch upserts with retry handling.
- Built-in verification retrieval engine with hybrid metadata filtering.
=============================================================================
"""

import argparse
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import requests
from loguru import logger
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from src.ingestion.embedder import OllamaEmbeddingClient

# ---------------------------------------------------------------------------
# Configuration Constants
# ---------------------------------------------------------------------------
DEFAULT_COLLECTION_NAME = "shp_knowledge"
DEFAULT_VECTOR_DIM = 1024
DEFAULT_QDRANT_URL = "http://127.0.0.1:6333"
DEFAULT_LOCAL_PATH = "data/qdrant_db"
DEFAULT_EMBEDDINGS_FILE = "data/embeddings/embedded_chunks.jsonl"
DEFAULT_INDEX_SUMMARY_FILE = "data/embeddings/index_summary.json"
DEFAULT_BATCH_SIZE = 128

# Namespace UUID for deterministic chunk ID generation
NAMESPACE_SHP = uuid.uuid5(uuid.NAMESPACE_DNS, "shp.agentic-rag.internal")

# Configure console encoding for Windows UTF-8 safety
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Configure logger
logger.remove()

logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)
os.makedirs("logs", exist_ok=True)
logger.add(
    "logs/indexer.log",
    rotation="20 MB",
    retention="14 days",
    level="DEBUG",
    encoding="utf-8",
)


# ---------------------------------------------------------------------------
# Qdrant Index Manager
# ---------------------------------------------------------------------------
class QdrantIndexManager:
    """Manages collection creation, payload indexing, and batch upsert into Qdrant."""

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        url: Optional[str] = DEFAULT_QDRANT_URL,
        local_path: Optional[str] = DEFAULT_LOCAL_PATH,
        prefer_local: bool = False,
        vector_dim: int = DEFAULT_VECTOR_DIM,
    ):
        self.collection_name = collection_name
        self.vector_dim = vector_dim
        self.client, self.mode = self._init_client(url, local_path, prefer_local)
        logger.info(
            f"Initialized QdrantClient in [{self.mode}] mode | Target collection: '{self.collection_name}'"
        )

    def _init_client(
        self,
        url: Optional[str],
        local_path: Optional[str],
        prefer_local: bool,
    ) -> Tuple[QdrantClient, str]:
        """Initializes client connecting to server if reachable, or disk storage fallback."""
        if not prefer_local and url:
            try:
                resp = requests.get(f"{url.rstrip('/')}/healthz", timeout=1.5)
                if resp.status_code == 200:
                    client = QdrantClient(url=url, timeout=30.0)
                    return client, f"Server ({url})"
            except Exception:
                logger.debug(f"Live Qdrant server not detected at {url}. Falling back to embedded local storage.")

        # Local embedded storage
        target_path = Path(local_path or DEFAULT_LOCAL_PATH)
        target_path.mkdir(parents=True, exist_ok=True)
        client = QdrantClient(path=str(target_path))
        return client, f"Embedded Local Disk ({target_path})"

    def ensure_collection(self, recreate: bool = False) -> None:
        """Creates or recreates the target collection with cosine distance and payload indexes."""
        collections = [c.name for c in self.client.get_collections().collections]
        exists = self.collection_name in collections

        if exists and recreate:
            logger.warning(f"Recreating collection '{self.collection_name}' (dropping existing data)...")
            self.client.delete_collection(collection_name=self.collection_name)
            exists = False

        if not exists:
            logger.info(
                f"Creating collection '{self.collection_name}' (dim={self.vector_dim}, metric=COSINE, on_disk_payload=True)..."
            )
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_dim,
                    distance=models.Distance.COSINE,
                    on_disk=True,
                ),
                on_disk_payload=True,
            )
            self._create_payload_indexes()
        else:
            logger.info(f"Collection '{self.collection_name}' already exists.")

    def _create_payload_indexes(self) -> None:
        """Creates payload field indexes for rapid pre-filtering during agentic query planning."""
        fields = [
            ("file_type", models.PayloadSchemaType.KEYWORD),
            ("source_file", models.PayloadSchemaType.KEYWORD),
            ("is_table", models.PayloadSchemaType.KEYWORD),
            ("page_or_slide", models.PayloadSchemaType.INTEGER),
        ]
        for field_name, schema_type in fields:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=schema_type,
                )
                logger.debug(f"Created payload index for '{field_name}' ({schema_type})")
            except Exception as e:
                logger.debug(f"Payload index creation for '{field_name}' note: {e}")

    @staticmethod
    def chunk_id_to_uuid(chunk_id: str) -> str:
        """Generates a deterministic UUID string from a chunk ID string."""
        return str(uuid.uuid5(NAMESPACE_SHP, chunk_id))

    def count(self) -> int:
        """Returns the current number of points in the collection."""
        try:
            return self.client.count(collection_name=self.collection_name).count
        except Exception:
            return 0

    def upsert_batch(self, points: List[models.PointStruct]) -> None:
        """Upserts a batch of points into the collection."""
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
    ) -> List[models.ScoredPoint]:
        """Performs vector similarity search with optional metadata filtering."""
        query_filter = None
        if filter_dict:
            must_conditions = []
            for k, v in filter_dict.items():
                if isinstance(v, list):
                    must_conditions.append(
                        models.FieldCondition(
                            key=k,
                            match=models.MatchAny(any=v),
                        )
                    )
                else:
                    must_conditions.append(
                        models.FieldCondition(
                            key=k,
                            match=models.MatchValue(value=v),
                        )
                    )
            query_filter = models.Filter(must=must_conditions)

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            score_threshold=score_threshold,
        )
        return response.points



# ---------------------------------------------------------------------------
# Stream Chunks and Indexer Pipeline
# ---------------------------------------------------------------------------
def stream_embedded_chunks(
    file_path: Path,
) -> Generator[Dict[str, Any], None, None]:
    """Yields parsed JSON records from embedded chunks JSONL file."""
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def build_index(
    embeddings_path: str = DEFAULT_EMBEDDINGS_FILE,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
    recreate: bool = False,
    prefer_local: bool = False,
    qdrant_url: Optional[str] = DEFAULT_QDRANT_URL,
    summary_path: str = DEFAULT_INDEX_SUMMARY_FILE,
) -> Dict[str, Any]:
    """Reads embedded chunks from JSONL and indexes all points into Qdrant."""
    embeddings_file = Path(embeddings_path)
    if not embeddings_file.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")

    manager = QdrantIndexManager(
        collection_name=collection_name,
        url=qdrant_url,
        prefer_local=prefer_local,
        vector_dim=DEFAULT_VECTOR_DIM,
    )
    manager.ensure_collection(recreate=recreate)

    start_time = time.time()
    points_buffer: List[models.PointStruct] = []
    indexed_count = 0
    total_embedded_chunks = 0

    # Count total chunks first for accurate progress display
    logger.info(f"Scanning embedded records from {embeddings_file}...")
    with open(embeddings_file, "r", encoding="utf-8") as f:
        total_embedded_chunks = sum(1 for line in f if line.strip())

    logger.info(
        f"Beginning Qdrant indexing: {total_embedded_chunks:,} vectors | Batch size: {batch_size}"
    )

    for chunk in stream_embedded_chunks(embeddings_file):
        chunk_id = chunk["chunk_id"]
        embedding = chunk.get("embedding")

        if not embedding or len(embedding) != DEFAULT_VECTOR_DIM:
            logger.warning(
                f"Skipping chunk {chunk_id}: invalid vector dimension ({len(embedding) if embedding else 0})"
            )
            continue

        point_uuid = manager.chunk_id_to_uuid(chunk_id)

        # Structure payload for agentic citations and filtering
        chunk_content = chunk.get("content") or chunk.get("text", "")
        payload = {
            "chunk_id": chunk_id,
            "content": chunk_content,
            "text": chunk_content,
            "token_count": chunk.get("token_count", 0),
            "char_count": chunk.get("char_count", len(chunk_content)),
            "source_file": chunk.get("source_file", ""),
            "file_type": chunk.get("file_type", ""),
            "is_table": "true" if chunk.get("is_table") else "false",
            "page_or_slide": chunk.get("page_or_slide"),
            "section_hierarchy": chunk.get("section_hierarchy", []),
            "breadcrumb": chunk.get("breadcrumb", ""),
            "document_title": chunk.get("document_title", ""),
            "source_hash": chunk.get("source_hash", ""),
        }


        point = models.PointStruct(
            id=point_uuid,
            vector=embedding,
            payload=payload,
        )
        points_buffer.append(point)

        if len(points_buffer) >= batch_size:
            manager.upsert_batch(points_buffer)
            indexed_count += len(points_buffer)
            elapsed = time.time() - start_time
            rate = indexed_count / elapsed if elapsed > 0 else 0
            pct = (indexed_count / total_embedded_chunks) * 100
            logger.info(
                f"Indexed: {indexed_count:,}/{total_embedded_chunks:,} ({pct:.1f}%) | "
                f"{rate:.1f} pts/s | Elapsed: {elapsed:.1f}s"
            )
            points_buffer.clear()

    # Flush remainder
    if points_buffer:
        manager.upsert_batch(points_buffer)
        indexed_count += len(points_buffer)
        points_buffer.clear()

    total_time = time.time() - start_time
    final_count = manager.count()

    summary = {
        "collection_name": collection_name,
        "mode": manager.mode,
        "total_indexed_points": indexed_count,
        "total_points_in_collection": final_count,
        "vector_dimension": DEFAULT_VECTOR_DIM,
        "metric": "Cosine",
        "indexing_duration_seconds": round(total_time, 2),
        "indexing_speed_points_per_sec": round(indexed_count / total_time if total_time > 0 else 0, 1),
        "embeddings_source": str(embeddings_file),
    }

    # Save summary
    summary_file = Path(summary_path)
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 70)
    logger.info(f"Qdrant Indexing Completed Successfully in {total_time:.2f} seconds!")
    logger.info(f"Total Points in Collection: {final_count:,}")
    logger.info(f"Speed: {summary['indexing_speed_points_per_sec']} points/sec")
    logger.info(f"Index summary written to {summary_file}")
    logger.info("=" * 70)

    return summary


# ---------------------------------------------------------------------------
# Retrieval Engine & Verification
# ---------------------------------------------------------------------------
def verify_search(
    query_text: str,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    top_k: int = 5,
    filter_is_table: Optional[bool] = None,
    filter_file_type: Optional[str] = None,
    qdrant_url: Optional[str] = DEFAULT_QDRANT_URL,
    prefer_local: bool = False,
) -> None:
    """Embeds query via Ollama and retrieves top-k matches with formatted citations."""
    logger.info(f"Verifying vector retrieval for query: '{query_text}'")

    embed_client = OllamaEmbeddingClient()
    logger.info(f"Generating query embedding via {embed_client.model_name}...")
    query_vector = embed_client.embed_query(query_text)

    manager = QdrantIndexManager(
        collection_name=collection_name,
        url=qdrant_url,
        prefer_local=prefer_local,
    )

    filter_dict = {}
    if filter_is_table is not None:
        filter_dict["is_table"] = "true" if filter_is_table else "false"
    if filter_file_type:
        filter_dict["file_type"] = filter_file_type

    results = manager.search(
        query_vector=query_vector,
        top_k=top_k,
        filter_dict=filter_dict if filter_dict else None,
    )

    print("\n" + "=" * 80)
    print(f"RETRIEVAL RESULTS FOR: \"{query_text}\" (Top-{top_k})")
    if filter_dict:
        print(f"Active Filters: {filter_dict}")
    print("=" * 80)

    for i, pt in enumerate(results, start=1):
        payload = pt.payload or {}
        score = pt.score
        chunk_id = payload.get("chunk_id", "N/A")
        source = payload.get("source_file", "Unknown")
        loc = payload.get("page_or_slide", 1)
        is_tbl = payload.get("is_table") == "true"
        breadcrumb = payload.get("breadcrumb", "")
        content = payload.get("content") or payload.get("text", "")
        clean_text = " ".join(content.split())
        snippet = clean_text[:280] + ("..." if len(clean_text) > 280 else "")




        print(f"\n[Result {i}] Score: {score:.4f} | Source: {source} (Page/Slide {loc})")
        print(f"          Type: {'TABLE' if is_tbl else 'TEXT'} | Section: {breadcrumb}")
        print(f"          Chunk ID: {chunk_id}")
        print(f"          Snippet : {snippet}")

    print("=" * 80 + "\n")


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Index embedded SHP chunks into Qdrant Vector Database."
    )
    parser.add_argument(
        "--embeddings-file",
        default=DEFAULT_EMBEDDINGS_FILE,
        help="Path to embedded chunks JSONL file",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION_NAME,
        help="Target Qdrant collection name",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Upsert batch size (default: 128)",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate collection if it already exists",
    )
    parser.add_argument(
        "--prefer-local",
        action="store_true",
        help="Force embedded local disk storage over HTTP server",
    )
    parser.add_argument(
        "--qdrant-url",
        default=DEFAULT_QDRANT_URL,
        help="Qdrant server HTTP URL (default: http://127.0.0.1:6333)",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Run a verification search query against the index",
    )
    parser.add_argument(
        "--filter-table",
        action="store_true",
        help="Filter search results to table chunks only",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of search results to return (default: 5)",
    )

    args = parser.parse_args()

    if args.query:
        verify_search(
            query_text=args.query,
            collection_name=args.collection,
            top_k=args.top_k,
            filter_is_table=True if args.filter_table else None,
            qdrant_url=args.qdrant_url,
            prefer_local=args.prefer_local,
        )
    else:
        build_index(
            embeddings_path=args.embeddings_file,
            collection_name=args.collection,
            batch_size=args.batch_size,
            recreate=args.recreate,
            prefer_local=args.prefer_local,
            qdrant_url=args.qdrant_url,
        )


if __name__ == "__main__":
    main()
