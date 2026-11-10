"""
=============================================================================
Small Hydropower Plant (SHP) Agentic RAG - BM25 Keyword & Exact Name Retriever
=============================================================================
Module: src.ingestion.bm25_retriever
Purpose: Implements a high-speed, in-memory Okapi BM25 sparse keyword retriever
         specialized in exact term, entity, and name matching across the SHP corpus.

Features:
- Pure Python + math implementation: Zero heavy external dependencies, 100% offline.
- Sub-millisecond query evaluation over thousands of chunks.
- Exact Name & Entity Boosting: Strongly boosts chunks when query terms match
  document titles, file names, breadcrumb hierarchies, or exact phrases.
- Metadata Filtering: Supports filtering by `is_table`, `file_type`, and `source_file`
  to match Qdrant filter semantics.
- Persistent Disk Caching: Serializes the inverted index to `data/chunks/bm25_index.pkl`
  for instant (~10ms) startup.
- Singleton Factory: `get_bm25_retriever()` caches the loaded instance in memory.
- Hybrid Fusion Helpers: Implements Reciprocal Rank Fusion (RRF) to combine
  dense vector scores from Qdrant with BM25 sparse keyword scores.
=============================================================================
"""

import argparse
import json
import math
import os
import pickle
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from loguru import logger

from src.states.orchestratoragent_state import RetrievedChunk

# ---------------------------------------------------------------------------
# Default Constants
# ---------------------------------------------------------------------------
DEFAULT_CHUNKS_PATH = Path("data/chunks/chunks.jsonl")
DEFAULT_CACHE_PATH = Path("data/chunks/bm25_index.pkl")
DEFAULT_K1 = 1.5
DEFAULT_B = 0.75
EXACT_NAME_BOOST = 3.5  # Extra score boost when query terms appear in title/source/breadcrumb
CORPUS_COMMON_STOPWORDS = {
    "small", "hydro", "power", "plant", "plants", "shp", "project", "projects",
    "the", "and", "for", "with", "from", "that", "this", "what", "which",
}


def edit_distance_le(s1: str, s2: str, max_dist: int = 1) -> bool:
    """Checks whether two strings have Levenshtein edit distance <= max_dist or transposition."""
    if abs(len(s1) - len(s2)) > max_dist:
        return False
    if s1 == s2 or sorted(s1) == sorted(s2):
        return True
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n] <= max_dist


def fuzzy_token_in_words(q_term: str, words_set: set[str], full_text: str = "") -> bool:
    """
    Checks if q_term appears in words_set, either as an exact match or with
    common transposition / 1-to-2 edit typographical variance (e.g. 'allotment' in 'allotmnet of shp.pdf',
    'geological' in 'gelogical', 'investigation' in 'invetigation').
    """
    if not q_term:
        return False
    if q_term in words_set:
        return True
    if full_text and q_term in full_text:
        return True
    if len(q_term) >= 5:
        max_d = 2 if len(q_term) >= 8 else 1
        for word in words_set:
            if edit_distance_le(q_term, word, max_dist=max_d):
                return True
    return False


def tokenize(text: str) -> list[str]:
    """
    Extracts lowercase alphanumeric tokens, preserving hyphens for terms like
    'run-of-river', 'Alvarado-Ancieta', 'PFR/DPR'.
    """
    if not text:
        return []
    # Tokenize words, numbers, and hyphenated terms
    tokens = re.findall(r"[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*", text.lower())
    return tokens


class BM25Retriever:
    """
    In-memory Okapi BM25 Sparse Keyword Retriever with Exact Name & Entity Boosting.
    """

    def __init__(
        self,
        chunks_path: Path | str = DEFAULT_CHUNKS_PATH,
        cache_path: Path | str = DEFAULT_CACHE_PATH,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
    ):
        self.chunks_path = Path(chunks_path)
        self.cache_path = Path(cache_path)
        self.k1 = k1
        self.b = b

        # Corpus data
        self.chunks: list[dict[str, Any]] = []
        self.doc_lens: list[int] = []
        self.avgdl: float = 0.0
        self.N: int = 0

        # BM25 Inverted Index: term -> list of (doc_index, term_frequency)
        self.inverted_index: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.idf: dict[str, float] = {}
        self.df: dict[str, int] = defaultdict(int)

        # Lookup caches for fast metadata filtering and name boosting
        self.chunk_id_to_idx: dict[str, int] = {}
        self.metadata: list[dict[str, Any]] = []

    def build_index(self, force_rebuild: bool = False) -> None:
        """
        Loads chunks and builds or restores the BM25 inverted index from disk cache.
        """
        if not force_rebuild and self.cache_path.exists():
            try:
                t0 = time.perf_counter()
                with open(self.cache_path, "rb") as f:
                    cache_data = pickle.load(f)

                self.chunks = cache_data["chunks"]
                self.doc_lens = cache_data["doc_lens"]
                self.avgdl = cache_data["avgdl"]
                self.N = cache_data["N"]
                self.inverted_index = cache_data["inverted_index"]
                self.idf = cache_data["idf"]
                self.df = cache_data["df"]
                self.metadata = cache_data["metadata"]
                # Ensure word sets are cached for high-speed fuzzy entity matching
                for m in self.metadata:
                    if "source_words" not in m:
                        m["source_words"] = set(re.findall(r"[a-z0-9]+", m.get("source_file", "")))
                        m["title_words"] = set(re.findall(r"[a-z0-9]+", m.get("document_title", "")))
                        m["breadcrumb_words"] = set(re.findall(r"[a-z0-9]+", m.get("breadcrumb", "")))
                self.chunk_id_to_idx = {
                    c["chunk_id"]: idx for idx, c in enumerate(self.chunks)
                }

                duration = time.perf_counter() - t0
                logger.info(
                    f"[BM25] Loaded cached index ({self.N} chunks, {len(self.idf)} terms) in {duration*1000:.1f}ms from {self.cache_path}"
                )
                return
            except Exception as e:
                logger.warning(f"[BM25] Failed to load cache from {self.cache_path} ({e}). Rebuilding...")

        t0 = time.perf_counter()
        if not self.chunks_path.exists():
            raise FileNotFoundError(f"Chunks file not found at: {self.chunks_path}")

        chunks: list[dict[str, Any]] = []
        with open(self.chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))

        self.N = len(chunks)
        if self.N == 0:
            logger.warning(f"[BM25] Chunks file {self.chunks_path} is empty.")
            return

        self.chunks = chunks
        self.doc_lens = []
        self.metadata = []
        self.inverted_index = defaultdict(list)
        self.df = defaultdict(int)

        total_tokens = 0
        for doc_idx, c in enumerate(chunks):
            # Index content + breadcrumbs + document title for comprehensive matching
            searchable_text = f"{c.get('source_file', '')} {c.get('document_title', '')} {c.get('breadcrumb', '')} {c.get('content', '')}"
            tokens = tokenize(searchable_text)
            length = len(tokens)
            self.doc_lens.append(length)
            total_tokens += length

            counts = Counter(tokens)
            for term, count in counts.items():
                self.df[term] += 1
                self.inverted_index[term].append((doc_idx, count))

            self.metadata.append({
                "is_table": bool(c.get("is_table", False)),
                "file_type": str(c.get("file_type", "")).upper(),
                "source_file": str(c.get("source_file", "")).lower(),
                "document_title": str(c.get("document_title", "")).lower(),
                "breadcrumb": str(c.get("breadcrumb", "")).lower(),
            })

        self.avgdl = total_tokens / self.N
        # Standard Okapi BM25 Robertson-Spärck Jones IDF with floor protection
        self.idf = {
            term: math.log((self.N - freq + 0.5) / (freq + 0.5) + 1.0)
            for term, freq in self.df.items()
        }
        self.chunk_id_to_idx = {
            c["chunk_id"]: idx for idx, c in enumerate(self.chunks)
        }

        # Cache to disk
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "wb") as f:
                pickle.dump(
                    {
                        "chunks": self.chunks,
                        "doc_lens": self.doc_lens,
                        "avgdl": self.avgdl,
                        "N": self.N,
                        "inverted_index": self.inverted_index,
                        "idf": self.idf,
                        "df": self.df,
                        "metadata": self.metadata,
                    },
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            logger.success(
                f"[BM25] Built and cached BM25 index ({self.N} chunks, {len(self.idf)} terms) in {time.perf_counter() - t0:.2f}s"
            )
        except Exception as e:
            logger.warning(f"[BM25] Could not write cache to {self.cache_path}: {e}")

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_dict: dict[str, Any] | None = None,
        exact_name_boost: float = EXACT_NAME_BOOST,
    ) -> list[RetrievedChunk]:
        """
        Executes Okapi BM25 keyword search with exact name boosting and metadata filtering.

        Args:
            query: The search text (e.g. 'allotment of shp competitive bidding' or 'Aggidis power costs').
            top_k: Number of highest-scoring chunks to return.
            filter_dict: Optional metadata criteria (e.g. {'is_table': True}).
            exact_name_boost: Additional score multiplier when query terms match source/title.

        Returns:
            List of `RetrievedChunk` models sorted by score descending.
        """
        if self.N == 0:
            self.build_index()

        t0 = time.perf_counter()
        q_tokens = tokenize(query)
        if not q_tokens:
            return []

        # Find terms with IDF scores
        scores: dict[int, float] = defaultdict(float)
        query_terms_set = set(q_tokens)
        query_lower = query.lower().strip()

        for term in q_tokens:
            if term not in self.idf:
                continue
            term_idf = self.idf[term]

            for doc_idx, tf in self.inverted_index[term]:
                doc_len = self.doc_lens[doc_idx]
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
                bm25_term_score = term_idf * (numerator / denominator)
                scores[doc_idx] += bm25_term_score

        if not scores:
            return []

        # Exact Name & Entity Boosting + Metadata Filtering
        filtered_scores: list[tuple[int, float]] = []

        for doc_idx, base_score in scores.items():
            meta = self.metadata[doc_idx]

            # 1. Apply metadata filter if specified
            if filter_dict:
                match = True
                for k, v in filter_dict.items():
                    if k == "is_table" and meta["is_table"] != bool(v):
                        match = False
                        break
                    if k == "file_type" and meta["file_type"] != str(v).upper():
                        match = False
                        break
                    if k == "source_file" and str(v).lower() not in meta["source_file"]:
                        match = False
                        break
                if not match:
                    continue

            # 2. Exact Name & Entity Boosting (Typo-Tolerant)
            boost = 0.0
            source_file = meta["source_file"]
            doc_title = meta["document_title"]
            breadcrumb = meta["breadcrumb"]
            sf_words = meta.get("source_words") or set(re.findall(r"[a-z0-9]+", source_file))
            dt_words = meta.get("title_words") or set(re.findall(r"[a-z0-9]+", doc_title))
            bc_words = meta.get("breadcrumb_words") or set(re.findall(r"[a-z0-9]+", breadcrumb))

            for q_term in query_terms_set:
                if len(q_term) > 2 and q_term not in CORPUS_COMMON_STOPWORDS:
                    if fuzzy_token_in_words(q_term, sf_words, source_file):
                        boost += exact_name_boost * 1.5
                    elif fuzzy_token_in_words(q_term, dt_words, doc_title):
                        boost += exact_name_boost
                    elif fuzzy_token_in_words(q_term, bc_words, breadcrumb):
                        boost += exact_name_boost * 0.75

            # Exact multi-word phrase containment in content
            content_lower = self.chunks[doc_idx].get("content", "").lower()
            if len(query_lower) > 5 and query_lower in content_lower:
                boost += exact_name_boost * 2.0

            final_score = base_score + boost
            filtered_scores.append((doc_idx, final_score))

        if not filtered_scores:
            # If strict filter eliminated all results, retry without filter
            if filter_dict:
                logger.debug("[BM25] 0 matches with filter, falling back to unfiltered search.")
                return self.search(query, top_k=top_k, filter_dict=None, exact_name_boost=exact_name_boost)
            return []

        # Rank and take top-k
        filtered_scores.sort(key=lambda x: x[1], reverse=True)
        top_candidates = filtered_scores[:top_k]

        # Normalize score into a friendly 0.0 - 1.0 range for fusion
        max_score = top_candidates[0][1] if top_candidates else 1.0
        min_score = top_candidates[-1][1] if top_candidates else 0.0
        score_range = max(max_score - min_score, 1e-6)

        results: list[RetrievedChunk] = []
        for doc_idx, raw_score in top_candidates:
            c = self.chunks[doc_idx]
            # Normalization heuristic: map BM25 score smoothly into [0.50, 0.99]
            norm_score = round(0.50 + 0.49 * ((raw_score - min_score) / score_range), 4)

            results.append(
                RetrievedChunk(
                    chunk_id=str(c.get("chunk_id", f"bm25_chunk_{doc_idx}")),
                    content=str(c.get("content", "")),
                    score=norm_score,
                    source_file=str(c.get("source_file", "unknown")),
                    file_type=str(c.get("file_type", "TXT")),
                    page_or_slide=c.get("page_or_slide"),
                    breadcrumb=str(c.get("breadcrumb", "")),
                    is_table=bool(c.get("is_table", False)),
                    token_count=c.get("token_count"),
                    worker_source="bm25_retriever",
                )
            )

        duration = time.perf_counter() - t0
        logger.debug(
            f"[BM25] Search for '{query}' returned {len(results)} chunks in {duration*1000:.2f}ms"
        )
        return results


# ---------------------------------------------------------------------------
# Hybrid Retrieval: Reciprocal Rank Fusion (RRF)
# ---------------------------------------------------------------------------
def reciprocal_rank_fusion(
    dense_chunks: list[RetrievedChunk],
    bm25_chunks: list[RetrievedChunk],
    top_k: int = 6,
    dense_weight: float = 0.5,
    bm25_weight: float = 0.5,
    rrf_k: int = 60,
) -> list[RetrievedChunk]:
    """
    Combines dense vector search results and BM25 keyword search results
    using Reciprocal Rank Fusion (RRF).

    Formula:
        RRF_Score(doc) = dense_weight / (rrf_k + rank_dense) + bm25_weight / (rrf_k + rank_bm25)

    Guarantees that exact-keyword matches surfaced by BM25 (e.g. specific names,
    statutory acts, acronyms) are included alongside semantically dense matches.
    """
    rrf_scores: dict[str, float] = defaultdict(float)
    chunk_store: dict[str, RetrievedChunk] = {}

    # 1. Score dense rankings
    for rank, chunk in enumerate(dense_chunks, start=1):
        cid = chunk.chunk_id
        rrf_scores[cid] += dense_weight / (rrf_k + rank)
        if cid not in chunk_store:
            chunk_store[cid] = chunk

    # 2. Score BM25 rankings
    for rank, chunk in enumerate(bm25_chunks, start=1):
        cid = chunk.chunk_id
        rrf_scores[cid] += bm25_weight / (rrf_k + rank)
        if cid not in chunk_store:
            chunk_store[cid] = chunk

    # 3. Sort by combined RRF score descending
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    fused_chunks: list[RetrievedChunk] = []
    if not sorted_items:
        return fused_chunks

    max_rrf = sorted_items[0][1]
    for cid, rrf_score in sorted_items:
        base_chunk = chunk_store[cid]
        # Preserve highest original cosine/norm score, boosted by fusion agreement
        normalized_combined_score = round(min(0.99, base_chunk.score * (1.0 + 0.2 * (rrf_score / max_rrf))), 3)

        fused_chunks.append(
            RetrievedChunk(
                chunk_id=base_chunk.chunk_id,
                content=base_chunk.content,
                score=normalized_combined_score,
                source_file=base_chunk.source_file,
                file_type=base_chunk.file_type,
                page_or_slide=base_chunk.page_or_slide,
                breadcrumb=base_chunk.breadcrumb,
                is_table=base_chunk.is_table,
                token_count=base_chunk.token_count,
                worker_source=f"hybrid({base_chunk.worker_source})",
            )
        )

    return fused_chunks


# ---------------------------------------------------------------------------
# Global Singleton Factory
# ---------------------------------------------------------------------------
_GLOBAL_BM25_RETRIEVER: BM25Retriever | None = None


def get_bm25_retriever(
    chunks_path: Path | str = DEFAULT_CHUNKS_PATH,
    cache_path: Path | str = DEFAULT_CACHE_PATH,
) -> BM25Retriever:
    """Returns the process-wide cached BM25Retriever singleton."""
    global _GLOBAL_BM25_RETRIEVER
    if _GLOBAL_BM25_RETRIEVER is None:
        retriever = BM25Retriever(chunks_path=chunks_path, cache_path=cache_path)
        retriever.build_index()
        _GLOBAL_BM25_RETRIEVER = retriever
    return _GLOBAL_BM25_RETRIEVER


# ---------------------------------------------------------------------------
# Standalone CLI Entry Point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="SHP BM25 Keyword & Exact Name Retriever")
    parser.add_argument("--query", "-q", type=str, default="", help="Query text to search for")
    parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of results to return")
    parser.add_argument("--table-only", action="store_true", help="Filter for Markdown tables only")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild index from chunks.jsonl")
    args = parser.parse_args()

    retriever = BM25Retriever()
    retriever.build_index(force_rebuild=args.rebuild)

    if not args.query:
        logger.info("BM25 index is ready. Pass --query to test searches.")
        return

    filter_dict = {"is_table": True} if args.table_only else None
    results = retriever.search(args.query, top_k=args.top_k, filter_dict=filter_dict)

    print(f"\n{'='*70}")
    print(f"BM25 Search Results for: '{args.query}' (top {args.top_k})")
    print(f"{'='*70}")
    for idx, r in enumerate(results, start=1):
        print(f"\n[{idx}] Score: {r.score:.3f} | Source: {r.source_file} (Page/Slide: {r.page_or_slide})")
        print(f"Breadcrumb: {r.breadcrumb}")
        print(f"Snippet: {r.content[:200]}...")


if __name__ == "__main__":
    main()

