"""
Markdown-Aware Hierarchical & Header-Preserving Semantic Chunker.

Features:
  1. Respects Markdown structural hierarchy (#, ##, ###, ####, ---).
  2. Injects hierarchical breadcrumbs into every chunk (ideal for local/smaller LLMs).
  3. Preserves Markdown tables atomically (repeats header rows if a giant table must split).
  4. Keeps presentation slides at clean slide boundaries.
  5. Computes token counts via tiktoken for precise embedding model budgeting.
  6. Attaches comprehensive citation & provenance metadata to every chunk.
  7. Outputs streaming JSONL (data/chunks/chunks.jsonl) and summary statistics.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, List, Optional

from loguru import logger
import tiktoken

# ── Logging Configuration ─────────────────────────────────────────────────────

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
    LOG_DIR / "chunker.log",
    rotation="20 MB",
    retention="7 days",
    encoding="utf-8",
    level="DEBUG",
)

# ── Defaults & Constants ──────────────────────────────────────────────────────

DEFAULT_PARSED_DIR = Path("data/parsed")
DEFAULT_CHUNKS_DIR = Path("data/chunks")
DEFAULT_MAX_CHARS = 1800      # ~400 - 500 tokens (optimal for Qwen3-Embedding-0.6B)
DEFAULT_MIN_CHARS = 150       # Merge tiny orphaned fragments
DEFAULT_OVERLAP_CHARS = 150   # Overlap between narrative sub-chunks within a section

# Tiktoken tokenizer for accurate token counting
try:
    _TOKENIZER = tiktoken.get_encoding("cl100k_base")
except Exception:
    _TOKENIZER = None


def count_tokens(text: str) -> int:
    """Return token count via tiktoken cl100k_base or fallback word ratio."""
    if _TOKENIZER is not None:
        return len(_TOKENIZER.encode(text, disallowed_special=()))
    # Fallback estimate: ~1.3 tokens per whitespace-separated word
    return int(len(text.split()) * 1.3)


# ── Data Models ───────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """Represents a single semantic chunk enriched with provenance & citation metadata."""
    chunk_id: str
    source_file: str
    file_type: str
    document_title: str
    content: str               # Text with injected breadcrumb context banner
    raw_content: str           # Raw section text without breadcrumb banner
    breadcrumb: str            # Hierarchical section path (e.g. Doc > Section > Sub)
    section_hierarchy: list[str] = field(default_factory=list)
    page_or_slide: Optional[int] = None
    chunk_index: int = 1
    total_chunks: int = 1
    is_table: bool = False
    char_count: int = 0
    token_count: int = 0
    created_at: float = field(default_factory=time.time)


@dataclass
class ChunkingResult:
    """Aggregated summary of a chunking run."""
    total_documents: int = 0
    total_chunks: int = 0
    table_chunks: int = 0
    avg_chars: float = 0.0
    avg_tokens: float = 0.0
    min_tokens: int = 0
    max_tokens: int = 0
    duration_s: float = 0.0
    output_file: str = ""


# ── Markdown Parser & Splitter ────────────────────────────────────────────────

_HEADING_REGEX = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)
_SLIDE_PAGE_REGEX = re.compile(r"(?:Slide|Page)\s*(\d+)", re.IGNORECASE)
_TABLE_ROW_REGEX = re.compile(r"^\s*\|.*\|\s*$")


def _is_table_block(lines: list[str]) -> bool:
    """Check if a sequence of lines forms a Markdown table."""
    if len(lines) < 2:
        return False
    table_lines = [l for l in lines if _TABLE_ROW_REGEX.match(l)]
    return len(table_lines) >= 2 and any("| ---" in l or "|:---" in l or "|---" in l for l in lines)


def _split_table_by_rows(
    table_lines: list[str],
    max_chars: int,
) -> list[str]:
    """
    Split an oversized Markdown table row-by-row, repeating the header on every slice
    so each chunk remains a valid, readable Markdown table.
    """
    if len(table_lines) <= 2:
        return ["\n".join(table_lines)]

    # Locate table header and separator
    header_idx = -1
    for idx, line in enumerate(table_lines[:3]):
        if "| ---" in line or "|:---" in line or "|---" in line:
            header_idx = idx
            break

    if header_idx <= 0:
        return ["\n".join(table_lines)]

    header = "\n".join(table_lines[:header_idx + 1])
    data_rows = table_lines[header_idx + 1:]

    slices: list[str] = []
    current_rows: list[str] = []
    current_chars = len(header)

    for row in data_rows:
        row_len = len(row) + 1
        if current_chars + row_len > max_chars and current_rows:
            slices.append(header + "\n" + "\n".join(current_rows))
            current_rows = [row]
            current_chars = len(header) + row_len
        else:
            current_rows.append(row)
            current_chars += row_len

    if current_rows:
        slices.append(header + "\n" + "\n".join(current_rows))

    return slices


def _split_text_into_paragraphs(
    text: str,
    max_chars: int,
    overlap_chars: int,
) -> list[str]:
    """
    Split large narrative text by double newlines or sentences,
    preserving natural boundaries and applying overlap.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return [text] if text.strip() else []

    chunks: list[str] = []
    current_chunk: list[str] = []
    current_length = 0

    for para in paragraphs:
        para_len = len(para) + 2
        # If single paragraph exceeds max_chars, split by sentence or single line
        if para_len > max_chars:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_length = 0

            sentences = re.split(r"(?<=[.!?])\s+", para)
            s_chunk: list[str] = []
            s_len = 0
            for s in sentences:
                if s_len + len(s) + 1 > max_chars and s_chunk:
                    chunks.append(" ".join(s_chunk))
                    s_chunk = [s]
                    s_len = len(s)
                else:
                    s_chunk.append(s)
                    s_len += len(s) + 1
            if s_chunk:
                chunks.append(" ".join(s_chunk))
            continue

        if current_length + para_len > max_chars and current_chunk:
            chunk_str = "\n\n".join(current_chunk)
            chunks.append(chunk_str)

            # Apply overlap: carry over trailing text if feasible
            if overlap_chars > 0 and len(current_chunk[-1]) <= overlap_chars:
                current_chunk = [current_chunk[-1], para]
                current_length = len(current_chunk[0]) + para_len
            else:
                current_chunk = [para]
                current_length = para_len
        else:
            current_chunk.append(para)
            current_length += para_len

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


# ── Hierarchical Section Chunking ─────────────────────────────────────────────

def chunk_markdown(
    markdown_text: str,
    source_file: str,
    file_type: str,
    document_title: str,
    file_hash: str,
    max_chars: int = DEFAULT_MAX_CHARS,
    min_chars: int = DEFAULT_MIN_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> list[Chunk]:
    """
    Parse Markdown document into semantic chunks with breadcrumb context injection,
    table preservation, and slide boundary tracking.
    """
    raw_lines = markdown_text.splitlines()
    if not raw_lines:
        return []

    # 1. First Pass: Split into Structural Sections based on Headings & Slide Dividers
    sections: list[dict[str, Any]] = []
    current_headings = {1: None, 2: None, 3: None, 4: None}
    current_section_lines: list[str] = []
    current_page_or_slide: Optional[int] = None

    def _flush_section():
        nonlocal current_section_lines
        if current_section_lines:
            text = "\n".join(current_section_lines).strip()
            if text:
                hierarchy = [
                    current_headings[lvl]
                    for lvl in (1, 2, 3, 4)
                    if current_headings[lvl]
                ]
                sections.append({
                    "hierarchy": hierarchy,
                    "page_or_slide": current_page_or_slide,
                    "text": text,
                })
            current_section_lines = []

    for line in raw_lines:
        # Check for slide divider
        if line.strip() == "---":
            _flush_section()
            continue

        # Check for heading
        heading_match = _HEADING_REGEX.match(line)
        if heading_match:
            _flush_section()
            level = len(heading_match.group(1))
            heading_title = heading_match.group(2).strip()

            # Detect page/slide number in heading
            ps_match = _SLIDE_PAGE_REGEX.search(heading_title)
            if ps_match:
                current_page_or_slide = int(ps_match.group(1))

            # Update heading hierarchy stack
            current_headings[level] = heading_title
            for sub_lvl in range(level + 1, 5):
                current_headings[sub_lvl] = None

            current_section_lines.append(line)
            continue

        current_section_lines.append(line)

    _flush_section()

    # 2. Second Pass: Process each section into appropriately sized chunks
    raw_chunks: list[dict[str, Any]] = []

    for sec in sections:
        hierarchy = sec["hierarchy"]
        sec_text = sec["text"]
        page_or_slide = sec["page_or_slide"]

        # Build clean breadcrumb path
        unique_hierarchy: list[str] = []
        for h in hierarchy:
            # Strip markdown formatting
            clean_h = re.sub(r"^[#\s*_-]+", "", h).strip()
            if clean_h and clean_h not in unique_hierarchy and clean_h != document_title:
                unique_hierarchy.append(clean_h)

        breadcrumb_parts = [document_title] + unique_hierarchy
        breadcrumb = " > ".join(breadcrumb_parts)

        # Break section into atomic components (tables vs paragraphs)
        lines = sec_text.splitlines()
        blocks: list[tuple[str, bool]] = []  # (block_text, is_table)
        tbl_buffer: list[str] = []
        text_buffer: list[str] = []

        for line in lines:
            if _TABLE_ROW_REGEX.match(line):
                if text_buffer:
                    blocks.append(("\n".join(text_buffer), False))
                    text_buffer = []
                tbl_buffer.append(line)
            else:
                if tbl_buffer:
                    if _is_table_block(tbl_buffer):
                        blocks.append(("\n".join(tbl_buffer), True))
                    else:
                        text_buffer.extend(tbl_buffer)
                    tbl_buffer = []
                text_buffer.append(line)

        if tbl_buffer:
            if _is_table_block(tbl_buffer):
                blocks.append(("\n".join(tbl_buffer), True))
            else:
                text_buffer.extend(tbl_buffer)
        if text_buffer:
            blocks.append(("\n".join(text_buffer), False))

        # Chunk blocks
        for block_text, is_tbl in blocks:
            if not block_text.strip():
                continue

            if is_tbl:
                tbl_lines = block_text.splitlines()
                if len(block_text) > max_chars:
                    table_slices = _split_table_by_rows(tbl_lines, max_chars)
                    for ts in table_slices:
                        raw_chunks.append({
                            "raw_text": ts,
                            "breadcrumb": breadcrumb,
                            "hierarchy": breadcrumb_parts,
                            "page_or_slide": page_or_slide,
                            "is_table": True,
                        })
                else:
                    raw_chunks.append({
                        "raw_text": block_text,
                        "breadcrumb": breadcrumb,
                        "hierarchy": breadcrumb_parts,
                        "page_or_slide": page_or_slide,
                        "is_table": True,
                    })
            else:
                if len(block_text) > max_chars:
                    para_chunks = _split_text_into_paragraphs(block_text, max_chars, overlap_chars)
                    for pc in para_chunks:
                        if len(pc.strip()) > 30:
                            raw_chunks.append({
                                "raw_text": pc,
                                "breadcrumb": breadcrumb,
                                "hierarchy": breadcrumb_parts,
                                "page_or_slide": page_or_slide,
                                "is_table": False,
                            })
                else:
                    raw_chunks.append({
                        "raw_text": block_text,
                        "breadcrumb": breadcrumb,
                        "hierarchy": breadcrumb_parts,
                        "page_or_slide": page_or_slide,
                        "is_table": False,
                    })

    # 3. Third Pass: Merge small fragments with adjacent chunks of the same section
    merged_chunks: list[dict[str, Any]] = []
    for rc in raw_chunks:
        if (
            merged_chunks
            and not rc["is_table"]
            and not merged_chunks[-1]["is_table"]
            and merged_chunks[-1]["breadcrumb"] == rc["breadcrumb"]
            and (len(rc["raw_text"]) < min_chars or len(merged_chunks[-1]["raw_text"]) + len(rc["raw_text"]) < max_chars)
        ):
            merged_chunks[-1]["raw_text"] += "\n\n" + rc["raw_text"]
        else:
            merged_chunks.append(rc)

    # 4. Final Pass: Construct Chunk objects with context banner injection
    final_chunks: list[Chunk] = []
    total_count = len(merged_chunks)
    file_stem = Path(source_file).stem

    for idx, mc in enumerate(merged_chunks, start=1):
        raw_text = mc["raw_text"].strip()
        bc = mc["breadcrumb"]
        ps = mc["page_or_slide"]

        # Build citation context banner
        banner_parts = [f"Source: {source_file}"]
        if ps is not None:
            banner_parts.append(f"Location: Slide/Page {ps}")
        banner_parts.append(f"Section: {bc}")
        banner = f"[{' | '.join(banner_parts)}]\n\n"

        enriched_content = banner + raw_text
        chunk_id = f"{file_stem}__{file_hash[:8]}__c{idx:04d}"

        final_chunks.append(
            Chunk(
                chunk_id=chunk_id,
                source_file=source_file,
                file_type=file_type,
                document_title=document_title,
                content=enriched_content,
                raw_content=raw_text,
                breadcrumb=bc,
                section_hierarchy=mc["hierarchy"],
                page_or_slide=ps,
                chunk_index=idx,
                total_chunks=total_count,
                is_table=mc["is_table"],
                char_count=len(enriched_content),
                token_count=count_tokens(enriched_content),
            )
        )

    return final_chunks


# ── Batch Chunker Runner ──────────────────────────────────────────────────────

def chunk_corpus(
    parsed_dir: Path = DEFAULT_PARSED_DIR,
    output_dir: Path = DEFAULT_CHUNKS_DIR,
    max_chars: int = DEFAULT_MAX_CHARS,
    min_chars: int = DEFAULT_MIN_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> ChunkingResult:
    """
    Process all parsed Markdown files from parsed_dir, generate semantic chunks,
    and persist streaming JSONL + summary statistics into output_dir.
    """
    t0 = time.perf_counter()
    parsed_dir = Path(parsed_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = parsed_dir / "metadata.json"
    doc_metadata_map: dict[str, dict] = {}

    if metadata_path.exists():
        try:
            meta_list = json.loads(metadata_path.read_text(encoding="utf-8"))
            for rec in meta_list:
                md_path = rec.get("parsed_md_path")
                if md_path:
                    doc_metadata_map[Path(md_path).name] = rec
        except Exception as e:
            logger.warning(f"Could not load metadata.json: {e}")

    # Gather parsed markdown files
    md_files = sorted([f for f in parsed_dir.glob("*.md") if f.name != "metadata.json"])
    if not md_files:
        logger.error(f"No parsed Markdown files found in {parsed_dir}")
        return ChunkingResult()

    logger.info(f"Starting hierarchical semantic chunking on {len(md_files)} documents...")

    all_chunks: list[Chunk] = []
    jsonl_output_path = output_dir / "chunks.jsonl"
    doc_chunk_stats: dict[str, int] = {}

    with open(jsonl_output_path, "w", encoding="utf-8") as jsonl_file:
        for md_path in md_files:
            meta = doc_metadata_map.get(md_path.name, {})
            source_file = meta.get("file_name", md_path.stem.split("__")[0])
            file_type = meta.get("file_type", "Unknown")
            file_hash = meta.get("file_hash", md_path.stem.split("__")[-1])
            doc_title = Path(source_file).stem

            try:
                md_text = md_path.read_text(encoding="utf-8", errors="replace")
                doc_chunks = chunk_markdown(
                    markdown_text=md_text,
                    source_file=source_file,
                    file_type=file_type,
                    document_title=doc_title,
                    file_hash=file_hash,
                    max_chars=max_chars,
                    min_chars=min_chars,
                    overlap_chars=overlap_chars,
                )

                for chk in doc_chunks:
                    jsonl_file.write(json.dumps(asdict(chk), ensure_ascii=False) + "\n")
                    all_chunks.append(chk)

                doc_chunk_stats[source_file] = len(doc_chunks)
                logger.debug(f"[CHUNKED] {source_file}: {len(doc_chunks)} chunks")

            except Exception as e:  # noqa: BLE001
                logger.error(f"[CHUNK FAIL] {source_file}: {e}")

    duration = time.perf_counter() - t0
    total_chunks = len(all_chunks)
    table_chunks = sum(1 for c in all_chunks if c.is_table)
    avg_chars = sum(c.char_count for c in all_chunks) / total_chunks if total_chunks else 0
    avg_tokens = sum(c.token_count for c in all_chunks) / total_chunks if total_chunks else 0
    min_tokens = min((c.token_count for c in all_chunks), default=0)
    max_tokens = max((c.token_count for c in all_chunks), default=0)

    # Persist summary
    summary = {
        "total_documents": len(md_files),
        "total_chunks": total_chunks,
        "table_chunks": table_chunks,
        "text_chunks": total_chunks - table_chunks,
        "avg_chars_per_chunk": round(avg_chars, 1),
        "avg_tokens_per_chunk": round(avg_tokens, 1),
        "min_tokens": min_tokens,
        "max_tokens": max_tokens,
        "duration_seconds": round(duration, 2),
        "chunks_jsonl_path": str(jsonl_output_path),
        "document_chunk_counts": doc_chunk_stats,
    }
    summary_path = output_dir / "chunk_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.success(
        f"Chunking complete — {total_chunks:,} chunks created across {len(md_files)} documents "
        f"({table_chunks} tables, avg {avg_tokens:.0f} tokens/chunk, {duration:.2f}s)"
    )

    return ChunkingResult(
        total_documents=len(md_files),
        total_chunks=total_chunks,
        table_chunks=table_chunks,
        avg_chars=avg_chars,
        avg_tokens=avg_tokens,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        duration_s=duration,
        output_file=str(jsonl_output_path),
    )


# ── CLI Interface ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cli = argparse.ArgumentParser(
        description="Markdown-Aware Hierarchical & Header-Preserving Semantic Chunker."
    )
    cli.add_argument(
        "--parsed-dir", type=Path, default=DEFAULT_PARSED_DIR,
        help="Input directory containing parsed Markdown files.",
    )
    cli.add_argument(
        "--output-dir", type=Path, default=DEFAULT_CHUNKS_DIR,
        help="Output directory for chunks.jsonl and summary.",
    )
    cli.add_argument(
        "--max-chars", type=int, default=DEFAULT_MAX_CHARS,
        help="Target maximum character count per chunk (default: 1800 chars ~400-500 tokens).",
    )
    cli.add_argument(
        "--min-chars", type=int, default=DEFAULT_MIN_CHARS,
        help="Minimum character threshold below which fragments are merged (default: 150).",
    )
    cli.add_argument(
        "--overlap-chars", type=int, default=DEFAULT_OVERLAP_CHARS,
        help="Character overlap for narrative text sections (default: 150).",
    )

    args = cli.parse_args()

    result = chunk_corpus(
        parsed_dir=args.parsed_dir,
        output_dir=args.output_dir,
        max_chars=args.max_chars,
        min_chars=args.min_chars,
        overlap_chars=args.overlap_chars,
    )

    print("\n" + "=" * 55)
    print("  CHUNKING SUMMARY")
    print("=" * 55)
    print(f"  Total Documents : {result.total_documents}")
    print(f"  Total Chunks    : {result.total_chunks:,}")
    print(f"  Table Chunks    : {result.table_chunks:,}")
    print(f"  Avg Tokens/Chunk: {result.avg_tokens:.1f}")
    print(f"  Token Range     : {result.min_tokens} - {result.max_tokens}")
    print(f"  Duration        : {result.duration_s:.2f}s")
    print(f"  Output File     : {result.output_file}")
    print("=" * 55 + "\n")
