"""
parser.py — Smart Multi-Tier Document Parser (Ingestion Phase, Step 1)
=====================================================================
Converts raw SHP documents (PDF, DOCX, DOC, PPT, PPTX, XLS, XLSX,
images, plain text) into clean, structured Markdown.

Architecture (3-Tier "Smart Routing" Pipeline):
  Tier 1 — Office & Text Fast Path:
    - .docx       -> python-docx (instant heading, paragraph & table extraction)
    - .pptx       -> python-pptx (instant slide titles, bullets, tables & notes)
    - .xlsx       -> openpyxl (worksheets to Markdown tables)
    - .xls        -> xlrd (legacy Excel to Markdown tables)
    - .txt, .md   -> Direct UTF-8 decode
    - .doc, .ppt  -> Windows COM / binary text extractor
  Tier 2 — Digital PDF Fast Path:
    - .pdf        -> Fast probe with pypdf. If digital text density >= threshold,
                     extracts clean digital text in ~0.05-0.2s with 100% fidelity.
  Tier 3 — Scanned PDF & Vision Fallback:
    - Scanned PDF -> If digital probe yields < 80 characters, engages Docling
                     with GPU-accelerated EasyOCR (RTX 5050 CUDA) & TableFormer.

Spec Reference: Spec 04 — Document Parser
  Input : Any file under data/raw/
  Output: ParsedDocument dataclass (markdown str + rich metadata)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
from loguru import logger

# Specialized lightweight parsers
import pypdf
import pymupdf
import pymupdf4llm
import docx
import pptx
import openpyxl

try:
    import xlrd
except ImportError:
    xlrd = None  # type: ignore

# Docling vision & OCR imports
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

# OCR backend priority: EasyOCR (PyTorch CUDA) -> RapidOCR -> disabled
try:
    from docling.datamodel.pipeline_options import EasyOcrOptions as _OcrOptions
    _OCR_BACKEND = "easyocr"
except ImportError:
    try:
        from docling.datamodel.pipeline_options import RapidOcrOptions as _OcrOptions  # type: ignore
        _OCR_BACKEND = "rapidocr"
    except ImportError:
        _OcrOptions = None  # type: ignore
        _OCR_BACKEND = "disabled"


# ── Constants & Directories ───────────────────────────────────────────────────

SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".pdf": "PDF",
    ".docx": "Word",
    ".doc": "Word (legacy)",
    ".pptx": "PowerPoint",
    ".ppt": "PowerPoint (legacy)",
    ".xlsx": "Excel",
    ".xls": "Excel (legacy)",
    ".txt": "Plain Text",
    ".md": "Markdown",
    ".png": "Image",
    ".jpg": "Image",
    ".jpeg": "Image",
}

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = _PROJECT_ROOT / "data" / "raw"
DEFAULT_PARSED_DIR = _PROJECT_ROOT / "data" / "parsed"
DEFAULT_METADATA_FILE = _PROJECT_ROOT / "data" / "parsed" / "metadata.json"
DEFAULT_LOGS_DIR = _PROJECT_ROOT / "logs"

# Configure persistent file logging alongside console streaming
DEFAULT_LOGS_DIR.mkdir(parents=True, exist_ok=True)
logger.add(
    DEFAULT_LOGS_DIR / "parser.log",
    rotation="20 MB",
    retention="7 days",
    encoding="utf-8",
    level="DEBUG",
)

MAX_WORKERS: int = max(1, (os.cpu_count() or 2) - 1)


# ── Data Models ───────────────────────────────────────────────────────────────

@dataclass
class ParsedDocument:
    """
    Output contract for the parser. Every downstream component (chunker,
    embedder) consumes this dataclass — do not change field names without
    updating the spec.
    """
    # --- Identity ---
    source_path: str            # Absolute path to the original file
    file_name: str              # Basename of the source file
    file_type: str              # Human-readable format label (e.g. "PDF")
    file_hash: str              # SHA-256 of the source file (for dedup)

    # --- Content ---
    markdown: str               # Full Markdown representation of the document
    page_count: int = 0         # Number of pages / slides / sheets

    # --- Metadata ---
    parsed_at: float = field(default_factory=time.time)  # Unix timestamp
    parse_duration_s: float = 0.0                        # Wall-clock parse time
    parsed_md_path: str = ""    # Path where the .md output was saved
    error: Optional[str] = None # Non-None if parsing failed (partial result)


@dataclass
class ParseBatchResult:
    """Summary returned by parse_directory()."""
    total: int = 0
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0            # Already parsed (cache hit)
    documents: List[ParsedDocument] = field(default_factory=list)
    failed_files: List[str] = field(default_factory=list)


# ── Markdown Table Utility ────────────────────────────────────────────────────

def _rows_to_markdown_table(rows: list[list[str]]) -> str:
    """Convert a 2D list of strings into a standard GitHub-flavored Markdown table."""
    if not rows:
        return ""
    # Filter empty trailing rows
    cleaned_rows = [[str(cell or "").strip().replace("\n", " ") for cell in row] for row in rows]
    cleaned_rows = [r for r in cleaned_rows if any(r)]
    if not cleaned_rows:
        return ""

    num_cols = max(len(r) for r in cleaned_rows)
    # Pad rows to uniform width
    padded = [r + [""] * (num_cols - len(r)) for r in cleaned_rows]

    headers = padded[0]
    separator = ["---"] * num_cols
    body = padded[1:]

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n\n"


# ── In-Memory Image OCR Utility ────────────────────────────────────────────────

try:
    from rapidocr import RapidOCR
    _SHARED_RAPID_OCR = None
except ImportError:
    RapidOCR = None  # type: ignore
    _SHARED_RAPID_OCR = None


def _ocr_image_bytes(image_bytes: bytes) -> str:
    """
    Run lightweight in-memory RapidOCR on embedded picture bytes
    (from a PowerPoint slide, Word figure, or Excel chart/diagram).
    """
    if not image_bytes or len(image_bytes) < 100 or RapidOCR is None:
        return ""
    global _SHARED_RAPID_OCR
    try:
        if _SHARED_RAPID_OCR is None:
            _SHARED_RAPID_OCR = RapidOCR()
        out = _SHARED_RAPID_OCR(image_bytes)
        if out and hasattr(out, "txts") and out.txts:
            words = [t.strip() for t in out.txts if t and len(t.strip()) > 1]
            return " ".join(words).strip()
    except Exception as e:
        logger.debug(f"Image OCR notice: {e}")
    return ""


# ── Tier 1: Specialized Office & Text Parsers ─────────────────────────────────

def _parse_plain_text(
    file_path: Path,
    file_type: str,
    file_hash: str,
    md_out_path: Path,
) -> ParsedDocument:
    """Read plain text / Markdown files directly."""
    t0 = time.perf_counter()
    logger.info(f"[PARSE] {file_type}: {file_path.name}")
    try:
        text = file_path.read_text(encoding="utf-8", errors="replace")
        duration = time.perf_counter() - t0
        md_out_path.parent.mkdir(parents=True, exist_ok=True)
        md_out_path.write_text(text, encoding="utf-8")
        logger.success(f"[DONE] {file_path.name} — {len(text):,} chars, {duration:.2f}s")
        return ParsedDocument(
            source_path=str(file_path),
            file_name=file_path.name,
            file_type=file_type,
            file_hash=file_hash,
            markdown=text,
            page_count=1,
            parse_duration_s=round(duration, 2),
            parsed_md_path=str(md_out_path),
        )
    except Exception as exc:  # noqa: BLE001
        duration = time.perf_counter() - t0
        logger.error(f"[FAIL] {file_path.name} — {exc}")
        return ParsedDocument(
            source_path=str(file_path),
            file_name=file_path.name,
            file_type=file_type,
            file_hash=file_hash,
            markdown="",
            parse_duration_s=round(duration, 2),
            error=str(exc),
        )


def _parse_docx(
    file_path: Path,
    file_type: str,
    file_hash: str,
    md_out_path: Path,
) -> ParsedDocument:
    """Extract headings, paragraphs, and tables from modern .docx files."""
    t0 = time.perf_counter()
    logger.info(f"[PARSE: DOCX] {file_path.name}")
    try:
        doc = docx.Document(str(file_path))
        lines: list[str] = [f"# {file_path.stem}\n"]

        for p in doc.paragraphs:
            text = p.text.strip()
            if not text:
                continue
            style_name = (p.style.name or "").lower()
            if "heading 1" in style_name:
                lines.append(f"## {text}\n")
            elif "heading 2" in style_name:
                lines.append(f"### {text}\n")
            elif "heading 3" in style_name:
                lines.append(f"#### {text}\n")
            elif "list" in style_name:
                lines.append(f"- {text}")
            else:
                lines.append(f"{text}\n")

        for table in doc.tables:
            table_data = []
            for row in table.rows:
                table_data.append([cell.text.strip() for cell in row.cells])
            tbl_md = _rows_to_markdown_table(table_data)
            if tbl_md:
                lines.append(tbl_md)

        # Extract & OCR any embedded figure/image in document parts
        try:
            for rel in doc.part.rels.values():
                if "image" in rel.target_ref:
                    img_bytes = rel.target_part.blob
                    ocr_text = _ocr_image_bytes(img_bytes)
                    if ocr_text:
                        lines.append(f"\n> **[Embedded Figure OCR]**: {ocr_text}\n")
        except Exception as img_err:
            logger.debug(f"Docx embedded image check notice: {img_err}")

        markdown = "\n".join(lines)
        duration = time.perf_counter() - t0
        md_out_path.parent.mkdir(parents=True, exist_ok=True)
        md_out_path.write_text(markdown, encoding="utf-8")
        logger.success(f"[DONE: DOCX] {file_path.name} — {len(markdown):,} chars, {duration:.2f}s")
        return ParsedDocument(
            source_path=str(file_path),
            file_name=file_path.name,
            file_type=file_type,
            file_hash=file_hash,
            markdown=markdown,
            page_count=max(1, len(doc.paragraphs) // 25),
            parse_duration_s=round(duration, 2),
            parsed_md_path=str(md_out_path),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"python-docx failed on {file_path.name} ({exc}), falling back to Docling...")
        return _parse_with_docling(file_path, file_type, file_hash, md_out_path)


def _convert_ppt_to_pptx_via_com(ppt_path: Path) -> Path | None:
    """Convert legacy binary .ppt to modern .pptx using Windows PowerPoint COM."""
    try:
        import gc
        import tempfile
        import win32com.client  # type: ignore

        ppt_app = win32com.client.Dispatch("PowerPoint.Application")
        tmp_dir = tempfile.mkdtemp()
        out_pptx = Path(tmp_dir) / f"{ppt_path.stem}__converted.pptx"
        try:
            pres = ppt_app.Presentations.Open(
                str(ppt_path.resolve()),
                ReadOnly=True,
                Untitled=False,
                WithWindow=False,
            )
            # 24 = ppSaveAsOpenXMLPresentation (.pptx)
            pres.SaveAs(str(out_pptx.resolve()), 24)
            pres.Close()
        finally:
            try:
                ppt_app.Quit()
            except Exception:
                pass
            gc.collect()

        if out_pptx.exists() and out_pptx.stat().st_size > 0:
            return out_pptx
    except Exception as e:
        logger.debug(f"PowerPoint COM .ppt to .pptx conversion notice for {ppt_path.name}: {e}")
    return None


def _parse_presentation(
    file_path: Path,
    file_type: str,
    file_hash: str,
    md_out_path: Path,
) -> ParsedDocument:
    """
    Hybrid Presentation Parser (.pptx and .ppt):
      1. Preserves 100% native table structures as GitHub-Flavored Markdown tables.
      2. Preserves native slide headings, bullet points, and speaker notes.
      3. Performs selective OCR:
         - Embedded charts/diagrams: OCR text appended as a clean blockquote.
         - Image-only slides (scans, full-slide screenshots with <30 chars text):
           OCR text formatted directly as primary slide body.
      4. Legacy .ppt files are seamlessly converted to .pptx via COM before parsing.
    """
    t0 = time.perf_counter()
    suffix = file_path.suffix.lower()
    logger.info(f"[PARSE: PRESENTATION HYBRID] {file_path.name}")

    target_pptx = file_path
    temp_converted_pptx: Path | None = None

    # Handle legacy .ppt by converting to .pptx via COM
    if suffix == ".ppt":
        temp_converted_pptx = _convert_ppt_to_pptx_via_com(file_path)
        if temp_converted_pptx is not None:
            target_pptx = temp_converted_pptx
        else:
            logger.warning(f"COM conversion failed on {file_path.name}, falling back to legacy office parser...")
            return _parse_legacy_office(file_path, file_type, file_hash, md_out_path)

    try:
        prs = pptx.Presentation(str(target_pptx))
        lines: list[str] = [f"# {file_path.stem}\n"]

        for i, slide in enumerate(prs.slides, start=1):
            title = f"Slide {i}"
            if slide.shapes.title and slide.shapes.title.text.strip():
                title = f"Slide {i}: {slide.shapes.title.text.strip()}"
            lines.append(f"## {title}\n")

            slide_text_chars = 0
            slide_has_table = False
            image_shapes = []

            for shape in slide.shapes:
                if shape == slide.shapes.title:
                    continue
                if shape.has_text_frame:
                    for para in shape.text_frame.paragraphs:
                        text = para.text.strip()
                        if text:
                            lines.append(f"- {text}")
                            slide_text_chars += len(text)
                elif shape.has_table:
                    slide_has_table = True
                    table_data = []
                    for row in shape.table.rows:
                        table_data.append([cell.text.strip() for cell in row.cells])
                    tbl_md = _rows_to_markdown_table(table_data)
                    if tbl_md:
                        lines.append(tbl_md)
                elif hasattr(shape, "image"):
                    image_shapes.append(shape)

            # Smart OCR on slide images
            for img_shp in image_shapes:
                try:
                    img_bytes = img_shp.image.blob
                    ocr_text = _ocr_image_bytes(img_bytes)
                    if ocr_text:
                        # Image-only slide (scanned slide, photo, screenshot)
                        if slide_text_chars < 30 and not slide_has_table:
                            lines.append(f"\n{ocr_text}\n")
                        else:
                            # Diagram/chart on an already structured slide
                            lines.append(f"\n> **[Slide {i} Image/Diagram OCR]**: {ocr_text}\n")
                except Exception as s_img_err:
                    logger.debug(f"Slide {i} image OCR notice: {s_img_err}")

            # Speaker notes
            if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
                notes = slide.notes_slide.notes_text_frame.text.strip()
                if notes:
                    lines.append(f"\n> **Notes:** {notes}\n")
            lines.append("\n---\n")

        markdown = "\n".join(lines)
        duration = time.perf_counter() - t0
        md_out_path.parent.mkdir(parents=True, exist_ok=True)
        md_out_path.write_text(markdown, encoding="utf-8")
        logger.success(
            f"[DONE: PRESENTATION HYBRID] {file_path.name} — {len(markdown):,} chars, "
            f"{len(prs.slides)} slides, {duration:.2f}s"
        )
        return ParsedDocument(
            source_path=str(file_path),
            file_name=file_path.name,
            file_type=file_type,
            file_hash=file_hash,
            markdown=markdown,
            page_count=len(prs.slides),
            parse_duration_s=round(duration, 2),
            parsed_md_path=str(md_out_path),
        )

    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Presentation parser failed on {file_path.name} ({exc}), falling back to Docling...")
        return _parse_with_docling(file_path, file_type, file_hash, md_out_path)

    finally:
        # Clean up temporary converted .pptx file
        if temp_converted_pptx is not None:
            try:
                temp_converted_pptx.unlink(missing_ok=True)
                temp_converted_pptx.parent.rmdir()
            except Exception:
                pass


def _parse_xlsx(
    file_path: Path,
    file_type: str,
    file_hash: str,
    md_out_path: Path,
) -> ParsedDocument:
    """Extract sheets and tabular data from .xlsx workbooks into Markdown tables."""
    t0 = time.perf_counter()
    logger.info(f"[PARSE: XLSX] {file_path.name}")
    try:
        wb = openpyxl.load_workbook(str(file_path), data_only=True, read_only=True)
        lines: list[str] = [f"# {file_path.stem}\n"]

        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            rows_data = []
            for row in ws.iter_rows(values_only=True):
                if any(row):
                    rows_data.append([str(c) if c is not None else "" for c in row])
            if rows_data:
                lines.append(f"## Sheet: {sheet_name}\n")
                lines.append(_rows_to_markdown_table(rows_data))

            # Check for embedded images/charts in worksheet
            try:
                if hasattr(ws, "_images"):
                    for img in getattr(ws, "_images", []):
                        img_bytes = img._data()
                        ocr_text = _ocr_image_bytes(img_bytes)
                        if ocr_text:
                            lines.append(f"\n> **[Sheet {sheet_name} Image/Chart OCR]**: {ocr_text}\n")
            except Exception as ws_img_err:
                logger.debug(f"Worksheet image check notice: {ws_img_err}")

        markdown = "\n".join(lines)
        duration = time.perf_counter() - t0
        md_out_path.parent.mkdir(parents=True, exist_ok=True)
        md_out_path.write_text(markdown, encoding="utf-8")
        logger.success(f"[DONE: XLSX] {file_path.name} — {len(markdown):,} chars, {len(wb.sheetnames)} sheets, {duration:.2f}s")
        return ParsedDocument(
            source_path=str(file_path),
            file_name=file_path.name,
            file_type=file_type,
            file_hash=file_hash,
            markdown=markdown,
            page_count=len(wb.sheetnames),
            parse_duration_s=round(duration, 2),
            parsed_md_path=str(md_out_path),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"openpyxl failed on {file_path.name} ({exc}), falling back to Docling...")
        return _parse_with_docling(file_path, file_type, file_hash, md_out_path)


def _parse_legacy_xls(
    file_path: Path,
    file_type: str,
    file_hash: str,
    md_out_path: Path,
) -> ParsedDocument:
    """Extract sheets and data from legacy .xls workbooks using xlrd."""
    t0 = time.perf_counter()
    logger.info(f"[PARSE: XLS] {file_path.name}")
    if xlrd is None:
        logger.warning("xlrd not installed, falling back to Docling...")
        return _parse_with_docling(file_path, file_type, file_hash, md_out_path)

    try:
        wb = xlrd.open_workbook(str(file_path))
        lines: list[str] = [f"# {file_path.stem}\n"]

        for sheet_idx in range(wb.nsheets):
            sheet = wb.sheet_by_index(sheet_idx)
            rows_data = []
            for r in range(sheet.nrows):
                row_vals = [str(sheet.cell_value(r, c)).strip() for c in range(sheet.ncols)]
                if any(row_vals):
                    rows_data.append(row_vals)
            if rows_data:
                lines.append(f"## Sheet: {sheet.name}\n")
                lines.append(_rows_to_markdown_table(rows_data))

        markdown = "\n".join(lines)
        duration = time.perf_counter() - t0
        md_out_path.parent.mkdir(parents=True, exist_ok=True)
        md_out_path.write_text(markdown, encoding="utf-8")
        logger.success(f"[DONE: XLS] {file_path.name} — {len(markdown):,} chars, {wb.nsheets} sheets, {duration:.2f}s")
        return ParsedDocument(
            source_path=str(file_path),
            file_name=file_path.name,
            file_type=file_type,
            file_hash=file_hash,
            markdown=markdown,
            page_count=wb.nsheets,
            parse_duration_s=round(duration, 2),
            parsed_md_path=str(md_out_path),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"xlrd failed on {file_path.name} ({exc}), falling back to Docling...")
        return _parse_with_docling(file_path, file_type, file_hash, md_out_path)


def _parse_legacy_office(
    file_path: Path,
    file_type: str,
    file_hash: str,
    md_out_path: Path,
) -> ParsedDocument:
    """Extract text from legacy binary Office files (.doc, .ppt) via COM or binary stream."""
    t0 = time.perf_counter()
    logger.info(f"[PARSE: LEGACY OFFICE] {file_path.name}")
    suffix = file_path.suffix.lower()

    # Attempt 1: Windows COM Automation if available
    try:
        import win32com.client  # type: ignore
        if suffix == ".doc":
            word = win32com.client.Dispatch("Word.Application")
            word.Visible = False
            doc = word.Documents.Open(str(file_path), ReadOnly=True)
            text = doc.Content.Text
            doc.Close(False)
            word.Quit()
            if text and len(text.strip()) > 50:
                duration = time.perf_counter() - t0
                md_out_path.parent.mkdir(parents=True, exist_ok=True)
                md_out_path.write_text(text, encoding="utf-8")
                logger.success(f"[DONE: COM DOC] {file_path.name} — {len(text):,} chars, {duration:.2f}s")
                return ParsedDocument(
                    source_path=str(file_path),
                    file_name=file_path.name,
                    file_type=file_type,
                    file_hash=file_hash,
                    markdown=text,
                    page_count=1,
                    parse_duration_s=round(duration, 2),
                    parsed_md_path=str(md_out_path),
                )
    except Exception as com_err:  # noqa: BLE001
        logger.debug(f"COM extraction skipped on {file_path.name}: {com_err}")

    # Attempt 2: Extract printable unicode/ASCII text stream segments
    try:
        with open(file_path, "rb") as f:
            raw_data = f.read()
        segments = re.findall(b"[\x20-\x7E\r\n\t]{4,}", raw_data)
        decoded = [s.decode("latin1", errors="ignore").strip() for s in segments if len(s.strip()) > 10]
        filtered = [s for s in decoded if not re.match(r"^[0-9a-fA-F\-_#]{12,}$", s)]
        text = f"# {file_path.stem}\n\n" + "\n\n".join(filtered)
        if len(text.strip()) > 100:
            duration = time.perf_counter() - t0
            md_out_path.parent.mkdir(parents=True, exist_ok=True)
            md_out_path.write_text(text, encoding="utf-8")
            logger.success(f"[DONE: STREAM EXTRACT] {file_path.name} — {len(text):,} chars, {duration:.2f}s")
            return ParsedDocument(
                source_path=str(file_path),
                file_name=file_path.name,
                file_type=file_type,
                file_hash=file_hash,
                markdown=text,
                page_count=1,
                parse_duration_s=round(duration, 2),
                parsed_md_path=str(md_out_path),
            )
    except Exception as bin_err:  # noqa: BLE001
        logger.debug(f"Binary text stream extraction failed: {bin_err}")

    # Attempt 3: Fall back to Docling
    return _parse_with_docling(file_path, file_type, file_hash, md_out_path)


# ── Tier 2 & 3: Smart PDF Routing (Digital Fast Path vs Scanned Vision OCR) ───

def _parse_pdf(
    file_path: Path,
    file_type: str,
    file_hash: str,
    md_out_path: Path,
) -> ParsedDocument:
    """
    Intelligently parse PDF files:
      - Uses pymupdf4llm for structured Markdown with heading hierarchy and native Markdown tables.
      - If output is low/empty (< 80 chars), falls back to GPU Docling OCR.
    """
    t0 = time.perf_counter()
    logger.info(f"[PARSE: PDF via PyMuPDF4LLM] {file_path.name}")

    try:
        # Step 1: Use pymupdf4llm for high-fidelity Markdown with native tables & headings
        markdown = pymupdf4llm.to_markdown(str(file_path))
        clean_text = (markdown or "").strip()

        # Determine page count
        doc = pymupdf.open(str(file_path))
        num_pages = len(doc)
        doc.close()

        # If sufficient digital text is found, save and return
        if len(clean_text) >= 80:
            duration = time.perf_counter() - t0
            md_out_path.parent.mkdir(parents=True, exist_ok=True)
            md_out_path.write_text(markdown, encoding="utf-8")
            logger.success(
                f"[DONE: PyMuPDF4LLM] {file_path.name} — {len(markdown):,} chars, "
                f"{num_pages} pages, {duration:.2f}s"
            )
            return ParsedDocument(
                source_path=str(file_path),
                file_name=file_path.name,
                file_type=file_type,
                file_hash=file_hash,
                markdown=markdown,
                page_count=num_pages,
                parse_duration_s=round(duration, 2),
                parsed_md_path=str(md_out_path),
            )

        # Scanned PDF detected: Fall through to Docling GPU OCR
        logger.info(
            f"[SCANNED PDF DETECTED] {file_path.name}: only {len(clean_text)} chars "
            f"across {num_pages} pages. Engaging GPU Docling OCR..."
        )
        return _parse_with_docling(file_path, file_type, file_hash, md_out_path)

    except Exception as exc:  # noqa: BLE001
        logger.warning(f"pymupdf4llm failed on {file_path.name} ({exc}), delegating to Docling...")
        return _parse_with_docling(file_path, file_type, file_hash, md_out_path)


# ── Docling GPU Converter Factory ─────────────────────────────────────────────

def _build_converter() -> DocumentConverter:
    """
    Construct a Docling DocumentConverter configured for scanned/complex documents:
      - Uses PyPdfiumDocumentBackend (Google Chromium PDFium) for bulletproof stability
      - GPU acceleration via CUDA (RTX 5050) for vision layout & EasyOCR
      - Table structure recovery enabled
    """
    has_cuda = torch.cuda.is_available()
    device = AcceleratorDevice.CUDA if has_cuda else AcceleratorDevice.CPU
    accelerator_opts = AcceleratorOptions(num_threads=4, device=device)

    if _OcrOptions is not None:
        ocr_kwargs: dict = {}
        if _OCR_BACKEND == "easyocr":
            ocr_kwargs = {"lang": ["en"], "use_gpu": has_cuda}
        ocr_options = _OcrOptions(**ocr_kwargs)
        pdf_options = PdfPipelineOptions(
            do_ocr=True,
            do_table_structure=True,
            ocr_options=ocr_options,
            accelerator_options=accelerator_opts,
        )
    else:
        pdf_options = PdfPipelineOptions(
            do_ocr=False,
            do_table_structure=True,
            accelerator_options=accelerator_opts,
        )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pdf_options,
                backend=PyPdfiumDocumentBackend,
            ),
        }
    )
    return converter


def _parse_with_docling(
    file_path: Path,
    file_type: str,
    file_hash: str,
    md_out_path: Path,
) -> ParsedDocument:
    """Run Docling conversion (vision layout + EasyOCR) and return a ParsedDocument."""
    t0 = time.perf_counter()
    logger.info(f"[DOCLING OCR] {file_type}: {file_path.name}")

    try:
        converter = _build_converter()
        result = converter.convert(str(file_path))
        markdown = result.document.export_to_markdown()

        page_count = 0
        try:
            page_count = len(result.document.pages)
        except AttributeError:
            pass

        duration = time.perf_counter() - t0
        logger.success(
            f"[DONE: DOCLING OCR] {file_path.name} — {len(markdown):,} chars, "
            f"{page_count} pages, {duration:.1f}s"
        )

        md_out_path.parent.mkdir(parents=True, exist_ok=True)
        md_out_path.write_text(markdown, encoding="utf-8")

        return ParsedDocument(
            source_path=str(file_path),
            file_name=file_path.name,
            file_type=file_type,
            file_hash=file_hash,
            markdown=markdown,
            page_count=page_count,
            parse_duration_s=round(duration, 2),
            parsed_md_path=str(md_out_path),
        )

    except Exception as exc:  # noqa: BLE001
        duration = time.perf_counter() - t0
        logger.error(f"[FAIL] {file_path.name} — {type(exc).__name__}: {exc}")
        return ParsedDocument(
            source_path=str(file_path),
            file_name=file_path.name,
            file_type=file_type,
            file_hash=file_hash,
            markdown="",
            parse_duration_s=round(duration, 2),
            error=f"{type(exc).__name__}: {exc}",
        )


# ── Core Parser Router ────────────────────────────────────────────────────────

def parse_file(
    file_path: Path,
    parsed_dir: Path,
    force: bool = False,
) -> ParsedDocument:
    """
    Parse a single document file into a ParsedDocument using the 3-tier smart router.

    Args:
        file_path  : Path to the source document.
        parsed_dir : Directory where the .md output is saved.
        force      : If True, re-parse even if a cached .md exists.

    Returns:
        ParsedDocument with markdown content and rich metadata.
    """
    file_path = Path(file_path).resolve()
    suffix = file_path.suffix.lower()
    file_type = SUPPORTED_EXTENSIONS.get(suffix, "Unknown")

    # 1. Compute file hash for deduplication and cache lookup
    file_hash = _sha256(file_path)
    md_out_path = parsed_dir / f"{file_path.stem}__{file_hash[:8]}.md"

    # 2. Cache hit check
    if md_out_path.exists() and not force:
        logger.debug(f"[CACHE HIT] Skipping already-parsed file: {file_path.name}")
        cached_md = md_out_path.read_text(encoding="utf-8")
        return ParsedDocument(
            source_path=str(file_path),
            file_name=file_path.name,
            file_type=file_type,
            file_hash=file_hash,
            markdown=cached_md,
            parsed_md_path=str(md_out_path),
            error=None,
        )

    # 3. Smart routing based on extension & content type
    if suffix in (".txt", ".md"):
        return _parse_plain_text(file_path, file_type, file_hash, md_out_path)

    if suffix == ".docx":
        return _parse_docx(file_path, file_type, file_hash, md_out_path)

    if suffix in (".pptx", ".ppt"):
        return _parse_presentation(file_path, file_type, file_hash, md_out_path)

    if suffix == ".xlsx":
        return _parse_xlsx(file_path, file_type, file_hash, md_out_path)

    if suffix == ".xls":
        return _parse_legacy_xls(file_path, file_type, file_hash, md_out_path)

    if suffix == ".doc":
        return _parse_legacy_office(file_path, file_type, file_hash, md_out_path)

    if suffix == ".pdf":
        return _parse_pdf(file_path, file_type, file_hash, md_out_path)

    # Images and any unhandled formats fall back to Docling
    return _parse_with_docling(file_path, file_type, file_hash, md_out_path)


# ── Batch Parsing ─────────────────────────────────────────────────────────────

def parse_directory(
    raw_dir: Path = DEFAULT_RAW_DIR,
    parsed_dir: Path = DEFAULT_PARSED_DIR,
    force: bool = False,
    workers: int = 1,
    extensions: Optional[list[str]] = None,
) -> ParseBatchResult:
    """
    Parse all supported documents in raw_dir using the smart multi-tier pipeline.

    Args:
        raw_dir    : Source directory containing raw documents.
        parsed_dir : Target directory for .md files and metadata.
        force      : Re-parse all files, ignoring cached outputs.
        workers    : Number of parallel worker processes.
        extensions : Optional list of extensions to filter by.
    """
    raw_dir = Path(raw_dir)
    parsed_dir = Path(parsed_dir)
    parsed_dir.mkdir(parents=True, exist_ok=True)

    allowed_exts = set(extensions or SUPPORTED_EXTENSIONS.keys())

    all_files = [
        f for f in raw_dir.iterdir()
        if f.is_file() and f.suffix.lower() in allowed_exts
    ]

    if not all_files:
        logger.warning(f"No supported files found in: {raw_dir}")
        return ParseBatchResult()

    logger.info(
        f"Starting smart batch parse: {len(all_files)} files in '{raw_dir.name}', "
        f"workers={workers}, force={force}"
    )

    batch = ParseBatchResult(total=len(all_files))

    if workers == 1:
        for f in all_files:
            _process_result(
                parse_file(f, parsed_dir, force=force),
                batch,
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(parse_file, f, parsed_dir, force): f
                for f in all_files
            }
            for future in as_completed(futures):
                try:
                    doc = future.result()
                    _process_result(doc, batch)
                except Exception as exc:  # noqa: BLE001
                    src = futures[future]
                    logger.error(f"[WORKER FAIL] {src.name}: {exc}")
                    batch.failed += 1
                    batch.failed_files.append(str(src))

    _save_metadata(batch, parsed_dir)

    logger.info(
        f"Batch complete — "
        f"✅ {batch.succeeded} parsed, "
        f"⏭ {batch.skipped} skipped (cached), "
        f"❌ {batch.failed} failed"
    )
    return batch


def _process_result(doc: ParsedDocument, batch: ParseBatchResult) -> None:
    """Update batch counters based on parse result."""
    batch.documents.append(doc)
    if doc.error:
        batch.failed += 1
        batch.failed_files.append(doc.source_path)
    elif doc.parsed_md_path and not doc.parse_duration_s:
        batch.skipped += 1
    else:
        batch.succeeded += 1


def _save_metadata(batch: ParseBatchResult, parsed_dir: Path) -> None:
    """Persist a JSON metadata index of all parsed documents, merging with existing records."""
    meta_path = parsed_dir / "metadata.json"
    records_by_hash: dict[str, dict] = {}

    # Load existing records if available
    if meta_path.exists():
        try:
            existing = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(existing, list):
                for r in existing:
                    if "file_hash" in r:
                        records_by_hash[r["file_hash"]] = r
        except Exception as e:
            logger.debug(f"Could not load existing metadata.json for merge: {e}")

    # Upsert current batch records
    for doc in batch.documents:
        d = asdict(doc)
        d.pop("markdown", None)
        records_by_hash[doc.file_hash] = d

    sorted_records = sorted(records_by_hash.values(), key=lambda r: r.get("file_name", ""))
    meta_path.write_text(
        json.dumps(sorted_records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.debug(f"Metadata index saved ({len(sorted_records)} records) → {meta_path}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file for deduplication."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def get_supported_files(raw_dir: Path = DEFAULT_RAW_DIR) -> List[Path]:
    """Return list of all supported files in raw_dir."""
    return [
        f for f in Path(raw_dir).iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


# ── CLI Entry Point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    cli = argparse.ArgumentParser(
        description="Smart Multi-Tier Document Parser (Digital Fast-Path + GPU OCR fallback)."
    )
    cli.add_argument(
        "--raw-dir", type=Path, default=DEFAULT_RAW_DIR,
        help="Directory containing raw source files.",
    )
    cli.add_argument(
        "--parsed-dir", type=Path, default=DEFAULT_PARSED_DIR,
        help="Output directory for parsed Markdown files.",
    )
    cli.add_argument(
        "--force", action="store_true",
        help="Re-parse files even if cached output exists.",
    )
    cli.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel worker processes (default: 1 = sequential).",
    )
    cli.add_argument(
        "--ext", nargs="+", default=None,
        help="Only process these extensions, e.g. --ext .pdf .docx",
    )
    args = cli.parse_args()

    result = parse_directory(
        raw_dir=args.raw_dir,
        parsed_dir=args.parsed_dir,
        force=args.force,
        workers=args.workers,
        extensions=args.ext,
    )

    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print(f"\n{'='*50}")
    print(f"  Total files   : {result.total}")
    print(f"  [OK] Succeeded: {result.succeeded}")
    print(f"  [--] Skipped  : {result.skipped}")
    print(f"  [XX] Failed   : {result.failed}")
    if result.failed_files:
        print("\nFailed files:")
        for f in result.failed_files:
            print(f"  - {f}")
    print(f"{'='*50}\n")
