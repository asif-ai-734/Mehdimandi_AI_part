"""
Drawing-aware PDF processing.

This module turns architectural PDF pages into structured analysis chunks before
embedding, so retrieval can reason over visual meaning, sheet IDs, schedules,
callouts, and construction constraints instead of raw OCR/parser fragments only.
"""

from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from app.config import settings
from app.services.openai_service import OpenAIService


PAGE_ANALYSIS_SYSTEM_PROMPT = """You are an expert architectural drawing analyst.
Analyze construction drawing sheets using both the rendered page image and parser text.
Return one JSON object only. Do not include markdown.
Do not invent IDs. If the sheet number or title is not visible, use "UNKNOWN" or "".
Prefer concise, searchable phrases over prose paragraphs.

Required schema:
{
  "sheet_number": "AR-201",
  "title": "North Elevation",
  "page": 4,
  "summary": "...",
  "scope_items": [],
  "keynotes": [],
  "visible_callouts": [],
  "tables": [],
  "materials": [],
  "locations": [],
  "risks_or_constraints": []
}

For keynotes, schedules, callouts, and tables, preserve exact IDs such as 3.23,
W126, D101, A-051, and AR-402/7 whenever visible.
Tables may be objects with title, headers, and rows when that structure is visible."""


DOCUMENT_SYNTHESIS_SYSTEM_PROMPT = """You synthesize per-sheet architectural drawing analysis.
Return one JSON object only. Do not include markdown.
Preserve exact sheet numbers, keynote IDs, window IDs, door IDs, and detail references.

Required schema:
{
  "overall_project_scope": "",
  "important_work_packages": [],
  "sheets": [],
  "all_keynotes": [],
  "window_door_schedule_highlights": [],
  "construction_constraints": [],
  "normalized_entities": {
    "sheets": [],
    "keynotes": [],
    "windows": [],
    "doors": [],
    "details": []
  }
}"""


SHEET_PATTERN = re.compile(r"\b[A-Z]{1,3}-\d{2,4}[A-Z]?\b")
KEYNOTE_PATTERN = re.compile(r"\b\d{1,2}\.\d{2}\b")
WINDOW_PATTERN = re.compile(r"\bW\d{2,4}[A-Z]?\b", re.IGNORECASE)
DOOR_PATTERN = re.compile(r"\bD\d{2,4}[A-Z]?\b", re.IGNORECASE)
DETAIL_PATTERN = re.compile(
    r"\b(?:[A-Z]{1,3}-\d{2,4}/\d+[A-Z]?|\d+[A-Z]?/[A-Z]{1,3}-\d{2,4})\b",
    re.IGNORECASE,
)


@dataclass
class PdfPageAsset:
    """Rendered image and parser text for one PDF page."""

    page: int
    raw_text: str
    image_path: Optional[str]


@dataclass
class DrawingChunk:
    """Text plus payload metadata ready for embedding."""

    text: str
    metadata: Dict[str, Any]


class DrawingPdfProcessor:
    """OpenAI-assisted processor for architectural PDF drawings."""

    def __init__(self, openai_service: OpenAIService):
        self.openai_service = openai_service

    def process_pdf(
        self,
        file_path: str,
        document_id: int,
        filename: str,
        project_id: int,
        user_id: int,
    ) -> List[DrawingChunk]:
        """Render, analyze, synthesize, normalize, and chunk a PDF drawing set."""
        pages = self.extract_pages(
            file_path=file_path,
            document_id=document_id,
            project_id=project_id,
            user_id=user_id,
        )
        if not pages:
            return []

        page_records = []
        for page in pages:
            analysis = self.analyze_page(page, filename)
            merged_text = " ".join(
                [
                    page.raw_text or "",
                    json.dumps(analysis, ensure_ascii=False),
                ]
            )
            entities = extract_entities(merged_text, sheet_number=analysis.get("sheet_number"))
            page_records.append(
                {
                    "page": page.page,
                    "raw_text": page.raw_text,
                    "image_path": page.image_path,
                    "analysis": analysis,
                    "entities": entities,
                }
            )

        document_summary = self.synthesize_document(filename, page_records)
        return build_drawing_chunks(
            filename=filename,
            project_id=project_id,
            page_records=page_records,
            document_summary=document_summary,
        )

    def extract_pages(
        self,
        file_path: str,
        document_id: int,
        project_id: int,
        user_id: int,
    ) -> List[PdfPageAsset]:
        """Extract parser text and render each PDF page as a high-resolution PNG."""
        try:
            return self._extract_pages_with_pymupdf(
                file_path=file_path,
                document_id=document_id,
                project_id=project_id,
                user_id=user_id,
            )
        except ImportError:
            return self._extract_pages_with_pypdf2(file_path)

    def _extract_pages_with_pymupdf(
        self,
        file_path: str,
        document_id: int,
        project_id: int,
        user_id: int,
    ) -> List[PdfPageAsset]:
        try:
            import fitz
        except ImportError as exc:
            raise ImportError("PyMuPDF is required to render PDF pages") from exc

        image_dir = Path(file_path).parent / f"document_{document_id}_pages"
        image_dir.mkdir(parents=True, exist_ok=True)

        pages: List[PdfPageAsset] = []
        with fitz.open(file_path) as pdf:
            for page_index, page in enumerate(pdf):
                page_number = page_index + 1
                image_path = image_dir / f"page_{page_number:04d}.png"
                raw_text = page.get_text("text") or ""
                self._render_page_to_png(page, str(image_path), settings.pdf_render_dpi)
                pages.append(
                    PdfPageAsset(
                        page=page_number,
                        raw_text=raw_text.strip(),
                        image_path=str(image_path),
                    )
                )

        return pages

    def _render_page_to_png(self, page: Any, image_path: str, starting_dpi: int) -> None:
        """Render a page, reducing DPI if the image would be too large for analysis."""
        import fitz

        dpi_candidates = [starting_dpi, 180, 150, 120]
        seen = set()
        for dpi in dpi_candidates:
            if dpi in seen:
                continue
            seen.add(dpi)
            scale = max(dpi, 72) / 72
            matrix = fitz.Matrix(scale, scale)
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            pixmap.save(image_path)
            if os.path.getsize(image_path) <= settings.max_page_image_bytes:
                return

    def _extract_pages_with_pypdf2(self, file_path: str) -> List[PdfPageAsset]:
        try:
            import PyPDF2
        except ImportError as exc:
            raise ImportError("PyPDF2 is required to extract PDF text") from exc

        pages: List[PdfPageAsset] = []
        with open(file_path, "rb") as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page_index, page in enumerate(pdf_reader.pages):
                raw_text = page.extract_text() or ""
                pages.append(
                    PdfPageAsset(
                        page=page_index + 1,
                        raw_text=raw_text.strip(),
                        image_path=None,
                    )
                )
        return pages

    def analyze_page(self, page: PdfPageAsset, filename: str) -> Dict[str, Any]:
        """Ask OpenAI for structured JSON for a single sheet/page."""
        user_text = f"""Analyze this PDF page as an architectural drawing sheet.

Filename: {filename}
PDF page number: {page.page}

Parser text:
{truncate(page.raw_text, 18000)}

Return the required JSON schema. Set "page" to {page.page}."""

        content: List[Dict[str, Any]] = [{"type": "text", "text": user_text}]
        if page.image_path:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_to_data_url(page.image_path),
                        "detail": "high",
                    },
                }
            )

        analysis = self.openai_service.generate_json(
            messages=[{"role": "user", "content": content}],
            system_prompt=PAGE_ANALYSIS_SYSTEM_PROMPT,
            model=settings.openai_vision_model,
            max_tokens=3000,
        )
        return normalize_page_analysis(analysis, page.page)

    def synthesize_document(
        self,
        filename: str,
        page_records: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Ask OpenAI for a document-level synthesis after all pages are analyzed."""
        compact_pages = []
        for record in page_records:
            analysis = record["analysis"]
            compact_pages.append(
                {
                    "page": record["page"],
                    "sheet_number": analysis.get("sheet_number", "UNKNOWN"),
                    "title": analysis.get("title", ""),
                    "summary": analysis.get("summary", ""),
                    "scope_items": analysis.get("scope_items", []),
                    "keynotes": analysis.get("keynotes", []),
                    "visible_callouts": analysis.get("visible_callouts", []),
                    "tables": analysis.get("tables", []),
                    "materials": analysis.get("materials", []),
                    "locations": analysis.get("locations", []),
                    "risks_or_constraints": analysis.get("risks_or_constraints", []),
                    "entities": record.get("entities", []),
                }
            )

        synthesis = self.openai_service.generate_json(
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Synthesize this drawing set: {filename}\n\n"
                        f"Per-page JSON:\n{json.dumps(compact_pages, ensure_ascii=False)}"
                    ),
                }
            ],
            system_prompt=DOCUMENT_SYNTHESIS_SYSTEM_PROMPT,
            model=settings.openai_model,
            max_tokens=3500,
        )
        return normalize_document_summary(synthesis, page_records)


def build_drawing_chunks(
    filename: str,
    project_id: int,
    page_records: List[Dict[str, Any]],
    document_summary: Dict[str, Any],
) -> List[DrawingChunk]:
    """Create semantic chunks and metadata from structured drawing analysis."""
    chunks: List[DrawingChunk] = []
    all_entities = sorted(
        {
            entity
            for record in page_records
            for entity in record.get("entities", [])
        }.union(set(flatten_entities(document_summary.get("normalized_entities", {}))))
    )

    doc_text = "\n".join(
        compact_lines(
            [
                f"Document summary for {filename}.",
                f"Overall project scope: {document_summary.get('overall_project_scope', '')}",
                list_sentence("Important work packages", document_summary.get("important_work_packages", [])),
                list_sentence("Sheets", document_summary.get("sheets", [])),
                list_sentence("All keynotes", document_summary.get("all_keynotes", [])),
                list_sentence(
                    "Window and door schedule highlights",
                    document_summary.get("window_door_schedule_highlights", []),
                ),
                list_sentence("Construction constraints", document_summary.get("construction_constraints", [])),
            ]
        )
    )
    add_chunk(
        chunks,
        doc_text,
        {
            "project": str(project_id),
            "chunk_type": "document_summary",
            "entities": all_entities,
            "source_text_ref": filename,
        },
    )

    for record in page_records:
        analysis = record["analysis"]
        sheet = clean_unknown(analysis.get("sheet_number"))
        title = analysis.get("title", "")
        page = record["page"]
        image_path = record.get("image_path")
        page_entities = sorted(set(record.get("entities", [])))
        base_metadata = {
            "project": str(project_id),
            "sheet": sheet,
            "page": page,
            "source_image": image_path,
            "source_text_ref": f"{filename}#page={page}",
            "entities": page_entities,
        }

        sheet_text = "\n".join(
            compact_lines(
                [
                    f"Sheet {sheet or 'UNKNOWN'}: {title} (page {page}).",
                    f"Summary: {analysis.get('summary', '')}",
                    list_sentence("Scope", analysis.get("scope_items", [])),
                    list_sentence("Keynotes", analysis.get("keynotes", [])),
                    list_sentence("Materials", analysis.get("materials", [])),
                    list_sentence("Locations", analysis.get("locations", [])),
                    list_sentence("Visible callouts", analysis.get("visible_callouts", [])),
                    list_sentence("Risks or constraints", analysis.get("risks_or_constraints", [])),
                    raw_text_sentence(record.get("raw_text", "")),
                ]
            )
        )
        add_chunk(
            chunks,
            sheet_text,
            {**base_metadata, "chunk_type": "sheet_summary"},
        )

        add_item_chunks(
            chunks,
            analysis.get("scope_items", []),
            base_metadata,
            "scope_item",
            f"Scope item on sheet {sheet or 'UNKNOWN'}, page {page}",
        )
        add_item_chunks(
            chunks,
            analysis.get("keynotes", []),
            base_metadata,
            "keynote",
            f"Keynote on sheet {sheet or 'UNKNOWN'}, page {page}",
        )
        add_item_chunks(
            chunks,
            analysis.get("visible_callouts", []),
            base_metadata,
            "visual_observation",
            f"Visible callout on sheet {sheet or 'UNKNOWN'}, page {page}",
        )
        add_item_chunks(
            chunks,
            analysis.get("materials", []),
            base_metadata,
            "scope_item",
            f"Material noted on sheet {sheet or 'UNKNOWN'}, page {page}",
        )
        add_item_chunks(
            chunks,
            analysis.get("locations", []),
            base_metadata,
            "visual_observation",
            f"Location noted on sheet {sheet or 'UNKNOWN'}, page {page}",
        )
        add_item_chunks(
            chunks,
            analysis.get("risks_or_constraints", []),
            base_metadata,
            "risk_or_constraint",
            f"Risk or constraint on sheet {sheet or 'UNKNOWN'}, page {page}",
        )
        add_table_chunks(chunks, analysis.get("tables", []), base_metadata, sheet, page)
        add_detail_chunks(chunks, page_entities, base_metadata, sheet, page)

    return dedupe_chunks(chunks)


def add_table_chunks(
    chunks: List[DrawingChunk],
    tables: Iterable[Any],
    base_metadata: Dict[str, Any],
    sheet: Optional[str],
    page: int,
) -> None:
    for table in ensure_list(tables):
        table_text = stringify_item(table)
        if not table_text:
            continue

        rows = table.get("rows") if isinstance(table, dict) else None
        title = table.get("title", "Table") if isinstance(table, dict) else "Table"
        if rows:
            for row in ensure_list(rows):
                row_text = stringify_item(row)
                if row_text:
                    chunk_type = "schedule_row" if looks_like_schedule(f"{title} {row_text}") else "visual_observation"
                    add_chunk(
                        chunks,
                        f"{title} row on sheet {sheet or 'UNKNOWN'}, page {page}: {row_text}",
                        enrich_metadata(base_metadata, chunk_type, row_text),
                    )
            continue

        chunk_type = "schedule_row" if looks_like_schedule(table_text) else "visual_observation"
        add_chunk(
            chunks,
            f"{title} on sheet {sheet or 'UNKNOWN'}, page {page}: {table_text}",
            enrich_metadata(base_metadata, chunk_type, table_text),
        )


def add_detail_chunks(
    chunks: List[DrawingChunk],
    entities: Iterable[str],
    base_metadata: Dict[str, Any],
    sheet: Optional[str],
    page: int,
) -> None:
    for entity in entities:
        if DETAIL_PATTERN.fullmatch(entity):
            add_chunk(
                chunks,
                f"Construction detail reference {entity} visible on sheet {sheet or 'UNKNOWN'}, page {page}.",
                enrich_metadata(base_metadata, "construction_detail", entity),
            )


def add_item_chunks(
    chunks: List[DrawingChunk],
    items: Iterable[Any],
    base_metadata: Dict[str, Any],
    chunk_type: str,
    prefix: str,
) -> None:
    for item in ensure_list(items):
        item_text = stringify_item(item)
        if item_text:
            add_chunk(
                chunks,
                f"{prefix}: {item_text}",
                enrich_metadata(base_metadata, chunk_type, item_text),
            )


def add_chunk(chunks: List[DrawingChunk], text: str, metadata: Dict[str, Any]) -> None:
    text = normalize_space(text)
    if not text:
        return
    chunks.append(DrawingChunk(text=text, metadata=metadata))


def enrich_metadata(base_metadata: Dict[str, Any], chunk_type: str, text: str) -> Dict[str, Any]:
    entities = sorted(set(base_metadata.get("entities", [])).union(extract_entities(text)))
    return {**base_metadata, "chunk_type": chunk_type, "entities": entities}


def dedupe_chunks(chunks: List[DrawingChunk]) -> List[DrawingChunk]:
    seen = set()
    deduped = []
    for chunk in chunks:
        key = (
            chunk.metadata.get("chunk_type"),
            chunk.metadata.get("sheet"),
            chunk.metadata.get("page"),
            chunk.text,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(chunk)
    return deduped


def normalize_page_analysis(payload: Dict[str, Any], page: int) -> Dict[str, Any]:
    normalized = {
        "sheet_number": clean_unknown(payload.get("sheet_number")) or "UNKNOWN",
        "title": stringify_scalar(payload.get("title")),
        "page": page,
        "summary": stringify_scalar(payload.get("summary")),
        "scope_items": ensure_list(payload.get("scope_items")),
        "keynotes": ensure_list(payload.get("keynotes")),
        "visible_callouts": ensure_list(payload.get("visible_callouts")),
        "tables": ensure_list(payload.get("tables")),
        "materials": ensure_list(payload.get("materials")),
        "locations": ensure_list(payload.get("locations")),
        "risks_or_constraints": ensure_list(payload.get("risks_or_constraints")),
    }
    return normalized


def normalize_document_summary(
    payload: Dict[str, Any],
    page_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    normalized_entities = payload.get("normalized_entities", {})
    if not isinstance(normalized_entities, dict):
        normalized_entities = {}

    fallback_sheets = [
        {
            "sheet_number": record["analysis"].get("sheet_number", "UNKNOWN"),
            "title": record["analysis"].get("title", ""),
            "page": record["page"],
            "summary": record["analysis"].get("summary", ""),
        }
        for record in page_records
    ]
    deterministic_entities = {
        "sheets": sorted(
            {
                record["analysis"].get("sheet_number")
                for record in page_records
                if clean_unknown(record["analysis"].get("sheet_number"))
            }
        ),
        "keynotes": [],
        "windows": [],
        "doors": [],
        "details": [],
    }
    for record in page_records:
        text = " ".join(
            [
                record.get("raw_text", ""),
                json.dumps(record.get("analysis", {}), ensure_ascii=False),
            ]
        )
        deterministic_entities["keynotes"].extend(KEYNOTE_PATTERN.findall(text))
        deterministic_entities["windows"].extend(WINDOW_PATTERN.findall(text))
        deterministic_entities["doors"].extend(DOOR_PATTERN.findall(text))
        deterministic_entities["details"].extend(DETAIL_PATTERN.findall(text))

    for key, values in deterministic_entities.items():
        combined = ensure_list(normalized_entities.get(key)) + values
        normalized_entities[key] = sorted({stringify_scalar(value).upper() for value in combined if stringify_scalar(value)})

    return {
        "overall_project_scope": stringify_scalar(payload.get("overall_project_scope")),
        "important_work_packages": ensure_list(payload.get("important_work_packages")),
        "sheets": ensure_list(payload.get("sheets")) or fallback_sheets,
        "all_keynotes": ensure_list(payload.get("all_keynotes")),
        "window_door_schedule_highlights": ensure_list(payload.get("window_door_schedule_highlights")),
        "construction_constraints": ensure_list(payload.get("construction_constraints")),
        "normalized_entities": normalized_entities,
    }


def extract_entities(text: str, sheet_number: Optional[str] = None) -> List[str]:
    entities = set()
    if clean_unknown(sheet_number):
        entities.add(clean_unknown(sheet_number).upper())
    for pattern in [SHEET_PATTERN, KEYNOTE_PATTERN, WINDOW_PATTERN, DOOR_PATTERN, DETAIL_PATTERN]:
        entities.update(match.upper() for match in pattern.findall(text or ""))
    return sorted(entities)


def flatten_entities(value: Any) -> List[str]:
    if isinstance(value, dict):
        flattened = []
        for child in value.values():
            flattened.extend(flatten_entities(child))
        return flattened
    if isinstance(value, list):
        flattened = []
        for child in value:
            flattened.extend(flatten_entities(child))
        return flattened
    scalar = stringify_scalar(value)
    return [scalar.upper()] if scalar else []


def image_to_data_url(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def truncate(value: str, max_chars: int) -> str:
    value = value or ""
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + "\n[truncated]"


def ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def stringify_item(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return normalize_space(value)
    if isinstance(value, dict):
        parts = []
        for key, child in value.items():
            child_text = stringify_item(child)
            if child_text:
                parts.append(f"{key}: {child_text}")
        return normalize_space("; ".join(parts))
    if isinstance(value, list):
        return normalize_space("; ".join(stringify_item(child) for child in value if stringify_item(child)))
    return normalize_space(str(value))


def stringify_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return stringify_item(value)
    return normalize_space(str(value))


def list_sentence(label: str, values: Iterable[Any]) -> str:
    items = [stringify_item(value) for value in ensure_list(values)]
    items = [item for item in items if item]
    if not items:
        return ""
    return f"{label}: " + "; ".join(items)


def raw_text_sentence(raw_text: str) -> str:
    raw_text = normalize_space(raw_text)
    if not raw_text:
        return ""
    return f"Raw parser text excerpt: {truncate(raw_text, 1200)}"


def compact_lines(lines: Iterable[str]) -> List[str]:
    return [line for line in lines if line and line.strip()]


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def clean_unknown(value: Any) -> Optional[str]:
    text = stringify_scalar(value).strip()
    if not text or text.upper() in {"UNKNOWN", "N/A", "NA", "NONE", "NULL"}:
        return None
    return text.upper()


def looks_like_schedule(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in ["schedule", "window", "door", "wdw", "glazing", "frame"])
