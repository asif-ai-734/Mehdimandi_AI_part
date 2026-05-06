"""
Scope of work extraction service.

Uses the existing project RAG index as context and asks OpenAI for structured
scope of work items.
"""

import json
import re
from typing import Any, Dict, List, Optional

from app.services.openai_service import get_openai_service
from app.services.rag_service import get_rag_service


CSI_DIVISIONS = {
    "00": "Procurement and Contracting Requirements",
    "01": "General Requirements",
    "02": "Existing Conditions",
    "03": "Concrete",
    "04": "Masonry",
    "05": "Metals",
    "06": "Wood, Plastics & Composites",
    "07": "Thermal and Moisture Protection",
    "08": "Openings",
    "09": "Finishes",
    "10": "Specialties",
    "11": "Equipment",
    "12": "Furnishings",
    "13": "Special Construction",
    "14": "Conveying Equipment",
    "21": "Fire Suppression",
    "22": "Plumbing",
    "23": "HVAC",
    "26": "Electrical",
    "27": "Communications",
    "28": "Electronic Safety and Security",
    "31": "Earthwork",
    "32": "Exterior Improvements",
    "33": "Utilities",
}


SCOPE_OF_WORK_SYSTEM_PROMPT = """You are an expert construction estimator and specification analyst.

Extract structured scope of work items ONLY from the provided tender context.

Rules:
- Use only supported information from retrieved context.
- Focus on selected CSI divisions.
- Do not invent quantities.
- Do not invent references.
- Do not invent specification sections.
- Keep titles concise.
- Group work into meaningful construction scope items.
- Quantities must be numeric when supported.
- If quantity is unknown, use value 0 and unit "unspecified".
- If page is unknown, use null.
- Return JSON only. Do not include markdown.

Required schema:
{
  "items": [
    {
      "title": "Interior Wood Doors",
      "division_code": "08",
      "division_label": "Openings",
      "quantity": {
        "value": 24,
        "unit": "units"
      },
      "specifications": "Solid core birch veneer flush wood doors.",
      "references": [
        {
          "code": "08 14 16",
          "title": "Flush Wood Doors",
          "page": 114,
          "division": "08"
        }
      ]
    }
  ]
}
"""


class SelectedDivision:
    def __init__(self, code: str, label: str):
        self.code = code
        self.label = label


class ScopeService:
    """Coordinates RAG retrieval and OpenAI scope extraction."""

    def __init__(self):
        self.rag_service = get_rag_service()
        self.openai_service = get_openai_service()

    def extract_scope_of_work(
        self,
        user_id: str,
        project_id: str,
        divisions: List[str],
        instructions: str,
    ) -> Dict[str, Any]:
        selected_divisions = build_selected_divisions(divisions)

        context, project_name, project_address = build_scope_context(
            rag_service=self.rag_service,
            user_id=user_id,
            project_id=project_id,
            selected_divisions=selected_divisions,
            instructions=instructions,
        )

        if not context.strip():
            raise ValueError("No relevant uploaded document context found for this project")

        payload = self.openai_service.generate_json(
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Project name: {project_name or 'Not provided'}\n"
                        f"Project address: {project_address or 'Not provided'}\n\n"
                        f"Selected CSI divisions:\n{format_selected_divisions(selected_divisions)}\n\n"
                        f"Estimator instructions:\n{instructions.strip() or 'No additional instructions provided.'}\n\n"
                        f"Retrieved tender document context:\n{context}"
                    ),
                }
            ],
            system_prompt=SCOPE_OF_WORK_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=3500,
        )

        return normalize_scope_extraction_response(payload)


def build_selected_divisions(divisions: List[str]) -> List[SelectedDivision]:
    selected = []
    seen = set()

    for division in divisions:
        code = normalize_division_code(division)

        if not code or code in seen:
            continue

        selected.append(
            SelectedDivision(
                code=code,
                label=CSI_DIVISIONS.get(code, f"Division {code}"),
            )
        )
        seen.add(code)

    return selected


def format_selected_divisions(selected_divisions: List[SelectedDivision]) -> str:
    if not selected_divisions:
        return "All relevant divisions found in the retrieved context."

    return "\n".join(
        f"Division {division.code} - {division.label}"
        for division in selected_divisions
    )


def build_scope_queries(
    selected_divisions: List[SelectedDivision],
    instructions: str,
) -> List[str]:
    division_terms = " ".join(
        f"Div {division.code} {division.label}"
        for division in selected_divisions
    )

    base_terms = (
        "scope of work quantities specification sections drawing references "
        "schedules door schedule window schedule finish schedule equipment schedule "
        "materials products execution installation submittals quality requirements "
        "allowances alternates addendum addenda revisions changed scope"
    )

    queries = [
        " ".join(["Detailed scope extraction", division_terms, instructions, base_terms]).strip(),
        " ".join(["specification section products execution quantity schedule", instructions]).strip(),
        " ".join(["drawing schedule references quantities materials scope items", instructions]).strip(),
        " ".join(["addendum addenda revisions changed scope quantities", instructions]).strip(),
    ]

    for division in selected_divisions:
        queries.append(
            " ".join(
                [
                    f"Division {division.code}",
                    division.label,
                    "scope requirements specifications products execution quantity references",
                    instructions,
                ]
            ).strip()
        )

    return dedupe_preserve_order(queries)


def build_scope_context(
    rag_service: Any,
    user_id: str,
    project_id: str,
    selected_divisions: List[SelectedDivision],
    instructions: str,
) -> tuple[str, str, str]:
    blocks = []
    seen_text = set()
    project_name = ""
    project_address = ""

    for query in build_scope_queries(selected_divisions, instructions):
        context, _sources = rag_service.retrieve_context(
            query=query,
            user_id=user_id,
            project_id=project_id,
            top_k=10,
        )
        add_context_blocks(blocks, seen_text, context)

    keyword_candidates = []

    try:
        keyword_candidates = rag_service.qdrant_service.scroll_project_payloads(
            user_id=user_id,
            project_id=project_id,
            limit=900,
        )
    except Exception:
        keyword_candidates = []

    for candidate in keyword_candidates:
        payload = candidate.get("payload", {})

        if not project_name:
            project_name = stringify(payload.get("project_name"))

        if not project_address:
            project_address = stringify(payload.get("project_address"))

        if project_name and project_address:
            break

    keyword_terms = build_scope_keyword_terms(selected_divisions, instructions)

    for candidate in top_keyword_candidates(keyword_candidates, keyword_terms, limit=24):
        payload = candidate.get("payload", {})
        text = stringify(payload.get("text"))

        if not text:
            continue

        source = format_payload_source(payload)
        block = f"{source}\n{text}"

        add_block(blocks, seen_text, block)

    relabeled_blocks = []

    for index, block in enumerate(blocks, start=1):
        block = re.sub(r"^\[S\d+\]\s*", "", block.strip())
        relabeled_blocks.append(f"[SC{index}] {block}")

    return (
        limit_context("\n\n".join(relabeled_blocks), 28000),
        project_name,
        project_address,
    )


def build_scope_keyword_terms(
    selected_divisions: List[SelectedDivision],
    instructions: str,
) -> List[str]:
    terms = [
        "scope",
        "scope of work",
        "quantity",
        "quantities",
        "schedule",
        "schedules",
        "door schedule",
        "window schedule",
        "finish schedule",
        "room finish",
        "equipment schedule",
        "specification",
        "specifications",
        "section",
        "products",
        "execution",
        "installation",
        "materials",
        "submittal",
        "submittals",
        "allowance",
        "allowances",
        "alternate",
        "alternates",
        "drawing",
        "drawings",
        "reference",
        "references",
        "addendum",
        "addenda",
        "revision",
        "revisions",
        "changed scope",
    ]

    for division in selected_divisions:
        terms.extend(
            [
                division.code,
                f"div {division.code}",
                f"division {division.code}",
                division.label,
            ]
        )

    terms.extend(
        token.lower()
        for token in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9./-]{2,}", instructions)
    )

    return dedupe_preserve_order([term.lower() for term in terms if stringify(term)])


def normalize_scope_extraction_response(payload: Dict[str, Any]) -> Dict[str, Any]:
    items = []

    for index, raw_item in enumerate(ensure_list(payload.get("items")), start=1):
        if not isinstance(raw_item, dict):
            continue

        quantity = raw_item.get("quantity")

        if not isinstance(quantity, dict):
            quantity = {}

        division_code = normalize_division_code(raw_item.get("division_code")) or "00"

        items.append(
            {
                "id": index,
                "title": stringify(raw_item.get("title")) or "Untitled Scope Item",
                "division_code": division_code,
                "division_label": stringify(raw_item.get("division_label"))
                or CSI_DIVISIONS.get(division_code, f"Division {division_code}"),
                "quantity": {
                    "value": parse_float(quantity.get("value")),
                    "unit": stringify(quantity.get("unit")) or "unspecified",
                },
                "specifications": stringify(raw_item.get("specifications"))
                or "No specifications found.",
                "references": [
                    normalize_scope_reference(reference)
                    for reference in ensure_list(raw_item.get("references"))
                    if isinstance(reference, dict)
                ],
            }
        )

    return {"items": items}


def normalize_scope_reference(value: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "code": stringify(value.get("code")) or "Not found",
        "title": stringify(value.get("title")) or "Untitled Reference",
        "page": parse_optional_int(value.get("page")),
        "division": normalize_division_code(value.get("division"))
        if value.get("division")
        else None,
    }


def top_keyword_candidates(
    candidates: List[Dict[str, Any]],
    terms: List[str],
    limit: int,
) -> List[Dict[str, Any]]:
    scored = []

    for candidate in candidates:
        payload = candidate.get("payload", {})

        haystack = " ".join(
            [
                stringify(payload.get("text")),
                stringify(payload.get("filename")),
                stringify(payload.get("source_text_ref")),
                stringify(payload.get("source_type")),
            ]
        ).lower()

        if not haystack:
            continue

        score = 0

        if stringify(payload.get("source_type")).lower() == "addendum":
            score += 8

        for term in terms:
            if term and term in haystack:
                score += 4 if " " in term or term.isdigit() else 1

        if score > 0:
            scored.append((score, candidate))

    scored.sort(key=lambda item: item[0], reverse=True)

    return [candidate for _, candidate in scored[:limit]]


def add_context_blocks(blocks: List[str], seen_text: set[str], context: str) -> None:
    for block in split_context_blocks(context):
        add_block(blocks, seen_text, block)


def split_context_blocks(context: str) -> List[str]:
    pieces = re.split(r"\n(?=\[S\d+\]\s)", context.strip())
    return [piece.strip() for piece in pieces if piece.strip()]


def add_block(blocks: List[str], seen_text: set[str], block: str) -> bool:
    text = block.split("\n", 1)[1] if "\n" in block else block
    key = normalize_dedupe_key(text)

    if not key or key in seen_text:
        return False

    seen_text.add(key)
    blocks.append(block)

    return True


def normalize_dedupe_key(text: str) -> str:
    return re.sub(r"\W+", "", text.lower())[:400]


def format_payload_source(payload: Dict[str, Any]) -> str:
    parts = [stringify(payload.get("filename")) or "unknown"]

    if payload.get("source_type"):
        parts.append(stringify(payload.get("source_type")))

    if payload.get("project_name"):
        parts.append(f"project {stringify(payload.get('project_name'))}")

    if payload.get("project_address"):
        parts.append(f"address {stringify(payload.get('project_address'))}")

    if payload.get("chunk_index") is not None:
        parts.append(f"chunk {payload['chunk_index']}")

    if payload.get("chunk_type"):
        parts.append(stringify(payload.get("chunk_type")))

    return " | ".join(parts)


def limit_context(context: str, max_chars: int) -> str:
    if len(context) <= max_chars:
        return context

    return context[:max_chars].rsplit("\n\n", 1)[0] + "\n\n[context truncated]"


def dedupe_preserve_order(values: List[str]) -> List[str]:
    deduped = []
    seen = set()

    for value in values:
        text = stringify(value)
        key = text.lower()

        if text and key not in seen:
            deduped.append(text)
            seen.add(key)

    return deduped


def ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []

    if isinstance(value, list):
        return value

    return [value]


def stringify(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        return re.sub(r"\s+", " ", value).strip()

    if isinstance(value, (dict, list)):
        return re.sub(r"\s+", " ", json.dumps(value, ensure_ascii=False)).strip()

    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_division_code(value: Any) -> str:
    text = str(value or "").strip()

    if not text:
        return ""

    match = re.search(r"\d{1,2}", text)

    if not match:
        return ""

    return match.group(0).zfill(2)


def parse_float(value: Any) -> float:
    match = re.search(r"\d+(\.\d+)?", str(value or ""))

    if not match:
        return 0.0

    return float(match.group(0))


def parse_optional_int(value: Any) -> Optional[int]:
    match = re.search(r"\d+", str(value or ""))

    if not match:
        return None

    return int(match.group(0))


_scope_service = None


def get_scope_service() -> ScopeService:
    global _scope_service

    if _scope_service is None:
        _scope_service = ScopeService()

    return _scope_service
