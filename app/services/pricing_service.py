"""
Pricing impacts extraction service.

Uses the existing project RAG index as context and asks OpenAI for structured
pricing impact items for the pricing screen.
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


PRICING_IMPACTS_SYSTEM_PROMPT = """You are an expert construction estimator.

Extract pricing impacts ONLY from the provided tender context.

Pricing impacts are clauses, requirements, risks, conditions, schedules, addenda,
allowances, alternates, bonds, warranties, logistics, coordination constraints,
or scope requirements that may affect bid price.

Rules:
- Use only supported information from retrieved context.
- Focus on selected CSI divisions and estimator instructions.
- Do not invent project facts.
- Do not invent dollar values.
- If a cost amount is explicitly stated, include it.
- If no dollar value is stated but the item clearly affects price, use "Not found".
- Keep each title short.
- Keep descriptions concise.
- Include practical estimator impact wording.
- Include the best available file/page/section reference.
- If page is unknown, use null.
- Return JSON only. Do not include markdown.

Required schema:
{
  "items": [
    {
      "title": "Night Work Required",
      "description": "Work between 10 PM–6 AM for noise-sensitive areas.",
      "impact": "Cost increase for overtime labor and supervision.",
      "amount": "$32,400",
      "reference": {
        "file": "General Conditions.pdf",
        "page": 8,
        "section": "4.2 Working Hours"
      }
    }
  ]
}
"""


class SelectedDivision:
    def __init__(self, code: str, label: str):
        self.code = code
        self.label = label


class PricingService:
    """Coordinates RAG retrieval and OpenAI pricing impact extraction."""

    def __init__(self):
        self.rag_service = get_rag_service()
        self.openai_service = get_openai_service()

    def extract_pricing_impacts(
        self,
        user_id: str,
        project_id: str,
        divisions: List[str],
        instructions: str,
    ) -> Dict[str, Any]:
        selected_divisions = build_selected_divisions(divisions)

        context, project_name, project_address = build_pricing_context(
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
                        f"Estimator instructions:\n"
                        f"{instructions.strip() or 'No additional instructions provided.'}\n\n"
                        f"Retrieved tender document context:\n{context}"
                    ),
                }
            ],
            system_prompt=PRICING_IMPACTS_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=3000,
        )

        return normalize_pricing_impacts_response(payload)


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


def build_pricing_queries(
    selected_divisions: List[SelectedDivision],
    instructions: str,
) -> List[str]:
    division_terms = " ".join(
        f"Div {division.code} {division.label}"
        for division in selected_divisions
    )

    base_terms = (
        "pricing impacts cost factors bid price cost increase allowance allowances "
        "alternate alternates unit prices bonds insurance warranty warranties "
        "liquidated damages schedule constraints overtime night work phasing "
        "site access logistics coordination delays temporary facilities "
        "submittals closeout testing inspection addendum addenda revisions "
        "changed scope escalation material price labor impact"
    )

    queries = [
        " ".join(["Pricing impacts cost factors", division_terms, instructions, base_terms]).strip(),
        " ".join(["allowances alternates unit prices bond insurance warranty liquidated damages", instructions]).strip(),
        " ".join(["schedule constraints overtime night work phasing site access logistics delays", instructions]).strip(),
        " ".join(["addendum addenda revisions changed scope pricing impact cost impact", instructions]).strip(),
        " ".join(["contract conditions bid requirements pricing risks estimator impacts", instructions]).strip(),
    ]

    for division in selected_divisions:
        queries.append(
            " ".join(
                [
                    f"Division {division.code}",
                    division.label,
                    "pricing cost impact allowance alternate warranty coordination schedule",
                    instructions,
                ]
            ).strip()
        )

        if division.code == "01":
            queries.append(
                "General Requirements pricing impacts temporary facilities bonds "
                "insurance allowances alternates liquidated damages schedule "
                "site access coordination closeout warranty"
            )

    return dedupe_preserve_order(queries)


def build_pricing_context(
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

    for query in build_pricing_queries(selected_divisions, instructions):
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

    keyword_terms = build_pricing_keyword_terms(selected_divisions, instructions)

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
        relabeled_blocks.append(f"[P{index}] {block}")

    return (
        limit_context("\n\n".join(relabeled_blocks), 28000),
        project_name,
        project_address,
    )


def build_pricing_keyword_terms(
    selected_divisions: List[SelectedDivision],
    instructions: str,
) -> List[str]:
    terms = [
        "pricing",
        "price",
        "cost",
        "costs",
        "cost impact",
        "pricing impact",
        "allowance",
        "allowances",
        "cash allowance",
        "alternate",
        "alternates",
        "unit price",
        "unit prices",
        "bond",
        "bonds",
        "performance bond",
        "insurance",
        "warranty",
        "warranties",
        "liquidated damages",
        "damages",
        "schedule",
        "completion",
        "delay",
        "delays",
        "coordination",
        "site access",
        "restricted access",
        "night work",
        "overtime",
        "phasing",
        "temporary facilities",
        "testing",
        "inspection",
        "submittals",
        "closeout",
        "escalation",
        "material price",
        "labor",
        "addendum",
        "addenda",
        "revision",
        "revisions",
        "changed scope",
        "clarification",
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


def normalize_pricing_impacts_response(payload: Dict[str, Any]) -> Dict[str, Any]:
    items = []

    for index, raw_item in enumerate(ensure_list(payload.get("items")), start=1):
        if not isinstance(raw_item, dict):
            continue

        reference = raw_item.get("reference")

        if not isinstance(reference, dict):
            reference = {}

        title = stringify(raw_item.get("title"))
        description = stringify(raw_item.get("description"))
        impact = stringify(raw_item.get("impact"))
        amount = stringify(raw_item.get("amount"))

        if not title and not description and not impact:
            continue

        items.append(
            {
                "id": index,
                "title": title or "Untitled Pricing Impact",
                "description": description or "No description found.",
                "impact": impact or "Pricing impact not specified.",
                "amount": amount or "Not found",
                "reference": normalize_pricing_reference(reference),
            }
        )

    return {"items": items}


def normalize_pricing_reference(value: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "file": stringify(value.get("file")) or "Not found",
        "page": parse_optional_int(value.get("page")),
        "section": stringify(value.get("section")) or None,
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


def parse_optional_int(value: Any) -> Optional[int]:
    match = re.search(r"\d+", str(value or ""))

    if not match:
        return None

    return int(match.group(0))


_pricing_service = None


def get_pricing_service() -> PricingService:
    global _pricing_service

    if _pricing_service is None:
        _pricing_service = PricingService()

    return _pricing_service
