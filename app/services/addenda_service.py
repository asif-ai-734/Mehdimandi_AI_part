"""
Addenda extraction service.
"""

import json
import re
from typing import Any, Dict, List, Optional

from app.services.openai_service import get_openai_service
from app.services.rag_service import get_rag_service


ADDENDA_SYSTEM_PROMPT = """You are an expert construction estimator.

Extract structured addenda changes from the provided tender context.

Focus ONLY on:
- addendum changes
- revised specifications
- revised schedules
- revised scope
- revised materials
- revised bid dates
- revised drawings
- revised quantities
- revised contractor responsibilities
- pricing impacts
- scope impacts

Rules:
- Use only supported information from context.
- Do not invent addenda.
- Do not invent dates.
- Do not invent pricing impacts.
- Keep titles concise.
- Summarize changes clearly.
- Identify affected CSI divisions where possible.
- Use short impact type labels:
  - Cost/Specs
  - Schedule
  - Scope
  - Coordination
  - Materials
  - Contract
- Return JSON only.

Required schema:
{
  "items": [
    {
      "addendum_number": "Addendum 01",
      "issued_date": "April 10, 2026",
      "title": "Window Specification Changed",
      "description": "Changed from single-pane to double-glazed low-E glass.",
      "impact_type": "Cost/Specs",
      "affected_divisions": ["08"],
      "scope_change": "Add",
      "pricing_impact": "+$12,500",
      "reference": {
        "file": "Addendum_01.pdf",
        "page": 2,
        "item": "Item 3"
      }
    }
  ]
}
"""


class AddendaService:
    def __init__(self):
        self.rag_service = get_rag_service()
        self.openai_service = get_openai_service()

    def extract_addenda(
        self,
        user_id: str,
        project_id: str,
        divisions: List[str],
        instructions: str,
    ) -> Dict[str, Any]:

        context = build_addenda_context(
            rag_service=self.rag_service,
            user_id=user_id,
            project_id=project_id,
            divisions=divisions,
            instructions=instructions,
        )

        if not context.strip():
            raise ValueError(
                "No relevant uploaded document context found for this project"
            )

        payload = self.openai_service.generate_json(
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Selected CSI divisions:\n{format_divisions(divisions)}\n\n"
                        f"Estimator instructions:\n"
                        f"{instructions.strip() or 'No instructions provided.'}\n\n"
                        f"Retrieved tender context:\n{context}"
                    ),
                }
            ],
            system_prompt=ADDENDA_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=3200,
        )

        return normalize_addenda_response(payload)


def build_addenda_context(
    rag_service: Any,
    user_id: str,
    project_id: str,
    divisions: List[str],
    instructions: str,
) -> str:

    division_text = " ".join(str(division) for division in divisions)

    queries = [
        f"addendum addenda revisions changed scope revised specifications {division_text}",
        f"revised drawings revised materials revised quantities revised schedule {division_text}",
        f"bid closing extension revised dates revised responsibilities {division_text}",
        f"pricing impacts revised contractor scope add delete changes {division_text}",
        f"bulletins clarifications revisions contract changes {division_text}",
        instructions,
    ]

    contexts = []
    seen = set()

    for context, _sources in rag_service.retrieve_contexts(
        queries=[stringify(query) for query in queries],
        user_id=user_id,
        project_id=project_id,
        top_k=10,
    ):
        for block in split_context_blocks(context):
            key = normalize_dedupe_key(block)

            if key and key not in seen:
                seen.add(key)
                contexts.append(block)

    return "\n\n".join(contexts)[:26000]


def normalize_addenda_response(
    payload: Dict[str, Any],
) -> Dict[str, Any]:

    items = []

    for index, raw_item in enumerate(
        ensure_list(payload.get("items")),
        start=1,
    ):

        if not isinstance(raw_item, dict):
            continue

        reference = raw_item.get("reference")

        if not isinstance(reference, dict):
            reference = {}

        divisions = ensure_list(raw_item.get("affected_divisions"))

        items.append(
            {
                "id": index,

                "addendum_number": stringify(
                    raw_item.get("addendum_number")
                ) or f"Addendum {index:02d}",

                "issued_date": stringify(
                    raw_item.get("issued_date")
                ) or None,

                "title": stringify(
                    raw_item.get("title")
                ) or "Untitled Addenda Change",

                "description": stringify(
                    raw_item.get("description")
                ) or "No description found.",

                "impact_type": stringify(
                    raw_item.get("impact_type")
                ) or "General",

                "affected_divisions": [
                    stringify(division)
                    for division in divisions
                    if stringify(division)
                ],

                "scope_change": stringify(
                    raw_item.get("scope_change")
                ) or "None",

                "pricing_impact": stringify(
                    raw_item.get("pricing_impact")
                ) or "None",

                "reference": {
                    "file": stringify(
                        reference.get("file")
                    ) or "Not found",

                    "page": parse_optional_int(
                        reference.get("page")
                    ),

                    "item": stringify(
                        reference.get("item")
                    ) or None,
                },
            }
        )

    return {
        "items": items
    }


def format_divisions(divisions: List[str]) -> str:
    text = ", ".join(
        stringify(division)
        for division in divisions
        if stringify(division)
    )
    return text or "All relevant divisions found in the retrieved context."


def split_context_blocks(context: str) -> List[str]:
    pieces = re.split(r"\n(?=\[S\d+\]\s)", stringify(context))
    return [piece.strip() for piece in pieces if piece.strip()]


def normalize_dedupe_key(text: str) -> str:
    return re.sub(r"\W+", "", stringify(text).lower())[:400]


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
        return re.sub(
            r"\s+",
            " ",
            json.dumps(value, ensure_ascii=False),
        ).strip()

    return re.sub(r"\s+", " ", str(value)).strip()


def parse_optional_int(value: Any) -> Optional[int]:
    match = re.search(r"\d+", str(value or ""))

    if not match:
        return None

    return int(match.group(0))


_addenda_service = None


def get_addenda_service() -> AddendaService:
    global _addenda_service

    if _addenda_service is None:
        _addenda_service = AddendaService()

    return _addenda_service
