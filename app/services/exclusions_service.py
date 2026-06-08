"""
Exclusions extraction service.
"""

import json
import re
from typing import Any, Dict, List, Optional

from app.services.openai_service import get_openai_service
from app.services.rag_service import get_rag_service


EXCLUSIONS_SYSTEM_PROMPT = """You are an expert construction estimator.

Extract items that are explicitly excluded from the contractor's scope,
not included in selected divisions, excluded by tender notes, excluded by
scope descriptions, or assigned to others.

Rules:
- Use only the provided tender context.
- Do not invent exclusions.
- Only include items that are clearly excluded, not included, by others, owner responsibility, or outside selected scope.
- Phrase each item as a concise exclusion statement.
- Include the best available source reference.
- If page is unknown, use null.
- Return JSON only. Do not include markdown.

Required schema:
{
  "items": [
    {
      "text": "Electrical rough-in and fixture installation",
      "reference": {
        "file": "Scope of Work.pdf",
        "page": 4,
        "section": "Division 26 (Not Included)"
      }
    }
  ]
}
"""


class ExclusionsService:
    def __init__(self):
        self.rag_service = get_rag_service()
        self.openai_service = get_openai_service()

    def extract_exclusions(
        self,
        user_id: str,
        project_id: str,
        divisions: List[str],
        instructions: str,
    ) -> Dict[str, Any]:
        context = build_exclusions_context(
            rag_service=self.rag_service,
            user_id=user_id,
            project_id=project_id,
            divisions=divisions,
            instructions=instructions,
        )

        if not context.strip():
            raise ValueError("No relevant uploaded document context found for this project")

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
            system_prompt=EXCLUSIONS_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=2600,
        )

        return normalize_exclusions_response(payload)


def build_exclusions_context(
    rag_service: Any,
    user_id: str,
    project_id: str,
    divisions: List[str],
    instructions: str,
) -> str:
    division_text = " ".join(str(division) for division in divisions)

    queries = [
        f"exclusions excluded from scope not included by others owner responsibility {division_text}",
        f"scope of work exclusions not in contract outside scope contractor not responsible {division_text}",
        f"division not included excluded trades electrical plumbing hvac permits fees {division_text}",
        f"temporary fencing security hoarding permits inspection fees excluded {division_text}",
        f"measurement payment exclusions inclusions responsibilities scope limits {division_text}",
        f"addendum addenda clarifications exclusions changed scope not included {division_text}",
        instructions,
    ]

    contexts = []
    seen = set()

    for context, _sources in rag_service.retrieve_contexts(
        queries=[stringify(query) for query in queries],
        user_id=user_id,
        project_id=project_id,
        top_k=8,
    ):
        for block in split_context_blocks(context):
            key = normalize_dedupe_key(block)

            if key and key not in seen:
                seen.add(key)
                contexts.append(block)

    return "\n\n".join(contexts)[:24000]


def normalize_exclusions_response(payload: Dict[str, Any]) -> Dict[str, Any]:
    items = []

    for index, raw_item in enumerate(ensure_list(payload.get("items")), start=1):
        if not isinstance(raw_item, dict):
            continue

        reference = raw_item.get("reference")

        if not isinstance(reference, dict):
            reference = {}

        text = stringify(raw_item.get("text"))

        if not text:
            continue

        items.append(
            {
                "id": index,
                "text": text,
                "reference": {
                    "file": stringify(reference.get("file")) or "Not found",
                    "page": parse_optional_int(reference.get("page")),
                    "section": stringify(reference.get("section")) or None,
                },
            }
        )

    return {"items": items}


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


_exclusions_service = None


def get_exclusions_service() -> ExclusionsService:
    global _exclusions_service

    if _exclusions_service is None:
        _exclusions_service = ExclusionsService()

    return _exclusions_service
