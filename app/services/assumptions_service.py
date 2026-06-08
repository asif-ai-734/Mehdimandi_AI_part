"""
Assumptions extraction service.
"""

import json
import re
from typing import Any, Dict, List, Optional

from app.services.openai_service import get_openai_service
from app.services.rag_service import get_rag_service


ASSUMPTIONS_SYSTEM_PROMPT = """You are an expert construction estimator.

Extract reasonable bid assumptions supported by the provided tender context.

Assumptions are statements an estimator can rely on when pricing, based on
requirements, notes, site conditions, responsibilities, schedules, references,
or incomplete information in the tender documents.

Rules:
- Use only the provided tender context.
- Do not invent assumptions.
- Do not create assumptions that contradict the documents.
- Phrase each item as a clear assumption statement.
- Keep each assumption concise.
- Focus on scope, site access, temporary facilities, utilities, substrates,
  coordination, preparation, working hours, storage, schedule, and responsibility.
- Include the best available source reference.
- If page is unknown, use null.
- Return JSON only. Do not include markdown.

Required schema:
{
  "items": [
    {
      "text": "Site access will be provided as per tender specifications during 9–11 AM delivery windows.",
      "reference": {
        "file": "General Conditions.pdf",
        "page": 11,
        "section": "5.1 Site Access"
      }
    }
  ]
}
"""


class AssumptionsService:
    def __init__(self):
        self.rag_service = get_rag_service()
        self.openai_service = get_openai_service()

    def extract_assumptions(
        self,
        user_id: str,
        project_id: str,
        divisions: List[str],
        instructions: str,
    ) -> Dict[str, Any]:
        context = build_assumptions_context(
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
            system_prompt=ASSUMPTIONS_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=2600,
        )

        return normalize_assumptions_response(payload)


def build_assumptions_context(
    rag_service: Any,
    user_id: str,
    project_id: str,
    divisions: List[str],
    instructions: str,
) -> str:
    division_text = " ".join(str(division) for division in divisions)

    queries = [
        f"bid assumptions tender requirements site access temporary facilities utilities {division_text}",
        f"contractor responsibilities owner provided general contractor provided by others {division_text}",
        f"substrate preparation existing conditions ready to receive finishes coordination {division_text}",
        f"working hours delivery windows storage staging logistics site conditions {division_text}",
        f"measurement payment scope responsibility exclusions inclusions {division_text}",
        f"addendum addenda clarifications assumptions changed requirements {division_text}",
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


def normalize_assumptions_response(payload: Dict[str, Any]) -> Dict[str, Any]:
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


_assumptions_service = None


def get_assumptions_service() -> AssumptionsService:
    global _assumptions_service

    if _assumptions_service is None:
        _assumptions_service = AssumptionsService()

    return _assumptions_service
