"""
Clarifications extraction service.
"""

import json
import re
from typing import Any, Dict, List, Optional

from app.services.openai_service import get_openai_service
from app.services.rag_service import get_rag_service


CLARIFICATIONS_SYSTEM_PROMPT = """You are an expert construction estimator.

Extract clarification questions that should be sent to the owner, architect,
consultant, or general contractor before final pricing.

Clarifications should focus on ambiguous, missing, conflicting, incomplete,
or risky tender requirements.

Rules:
- Use only the provided tender context.
- Do not invent issues.
- Phrase each item as a clear question or verification request.
- Keep each question concise.
- Focus on pricing, scope, schedule, responsibility, site access, allowances,
  substitutions, working hours, temporary facilities, coordination, and addenda.
- Include the best available source reference.
- If page is unknown, use null.
- Return JSON only. Do not include markdown.

Required schema:
{
  "items": [
    {
      "question": "Confirm exact working hours permitted for night work and any noise level restrictions.",
      "reference": {
        "file": "General Conditions.pdf",
        "page": 8,
        "section": "4.2 Working Hours"
      }
    }
  ]
}
"""


class ClarificationsService:
    def __init__(self):
        self.rag_service = get_rag_service()
        self.openai_service = get_openai_service()

    def extract_clarifications(
        self,
        user_id: str,
        project_id: str,
        divisions: List[str],
        instructions: str,
    ) -> Dict[str, Any]:
        context = build_clarifications_context(
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
            system_prompt=CLARIFICATIONS_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=2600,
        )

        return normalize_clarifications_response(payload)


def build_clarifications_context(
    rag_service: Any,
    user_id: str,
    project_id: str,
    divisions: List[str],
    instructions: str,
) -> str:
    division_text = " ".join(str(division) for division in divisions)

    queries = [
        f"clarification questions ambiguities missing information conflicts {division_text}",
        f"working hours site access temporary facilities contractor responsibility {division_text}",
        f"allowances alternates substitutions measurement payment scope gaps {division_text}",
        f"coordination with other trades approvals submittals review period {division_text}",
        f"addendum addenda revisions clarifications changed requirements {division_text}",
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


def normalize_clarifications_response(payload: Dict[str, Any]) -> Dict[str, Any]:
    items = []

    for index, raw_item in enumerate(ensure_list(payload.get("items")), start=1):
        if not isinstance(raw_item, dict):
            continue

        reference = raw_item.get("reference")

        if not isinstance(reference, dict):
            reference = {}

        question = stringify(raw_item.get("question"))

        if not question:
            continue

        items.append(
            {
                "id": index,
                "question": question,
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


_clarifications_service = None


def get_clarifications_service() -> ClarificationsService:
    global _clarifications_service

    if _clarifications_service is None:
        _clarifications_service = ClarificationsService()

    return _clarifications_service
