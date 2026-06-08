"""
Risks extraction service.
"""

import json
import re
from typing import Any, Dict, List, Optional

from app.services.openai_service import get_openai_service
from app.services.rag_service import get_rag_service


RISKS_SYSTEM_PROMPT = """You are an expert construction estimator and risk analyst.

Extract ONLY meaningful risks, coordination concerns, contractual risks,
schedule risks, pricing risks, logistics constraints, approval bottlenecks,
site limitations, warranty concerns, and execution challenges from the
provided tender context.

Rules:
- Use only supported information from context.
- Do not invent risks.
- Keep titles concise.
- Keep descriptions practical and estimator-focused.
- Categorize each item.
- Categories should be short:
  - Pricing Impact
  - Schedule Impact
  - Coordination Item
  - Contractual Requirement
  - Site Constraint
  - Material Risk
  - Approval Risk
  - Execution Risk
- Include best source reference.
- Return JSON only.

Required schema:
{
  "items": [
    {
      "title": "Night Work Required",
      "description": "Tender specifies work between 10 PM–6 AM for noise-sensitive areas.",
      "category": "Pricing Impact",
      "reference": {
        "file": "General Conditions.pdf",
        "page": 8,
        "section": "4.2 Working Hours"
      }
    }
  ]
}
"""


class RisksService:

    def __init__(self):
        self.rag_service = get_rag_service()
        self.openai_service = get_openai_service()

    def extract_risks(
        self,
        user_id: str,
        project_id: str,
        divisions: List[str],
        instructions: str,
    ) -> Dict[str, Any]:

        context = build_risk_context(
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
            system_prompt=RISKS_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=2500,
        )

        return normalize_risks_response(payload)


def build_risk_context(
    rag_service: Any,
    user_id: str,
    project_id: str,
    divisions: List[str],
    instructions: str,
) -> str:

    queries = [
        "construction risks coordination issues contractual requirements schedule impacts",
        "liquidated damages warranties bonds insurance penalties approvals",
        "site access logistics phasing restricted hours delays",
        "material lead times approvals substitutions procurement",
        "addenda revisions changed scope coordination issues",
        instructions,
    ]

    contexts = []

    for context, _sources in rag_service.retrieve_contexts(
        queries=queries,
        user_id=user_id,
        project_id=project_id,
        top_k=8,
    ):
        if context:
            contexts.append(context)

    return "\n\n".join(contexts)[:24000]


def normalize_risks_response(
    payload: Dict[str, Any],
) -> Dict[str, Any]:

    items = []

    for index, raw_item in enumerate(payload.get("items", []), start=1):

        if not isinstance(raw_item, dict):
            continue

        reference = raw_item.get("reference")

        if not isinstance(reference, dict):
            reference = {}

        items.append(
            {
                "id": index,
                "title": stringify(raw_item.get("title"))
                or "Untitled Risk",

                "description": stringify(raw_item.get("description"))
                or "No description found.",

                "category": stringify(raw_item.get("category"))
                or "General Risk",

                "reference": {
                    "file": stringify(reference.get("file"))
                    or "Not found",

                    "page": parse_optional_int(reference.get("page")),

                    "section": stringify(reference.get("section"))
                    or None,
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


_risks_service = None


def get_risks_service() -> RisksService:
    global _risks_service

    if _risks_service is None:
        _risks_service = RisksService()

    return _risks_service
