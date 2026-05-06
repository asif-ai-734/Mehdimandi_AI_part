"""
Quote draft generation service.
"""

import json
import re
from typing import Any, Dict, List

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


QUOTE_DRAFT_SYSTEM_PROMPT = """You are an expert construction estimator.

Generate preliminary quote draft information from the provided tender context.

The output is for review only. Use supported tender information, scope items,
pricing impacts, allowances, alternates, bonds, schedule impacts, and selected
CSI divisions.

Rules:
- Use only the provided context.
- Do not invent unsupported costs.
- If exact values are not found, provide a reasonable estimator-style placeholder only when the context supports that cost category.
- Recommended lump sum should include HST in the displayed string if project context implies tax should be included.
- Amounts must be display strings, such as "$485,000 + HST" or "$72,750".
- Division breakdown should include selected divisions only when supported.
- Additional costs should include pricing impacts such as night work, bonds, escalation, special logistics, or schedule constraints.
- Return JSON only. Do not include markdown.

Required schema:
{
  "recommended_lump_sum": "$485,000 + HST",
  "lump_sum_note": "Based on scope analysis and pricing factors",
  "division_breakdown": [
    {
      "division_code": "01",
      "division_label": "General Requirements",
      "amount": "$72,750"
    }
  ],
  "additional_costs": [
    {
      "title": "Night Work Premium (18%)",
      "amount": "$32,400"
    }
  ],
  "build_quote_label": "Open Build Quote"
}
"""


class QuoteDraftService:
    def __init__(self):
        self.rag_service = get_rag_service()
        self.openai_service = get_openai_service()

    def generate_quote_draft(
        self,
        user_id: str,
        project_id: str,
        divisions: List[str],
        instructions: str,
    ) -> Dict[str, Any]:
        selected_divisions = build_selected_divisions(divisions)

        context = build_quote_draft_context(
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
                        f"Selected CSI divisions:\n{format_selected_divisions(selected_divisions)}\n\n"
                        f"Estimator instructions:\n"
                        f"{instructions.strip() or 'No instructions provided.'}\n\n"
                        f"Retrieved tender context:\n{context}"
                    ),
                }
            ],
            system_prompt=QUOTE_DRAFT_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=2600,
        )

        return normalize_quote_draft_response(
            payload=payload,
            selected_divisions=selected_divisions,
        )


def build_quote_draft_context(
    rag_service: Any,
    user_id: str,
    project_id: str,
    divisions: List[str],
    instructions: str,
) -> str:
    division_text = " ".join(str(division) for division in divisions)

    queries = [
        f"estimated value lump sum total bid amount quote pricing breakdown {division_text}",
        f"division breakdown costs CSI divisions pricing estimate {division_text}",
        f"pricing impacts allowances alternates bonds night work premium escalation {division_text}",
        f"scope quantities labor hours duration schedule cost factors {division_text}",
        f"addendum addenda revisions changed scope pricing impact {division_text}",
        instructions,
    ]

    contexts = []
    seen = set()

    for query in queries:
        query = stringify(query)

        if not query:
            continue

        context, _sources = rag_service.retrieve_context(
            query=query,
            user_id=user_id,
            project_id=project_id,
            top_k=10,
        )

        for block in split_context_blocks(context):
            key = normalize_dedupe_key(block)

            if key and key not in seen:
                seen.add(key)
                contexts.append(block)

    return "\n\n".join(contexts)[:26000]


def normalize_quote_draft_response(
    payload: Dict[str, Any],
    selected_divisions: List[Dict[str, str]],
) -> Dict[str, Any]:
    division_items = []

    for raw_item in ensure_list(payload.get("division_breakdown")):
        if not isinstance(raw_item, dict):
            continue

        code = normalize_division_code(raw_item.get("division_code"))

        if not code:
            continue

        division_items.append(
            {
                "division_code": code,
                "division_label": stringify(raw_item.get("division_label"))
                or CSI_DIVISIONS.get(code, f"Division {code}"),
                "amount": stringify(raw_item.get("amount")) or "Not found",
            }
        )

    if not division_items:
        for division in selected_divisions:
            division_items.append(
                {
                    "division_code": division["code"],
                    "division_label": division["label"],
                    "amount": "Not found",
                }
            )

    additional_costs = []

    for raw_item in ensure_list(payload.get("additional_costs")):
        if not isinstance(raw_item, dict):
            continue

        title = stringify(raw_item.get("title"))
        amount = stringify(raw_item.get("amount"))

        if not title:
            continue

        additional_costs.append(
            {
                "title": title,
                "amount": amount or "Not found",
            }
        )

    return {
        "recommended_lump_sum": stringify(payload.get("recommended_lump_sum"))
        or "Not found",
        "lump_sum_note": stringify(payload.get("lump_sum_note"))
        or "Based on scope analysis and pricing factors",
        "division_breakdown": division_items,
        "additional_costs": additional_costs,
        "build_quote_label": stringify(payload.get("build_quote_label"))
        or "Open Build Quote",
    }


def build_selected_divisions(divisions: List[str]) -> List[Dict[str, str]]:
    selected = []
    seen = set()

    for division in divisions:
        code = normalize_division_code(division)

        if not code or code in seen:
            continue

        selected.append(
            {
                "code": code,
                "label": CSI_DIVISIONS.get(code, f"Division {code}"),
            }
        )
        seen.add(code)

    return selected


def format_selected_divisions(selected_divisions: List[Dict[str, str]]) -> str:
    if not selected_divisions:
        return "All relevant divisions found in the retrieved context."

    return "\n".join(
        f"Division {division['code']} - {division['label']}"
        for division in selected_divisions
    )


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


def normalize_division_code(value: Any) -> str:
    text = str(value or "").strip()

    if not text:
        return ""

    match = re.search(r"\d{1,2}", text)

    if not match:
        return ""

    return match.group(0).zfill(2)


_quote_draft_service = None


def get_quote_draft_service() -> QuoteDraftService:
    global _quote_draft_service

    if _quote_draft_service is None:
        _quote_draft_service = QuoteDraftService()

    return _quote_draft_service
