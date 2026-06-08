"""
Build UI-ready proposed-change summaries for section re-analysis.
"""

import json
from typing import Any, Dict, List, Optional

from app.schemas.section_reanalysis import ProposedChanges
from app.services.openai_service import get_openai_service


PROPOSED_CHANGES_SYSTEM_PROMPT = """You are an estimating workflow assistant.
Compare the previous section JSON with the updated section JSON after a user's AI instruction.
Return only a JSON object with these keys:
{
  "changes_label": "Scope Changes",
  "changes": ["short action-oriented change"],
  "pricing_impact": "short pricing impact or null",
  "affected_tabs": ["Pricing", "Exclusions"],
  "notes": "short note or null"
}

Rules:
- Use concise text suitable for a Proposed Changes UI card.
- Do not use markdown bullets.
- Keep changes to 3-6 items.
- Do not invent exact dollar amounts unless they appear in the provided JSON.
- affected_tabs must use only these display labels: Scope, Risks, Assumptions, Pricing, Exclusions, Quote Builder, Clarifications, Addenda.
- If there is no clear pricing impact, use null for pricing_impact.
"""


CHANGE_LABELS = {
    "addenda": "Addenda Changes",
    "scope": "Scope Changes",
    "risks": "Risk Changes",
    "assumptions": "Assumption Changes",
    "clarifications": "Clarification Changes",
    "exclusions": "Exclusion Changes",
    "pricing": "Pricing Changes",
}

DISPLAY_TABS = {
    "addenda": "Addenda",
    "scope": "Scope",
    "risks": "Risks",
    "assumptions": "Assumptions",
    "clarifications": "Clarifications",
    "exclusions": "Exclusions",
    "pricing": "Pricing",
}

ALLOWED_AFFECTED_TABS = {
    "scope": "Scope",
    "risk": "Risks",
    "risks": "Risks",
    "assumption": "Assumptions",
    "assumptions": "Assumptions",
    "pricing": "Pricing",
    "exclusion": "Exclusions",
    "exclusions": "Exclusions",
    "quote": "Quote Builder",
    "quote builder": "Quote Builder",
    "quote_builder": "Quote Builder",
    "quote draft": "Quote Builder",
    "quote_draft": "Quote Builder",
    "clarification": "Clarifications",
    "clarifications": "Clarifications",
    "addenda": "Addenda",
    "addendum": "Addenda",
}


class SectionReanalysisService:
    """Generate proposed-change summaries from section re-analysis output."""

    def __init__(self):
        self.openai_service = get_openai_service()

    def build_proposed_changes(
        self,
        tab: str,
        ai_instructions: str,
        previous: Dict[str, Any],
        updated: Dict[str, Any],
    ) -> ProposedChanges:
        messages = [
            {
                "role": "user",
                "content": (
                    f"Section tab: {tab}\n\n"
                    f"User AI instruction:\n{ai_instructions}\n\n"
                    "Previous section JSON:\n"
                    f"{json.dumps(previous, ensure_ascii=False, default=str)}\n\n"
                    "Updated section JSON:\n"
                    f"{json.dumps(updated, ensure_ascii=False, default=str)}"
                ),
            }
        ]

        payload = self.openai_service.generate_json(
            messages=messages,
            system_prompt=PROPOSED_CHANGES_SYSTEM_PROMPT,
            temperature=0.1,
            max_tokens=900,
        )

        return normalize_proposed_changes(
            tab=tab,
            payload=payload,
            ai_instructions=ai_instructions,
            previous=previous,
            updated=updated,
        )


def normalize_proposed_changes(
    tab: str,
    payload: Dict[str, Any],
    ai_instructions: str,
    previous: Dict[str, Any],
    updated: Dict[str, Any],
) -> ProposedChanges:
    changes = normalize_changes(payload.get("changes"))
    if not changes:
        changes = build_fallback_changes(tab, ai_instructions, previous, updated)

    return ProposedChanges(
        changes_label=clean_text(payload.get("changes_label"))
        or CHANGE_LABELS.get(tab, "Section Changes"),
        changes=changes,
        pricing_impact=clean_optional_text(payload.get("pricing_impact")),
        affected_tabs=normalize_affected_tabs(payload.get("affected_tabs"), tab),
        notes=clean_optional_text(payload.get("notes")),
    )


def normalize_changes(value: Any) -> List[str]:
    if isinstance(value, list):
        raw_items = value
    elif isinstance(value, str):
        raw_items = [line for line in value.splitlines() if line.strip()]
    else:
        raw_items = []

    changes = []
    seen = set()
    for item in raw_items:
        text = clean_text(item).lstrip("-* ")
        key = text.lower()
        if text and key not in seen:
            changes.append(text)
            seen.add(key)
        if len(changes) >= 6:
            break

    return changes


def normalize_affected_tabs(value: Any, active_tab: str) -> List[str]:
    raw_items = value if isinstance(value, list) else [value]
    tabs = []
    seen = set()

    for item in raw_items:
        text = clean_text(item).lower().replace("-", " ").strip()
        tab = ALLOWED_AFFECTED_TABS.get(text)
        if tab and tab not in seen:
            tabs.append(tab)
            seen.add(tab)

    active_display = DISPLAY_TABS.get(active_tab)
    if active_display and active_display not in seen:
        tabs.insert(0, active_display)

    return tabs


def build_fallback_changes(
    tab: str,
    ai_instructions: str,
    previous: Dict[str, Any],
    updated: Dict[str, Any],
) -> List[str]:
    previous_count = item_count(previous)
    updated_count = item_count(updated)
    display_tab = DISPLAY_TABS.get(tab, "Section")
    changes = [f"Applied AI instruction to {display_tab.lower()} analysis"]

    if previous_count is not None and updated_count is not None:
        if previous_count != updated_count:
            changes.append(
                f"Item count changed from {previous_count} to {updated_count}"
            )
        else:
            changes.append(f"Reviewed {updated_count} {display_tab.lower()} items")

    instruction = clean_text(ai_instructions)
    if instruction:
        changes.append(f"Instruction considered: {instruction}")

    return changes[:6]


def item_count(value: Dict[str, Any]) -> Optional[int]:
    if isinstance(value.get("items"), list):
        return len(value["items"])

    count = value.get("total_items")
    if isinstance(count, int):
        return count

    return None


def clean_optional_text(value: Any) -> Optional[str]:
    text = clean_text(value)
    if not text or text.lower() in {"none", "null", "n/a", "not applicable"}:
        return None
    return text


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


_section_reanalysis_service = None


def get_section_reanalysis_service() -> SectionReanalysisService:
    """Get or create the global section re-analysis service."""
    global _section_reanalysis_service
    if _section_reanalysis_service is None:
        _section_reanalysis_service = SectionReanalysisService()
    return _section_reanalysis_service
