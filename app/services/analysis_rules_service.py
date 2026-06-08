from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from app.models import AnalysisRules


RULE_FIELDS = {
    "general_instructions",
    "pricing_specific_instructions",
    "scope_analysis_instructions",
    "assumptions_instructions",
    "exclusions_instructions",
}

SECTION_RULE_FIELDS = {
    "pricing": "pricing_specific_instructions",
    "scope": "scope_analysis_instructions",
    "assumptions": "assumptions_instructions",
    "exclusions": "exclusions_instructions",
}

SECTION_LABELS = {
    "pricing": "Pricing-specific",
    "scope": "Scope analysis",
    "assumptions": "Assumptions",
    "exclusions": "Exclusions",
}


def get_analysis_rules(db: Session, user_id: str) -> Optional[AnalysisRules]:
    return db.query(AnalysisRules).filter(AnalysisRules.user_id == user_id).first()


def upsert_analysis_rules(
    db: Session,
    user_id: str,
    values: Dict[str, Any],
) -> AnalysisRules:
    rules = get_analysis_rules(db, user_id)

    if rules is None:
        rules = AnalysisRules(user_id=user_id)
        db.add(rules)

    for field, value in values.items():
        if field in RULE_FIELDS:
            setattr(rules, field, clean_instruction(value))

    db.commit()
    db.refresh(rules)
    return rules


def build_section_rules_text(
    rules: Optional[AnalysisRules],
    section: str,
) -> str:
    if rules is None:
        return ""

    section = (section or "").strip().lower()
    parts = []
    general = clean_instruction(rules.general_instructions)

    if general:
        parts.append(f"General analysis rules:\n{general}")

    section_field = SECTION_RULE_FIELDS.get(section)
    if section_field:
        section_rules = clean_instruction(getattr(rules, section_field, ""))
        if section_rules:
            label = SECTION_LABELS.get(section, section.title())
            parts.append(f"{label} rules:\n{section_rules}")

    return "\n\n".join(parts)


def merge_analysis_rules(
    db: Session,
    user_id: str,
    instructions: str,
    section: str,
) -> str:
    rules_text = build_section_rules_text(get_analysis_rules(db, user_id), section)
    base = clean_instruction(instructions)

    return "\n\n".join(part for part in (base, rules_text) if part)


def clean_instruction(value: Any) -> str:
    if value is None:
        return ""

    return str(value).strip()
