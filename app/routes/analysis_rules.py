"""
User-level AI analysis rules.
"""

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import AnalysisRules
from app.schemas.analysis_rules import AnalysisRulesRequest, AnalysisRulesResponse
from app.services.analysis_rules_service import (
    build_section_rules_text,
    upsert_analysis_rules,
)
from app.utils.scopes import is_valid_scope_value, normalize_scope_value


router = APIRouter(prefix="/analysis_rules", tags=["user setting"])
MAX_RULE_LENGTH = 5000


@router.patch("", response_model=AnalysisRulesResponse)
async def update_analysis_rules(
    request: AnalysisRulesRequest,
    db: Session = Depends(get_db),
) -> AnalysisRulesResponse:
    """Partially update user-level AI analysis rules."""
    user_id = normalize_scope_value(request.user_id)

    if not is_valid_scope_value(user_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="user_id is required",
        )

    values = extract_rule_updates(request)
    validate_rule_lengths(values)
    rules = upsert_analysis_rules(db=db, user_id=user_id, values=values)

    return build_analysis_rules_response(rules)


def extract_rule_updates(request: AnalysisRulesRequest) -> Dict[str, Any]:
    return {
        key: value
        for key, value in request.model_dump(exclude_unset=True).items()
        if key != "user_id"
    }


def validate_rule_lengths(values: Dict[str, Any]) -> None:
    for field, value in values.items():
        text = "" if value is None else str(value)
        if len(text) > MAX_RULE_LENGTH:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"{field} is too long (max {MAX_RULE_LENGTH} characters)",
            )


def build_analysis_rules_response(rules: AnalysisRules) -> AnalysisRulesResponse:
    return AnalysisRulesResponse(
        user_id=rules.user_id,
        general_instructions=rules.general_instructions or "",
        pricing_specific_instructions=rules.pricing_specific_instructions or "",
        scope_analysis_instructions=rules.scope_analysis_instructions or "",
        assumptions_instructions=rules.assumptions_instructions or "",
        exclusions_instructions=rules.exclusions_instructions or "",
        saved_pricing_instructions=build_section_rules_text(rules, "pricing"),
        saved_scope_instructions=build_section_rules_text(rules, "scope"),
        saved_assumptions_instructions=build_section_rules_text(rules, "assumptions"),
        saved_exclusions_instructions=build_section_rules_text(rules, "exclusions"),
    )
