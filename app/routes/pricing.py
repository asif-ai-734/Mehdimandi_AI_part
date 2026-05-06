"""
API route for the pricing impacts screen.
"""

import logging
from typing import Any, List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Document
from app.schemas.pricing import (
    PricingFilter,
    PricingImpactItem,
    PricingImpactsResponse,
)
from app.services.pricing_service import get_pricing_service
from app.utils.analysis_inputs import has_valid_division_code, normalize_divisions
from app.utils.scopes import (
    is_valid_scope_value,
    normalize_scope_value,
)


router = APIRouter(prefix="/pricing", tags=["analysis"])
logger = logging.getLogger(__name__)


@router.get("", response_model=PricingImpactsResponse)
async def get_pricing(
    user_id: str,
    project_id: str,
    db: Session = Depends(get_db),
) -> PricingImpactsResponse:
    """
    Return structured pricing impacts for a project.
    """
    normalized_user_id = normalize_scope_value(user_id)
    normalized_project_id = normalize_scope_value(project_id)

    if not is_valid_scope_value(normalized_user_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="user_id is required",
        )

    if not is_valid_scope_value(normalized_project_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="project_id is required",
        )

    stored_divisions, stored_instructions = get_saved_pricing_inputs(
        db=db,
        user_id=normalized_user_id,
        project_id=normalized_project_id,
    )

    payload = run_pricing_analysis(
        user_id=normalized_user_id,
        project_id=normalized_project_id,
        divisions=stored_divisions,
        instructions=stored_instructions,
    )

    return build_pricing_response(payload)


def get_saved_pricing_inputs(
    db: Session,
    user_id: str,
    project_id: str,
) -> tuple[List[str], str]:
    """
    Read saved divisions and instructions for a project.
    """
    documents = db.query(Document).filter(
        Document.project_id == project_id,
        Document.user_id == user_id,
    ).order_by(Document.created_at.desc()).all()

    if not documents:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No uploaded documents found for this project",
        )

    divisions = []
    instructions = ""

    for document in documents:
        if not divisions:
            divisions = normalize_divisions(document.divisions)

        if not instructions:
            instructions = (document.instructions or "").strip()

        if divisions and instructions:
            break

    return divisions, instructions


def run_pricing_analysis(
    user_id: str,
    project_id: str,
    divisions: List[str],
    instructions: str,
) -> dict[str, Any]:
    """
    Validate pricing inputs and call pricing service.
    """
    if not divisions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No saved divisions found for this project",
        )

    if not has_valid_division_code(divisions):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one valid division code must be selected",
        )

    if len(instructions) > 5000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Instructions are too long (max 5000 characters)",
        )

    try:
        pricing_service = get_pricing_service()

        return pricing_service.extract_pricing_impacts(
            user_id=user_id,
            project_id=project_id,
            divisions=divisions,
            instructions=instructions,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.error(f"Error generating pricing impacts: {str(exc)}")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating pricing impacts: {str(exc)}",
        ) from exc


def build_pricing_response(
    payload: dict[str, Any],
) -> PricingImpactsResponse:
    """
    Convert pricing payload into frontend response shape.
    """
    items = [
        PricingImpactItem.model_validate(item)
        for item in payload.get("items", [])
        if isinstance(item, dict)
    ]

    return PricingImpactsResponse(
        total_items=len(items),
        showing=f"{len(items)} of {len(items)}",
        filters=build_pricing_filters(items),
        items=items,
    )


def build_pricing_filters(
    items: List[PricingImpactItem],
) -> List[PricingFilter]:
    """
    Build pricing filters.

    For now only "All Pricing" exists.
    Later you can add:
    - allowances
    - schedule
    - logistics
    - risk
    - addenda
    """
    return [
        PricingFilter(
            code="all",
            label="All Pricing",
            active=True,
        )
    ]
