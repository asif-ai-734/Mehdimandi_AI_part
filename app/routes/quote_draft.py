"""
API route for the quote draft screen.
"""

import logging
from typing import Any, List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Document
from app.schemas.quote_draft import QuoteDraftResponse
from app.services.analysis_rules_service import merge_analysis_rules
from app.services.quote_draft_service import get_quote_draft_service
from app.utils.analysis_inputs import has_valid_division_code, normalize_divisions
from app.utils.scopes import is_valid_scope_value, normalize_scope_value


router = APIRouter(prefix="/quote-draft", tags=["analysis"])
logger = logging.getLogger(__name__)


@router.get("", response_model=QuoteDraftResponse)
async def get_quote_draft(
    user_id: str,
    project_id: str,
    db: Session = Depends(get_db),
) -> QuoteDraftResponse:
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

    divisions, instructions = get_saved_quote_draft_inputs(
        db=db,
        user_id=normalized_user_id,
        project_id=normalized_project_id,
    )

    payload = run_quote_draft_analysis(
        user_id=normalized_user_id,
        project_id=normalized_project_id,
        divisions=divisions,
        instructions=instructions,
    )

    return QuoteDraftResponse.model_validate(payload)


def get_saved_quote_draft_inputs(
    db: Session,
    user_id: str,
    project_id: str,
) -> tuple[List[str], str]:
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

    return divisions, merge_analysis_rules(
        db=db,
        user_id=user_id,
        instructions=instructions,
        section="quote_draft",
    )


def run_quote_draft_analysis(
    user_id: str,
    project_id: str,
    divisions: List[str],
    instructions: str,
) -> dict[str, Any]:
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

    if len(instructions) > 15000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Instructions are too long (max 15000 characters)",
        )

    try:
        service = get_quote_draft_service()

        return service.generate_quote_draft(
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
        logger.error(f"Error generating quote draft: {str(exc)}")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating quote draft: {str(exc)}",
        ) from exc
