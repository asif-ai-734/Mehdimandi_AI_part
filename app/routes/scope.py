"""
API route for the scope of work screen.
"""

import logging
from typing import Any, List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Document
from app.schemas.scope import ScopeFilter, ScopeItem, ScopeOfWorkResponse
from app.services.analysis_rules_service import merge_analysis_rules
from app.services.scope_service import get_scope_service
from app.utils.analysis_inputs import has_valid_division_code, normalize_divisions
from app.utils.scopes import is_valid_scope_value, normalize_scope_value


router = APIRouter(prefix="/scope", tags=["analysis"])
logger = logging.getLogger(__name__)


@router.get("", response_model=ScopeOfWorkResponse)
async def get_scope(
    user_id: str,
    project_id: str,
    db: Session = Depends(get_db),
) -> ScopeOfWorkResponse:
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

    stored_divisions, stored_instructions = get_saved_scope_inputs(
        db=db,
        user_id=normalized_user_id,
        project_id=normalized_project_id,
    )

    payload = run_scope_analysis(
        user_id=normalized_user_id,
        project_id=normalized_project_id,
        divisions=stored_divisions,
        instructions=stored_instructions,
    )

    return build_scope_response(payload)


def get_saved_scope_inputs(
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
        section="scope",
    )


def run_scope_analysis(
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
        scope_service = get_scope_service()

        return scope_service.extract_scope_of_work(
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
        logger.error(f"Error generating scope of work: {str(exc)}")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating scope of work: {str(exc)}",
        ) from exc


def build_scope_response(payload: dict[str, Any]) -> ScopeOfWorkResponse:
    items = [
        ScopeItem.model_validate(item)
        for item in payload.get("items", [])
        if isinstance(item, dict)
    ]

    return ScopeOfWorkResponse(
        total_items=len(items),
        showing=f"{len(items)} of {len(items)}",
        filters=build_scope_filters(items),
        items=items,
    )


def build_scope_filters(items: List[ScopeItem]) -> List[ScopeFilter]:
    filters = [
        ScopeFilter(
            code="all",
            label="All Scope",
            active=True,
        )
    ]

    seen = set()

    for item in items:
        if item.division in seen:
            continue

        seen.add(item.division)

        filters.append(
            ScopeFilter(
                code=item.division,
                label=f"Div {item.division}",
                active=False,
            )
        )

    return filters
