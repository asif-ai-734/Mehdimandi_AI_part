"""
API route for the analysis results summary screen.
"""

import logging
import re
from typing import Any, List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Document
from app.schemas import (
    ProjectSummaryDivision,
    ProjectSummaryHighlight,
    ProjectSummaryResponse,
    TenderAnalysisRequest,
    TenderAnalysisResponse,
)
from app.services.analysis_service import get_analysis_service
from app.utils.analysis_inputs import has_valid_division_code, normalize_divisions
from app.utils.scopes import is_valid_scope_value, normalize_scope_value


router = APIRouter(prefix="/summary", tags=["analysis"])
logger = logging.getLogger(__name__)


@router.get("", response_model=ProjectSummaryResponse)
async def get_summary(
    user_id: str,
    project_id: str,
    db: Session = Depends(get_db),
) -> ProjectSummaryResponse:
    """
    Return the dashboard-shaped AI analysis summary for a project.

    Inputs are only user_id and project_id. Saved upload divisions and
    instructions are used for analysis, and Total Items is the uploaded file
    count for the project.
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

    stored_divisions, stored_instructions, file_count = get_saved_summary_inputs(
        db=db,
        user_id=normalized_user_id,
        project_id=normalized_project_id,
    )

    analysis = run_summary_analysis(
        TenderAnalysisRequest(
            user_id=normalized_user_id,
            project_id=normalized_project_id,
            divisions=stored_divisions,
            instructions=stored_instructions,
        )
    )
    return build_project_summary(analysis, file_count=file_count)


def get_saved_summary_inputs(
    db: Session,
    user_id: str,
    project_id: str,
) -> tuple[List[str], str, int]:
    """Read saved divisions/instructions and uploaded file count for a project."""
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

    return divisions, instructions, len(documents)


def run_summary_analysis(request: TenderAnalysisRequest) -> TenderAnalysisResponse:
    """Validate summary input and call the existing analysis service."""
    user_id = normalize_scope_value(request.user_id)
    project_id = normalize_scope_value(request.project_id)

    if not is_valid_scope_value(user_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="user_id is required",
        )

    if not is_valid_scope_value(project_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="project_id is required",
        )

    if not request.divisions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No saved divisions found for this project",
        )

    if not has_valid_division_code(request.divisions):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one valid division code must be selected",
        )

    if len(request.instructions) > 5000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Instructions are too long (max 5000 characters)",
        )

    try:
        analysis_service = get_analysis_service()
        payload = analysis_service.analyze_tender(
            user_id=user_id,
            project_id=project_id,
            divisions=request.divisions,
            instructions=request.instructions,
        )
        return TenderAnalysisResponse.model_validate(payload)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.error(f"Error generating project summary: {str(exc)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating project summary: {str(exc)}",
        ) from exc


def build_project_summary(
    analysis: TenderAnalysisResponse,
    file_count: int,
) -> ProjectSummaryResponse:
    """Convert the structured analysis payload into the screenshot UI shape."""
    preview = analysis.analysis_preview

    return ProjectSummaryResponse(
        estimated_value=parse_int(analysis.metrics.estimated_value),
        duration_weeks=parse_int(analysis.metrics.duration),
        labor_hours=parse_int(analysis.metrics.labor_hours),
        total_items=file_count,
        key_highlights=[
            ProjectSummaryHighlight(
                title="Scope Summary",
                description=summary_text(
                    preview.scope_of_work.badge,
                    preview.scope_of_work.items,
                    preview.executive_summary.content,
                ),
                type="scope",
            ),
            ProjectSummaryHighlight(
                title="Pricing Impacts",
                description=pricing_text(preview.pricing_impacts.badge, preview.pricing_impacts.items),
                type="pricing",
            ),
            ProjectSummaryHighlight(
                title="Risks & Coordination",
                description=risk_text(preview.risk_assessment.badge, preview.risk_assessment.items),
                type="risk",
            ),
            ProjectSummaryHighlight(
                title="Addenda Changes",
                description=preview.addenda_summary.content,
                type="addenda",
            ),
        ],
        selected_divisions=[
            ProjectSummaryDivision(
                code=division.code,
                name=summary_division_name(division.code, division.label),
            )
            for division in analysis.selected_divisions
        ],
    )


def summary_text(badge: str, items: List[str], fallback: str) -> str:
    if items:
        return join_with_badge(badge, items)
    return fallback


def pricing_text(badge: str, items: List[Any]) -> str:
    values = [
        f"{item.label} ({item.value})" if item.value and item.value != "Not found" else item.label
        for item in items
    ]
    return join_with_badge(badge, values) if values else badge


def risk_text(badge: str, items: List[Any]) -> str:
    values = [
        f"{item.label} ({item.severity})" if item.severity else item.label
        for item in items
    ]
    return join_with_badge(badge, values) if values else badge


def join_with_badge(badge: str, values: List[str]) -> str:
    preview = "; ".join(values[:4])
    if len(values) > 4:
        preview += f"; +{len(values) - 4} more"
    return f"{badge}. {preview}" if badge else preview


def parse_int(value: str) -> int:
    """Extract an integer from formatted metric text like '$485,000'."""
    match = re.search(r"\d[\d,]*", value or "")
    if not match:
        return 0
    return int(match.group(0).replace(",", ""))


def summary_division_name(code: str, label: str) -> str:
    """Use concise names expected by the summary UI."""
    short_names = {
        "06": "Wood & Plastics",
    }
    return short_names.get(code, label)
