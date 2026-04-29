"""
API routes for tender analysis.
"""

import logging
import re

from fastapi import APIRouter, HTTPException, status

from app.schemas import TenderAnalysisRequest, TenderAnalysisResponse
from app.services.analysis_service import get_analysis_service
from app.utils.scopes import is_valid_scope_value, normalize_scope_value


router = APIRouter(prefix="/analysis", tags=["analysis"])
logger = logging.getLogger(__name__)


@router.post("/tender", response_model=TenderAnalysisResponse)
async def analyze_tender(
    request: TenderAnalysisRequest,
) -> TenderAnalysisResponse:
    """
    Analyze uploaded project tender documents for selected CSI divisions.

    This uses the existing RAG index for the supplied user/project scope and
    returns a structured preview suitable for the analysis screen.
    """
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
            detail="At least one division must be selected",
        )

    if not any(re.search(r"\d{1,2}", str(division or "")) for division in request.divisions):
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
        return analysis_service.analyze_tender(
            user_id=user_id,
            project_id=project_id,
            divisions=request.divisions,
            instructions=request.instructions,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.error(f"Error generating tender analysis: {str(exc)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating tender analysis: {str(exc)}",
        ) from exc
