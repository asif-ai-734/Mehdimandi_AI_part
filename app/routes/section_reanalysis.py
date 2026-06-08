"""
API route for section re-analysis with temporary AI instructions.
"""

import asyncio
import logging
from typing import Any, Callable, Dict, List

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app.routes.addenda import (
    build_addenda_response,
    get_saved_addenda_inputs,
    run_addenda_analysis,
)
from app.routes.assumptions import (
    build_assumptions_response,
    get_saved_assumption_inputs,
    run_assumptions_analysis,
)
from app.routes.clarifications import (
    build_clarifications_response,
    get_saved_clarification_inputs,
    run_clarifications_analysis,
)
from app.routes.exclusions import (
    build_exclusions_response,
    get_saved_exclusion_inputs,
    run_exclusions_analysis,
)
from app.routes.pricing import (
    build_pricing_response,
    get_saved_pricing_inputs,
    run_pricing_analysis,
)
from app.routes.risks import (
    build_risks_response,
    get_saved_risk_inputs,
    run_risk_analysis,
)
from app.routes.scope import (
    build_scope_response,
    get_saved_scope_inputs,
    run_scope_analysis,
)
from app.schemas.section_reanalysis import (
    ProposedChanges,
    SectionReanalysisRequest,
    SectionReanalysisResponse,
)
from app.services.section_reanalysis_service import (
    get_section_reanalysis_service,
    normalize_proposed_changes,
)
from app.utils.scopes import is_valid_scope_value, normalize_scope_value


router = APIRouter(prefix="/analysis", tags=["analysis"])
logger = logging.getLogger(__name__)


SectionBuilder = Callable[[dict[str, Any]], BaseModel]
SavedInputGetter = Callable[[Session, str, str], tuple[List[str], str]]
AnalysisRunner = Callable[[str, str, List[str], str], dict[str, Any]]


TAB_ALIASES = {
    "addenda": "addenda",
    "addendum": "addenda",
    "clarification": "clarifications",
    "clarifications": "clarifications",
    "exclusion": "exclusions",
    "exclusions": "exclusions",
    "price": "pricing",
    "pricing": "pricing",
    "scope": "scope",
    "risk": "risks",
    "risks": "risks",
    "assumption": "assumptions",
    "assumptions": "assumptions",
}


TAB_HANDLERS: Dict[str, tuple[SavedInputGetter, AnalysisRunner, SectionBuilder]] = {
    "addenda": (
        get_saved_addenda_inputs,
        run_addenda_analysis,
        build_addenda_response,
    ),
    "scope": (
        get_saved_scope_inputs,
        run_scope_analysis,
        build_scope_response,
    ),
    "clarifications": (
        get_saved_clarification_inputs,
        run_clarifications_analysis,
        build_clarifications_response,
    ),
    "exclusions": (
        get_saved_exclusion_inputs,
        run_exclusions_analysis,
        build_exclusions_response,
    ),
    "pricing": (
        get_saved_pricing_inputs,
        run_pricing_analysis,
        build_pricing_response,
    ),
    "risks": (
        get_saved_risk_inputs,
        run_risk_analysis,
        build_risks_response,
    ),
    "assumptions": (
        get_saved_assumption_inputs,
        run_assumptions_analysis,
        build_assumptions_response,
    ),
}


@router.post("/reanalyze", response_model=SectionReanalysisResponse)
async def reanalyze_section(
    request: SectionReanalysisRequest,
    db: Session = Depends(get_db),
) -> SectionReanalysisResponse:
    """
    Re-run a section with a temporary AI instruction and return proposed changes.
    """
    user_id = normalize_scope_value(request.user_id)
    project_id = normalize_scope_value(request.project_id)
    tab = normalize_tab(request.tab)
    ai_instructions = (request.ai_instructions or "").strip()

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

    if not ai_instructions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="ai_instructions is required",
        )

    if len(ai_instructions) > 5000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="AI instructions are too long (max 5000 characters)",
        )

    get_saved_inputs, run_analysis, build_response = TAB_HANDLERS[tab]
    divisions, saved_instructions = get_saved_inputs(
        db=db,
        user_id=user_id,
        project_id=project_id,
    )

    updated_instructions = merge_instructions(
        saved_instructions=saved_instructions,
        ai_instructions=ai_instructions,
        tab=tab,
    )
    previous_payload, updated_payload = await asyncio.gather(
        run_in_threadpool(
            run_analysis,
            user_id=user_id,
            project_id=project_id,
            divisions=divisions,
            instructions=saved_instructions,
        ),
        run_in_threadpool(
            run_analysis,
            user_id=user_id,
            project_id=project_id,
            divisions=divisions,
            instructions=updated_instructions,
        ),
    )
    previous = dump_response(build_response(previous_payload))

    updated = dump_response(build_response(updated_payload))

    proposed_changes = await run_in_threadpool(
        build_proposed_changes,
        tab=tab,
        ai_instructions=ai_instructions,
        previous=previous,
        updated=updated,
    )

    return SectionReanalysisResponse(
        user_id=user_id,
        project_id=project_id,
        tab=tab,
        ai_instructions=ai_instructions,
        previous=previous,
        updated=updated,
        proposed_changes=proposed_changes,
    )


def normalize_tab(value: str) -> str:
    tab = TAB_ALIASES.get((value or "").strip().lower())
    if not tab:
        supported = ", ".join(sorted(TAB_HANDLERS.keys()))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported tab '{value}'. Supported tabs: {supported}",
        )
    return tab


def merge_instructions(
    saved_instructions: str,
    ai_instructions: str,
    tab: str,
) -> str:
    parts = []
    saved = (saved_instructions or "").strip()
    if saved:
        parts.append(saved)

    parts.append(
        f"Temporary AI instruction for this {tab} re-analysis: {ai_instructions}"
    )
    return "\n\n".join(parts)


def dump_response(response: BaseModel) -> Dict[str, Any]:
    return response.model_dump()


def build_proposed_changes(
    tab: str,
    ai_instructions: str,
    previous: Dict[str, Any],
    updated: Dict[str, Any],
) -> ProposedChanges:
    try:
        service = get_section_reanalysis_service()
        return service.build_proposed_changes(
            tab=tab,
            ai_instructions=ai_instructions,
            previous=previous,
            updated=updated,
        )
    except Exception as exc:
        logger.error(f"Error generating proposed changes: {str(exc)}")
        return normalize_proposed_changes(
            tab=tab,
            payload={
                "notes": (
                    "AI change summary could not be generated; showing automatic summary."
                )
            },
            ai_instructions=ai_instructions,
            previous=previous,
            updated=updated,
        )
