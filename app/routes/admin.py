"""
Admin API routes for project/user visibility and AI processing logs.
"""

from datetime import datetime
from typing import Any, List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Document
from app.utils.analysis_inputs import normalize_divisions


router = APIRouter(prefix="/admin", tags=["admin"])


class AdminUserProject(BaseModel):
    user_id: str
    project_id: str
    project_name: str = ""


class AiLogActions(BaseModel):
    view: bool = True
    retry: bool = False
    review_and_fix: bool = False


class AiLogDocumentDetails(BaseModel):
    file_name: str
    document_id: str
    file_type: str = ""
    source_type: str = "document"


class AiLogProjectDetails(BaseModel):
    user_id: str
    user_owner: str
    project_id: str
    project_name: str = ""


class AiLogProcessingDetails(BaseModel):
    stage: str
    status: str
    error_message: Optional[str] = None
    divisions_selected: List[str] = Field(default_factory=list)
    instructions_given: str = ""


class AiLogTimingDetails(BaseModel):
    timestamp: str
    duration: str


class AiLogOutputDetails(BaseModel):
    output_version: str = "v2"
    total_chunks: int = 0
    page_count: Optional[int] = None
    file_size: Optional[int] = None


class AiLogDetails(BaseModel):
    document: AiLogDocumentDetails
    project: AiLogProjectDetails
    processing: AiLogProcessingDetails
    timing: AiLogTimingDetails
    output: AiLogOutputDetails


class AiLogItem(BaseModel):
    file_name: str
    project_name: str = ""
    stage: str
    status: str
    error_message: Optional[str] = None
    timestamp: str
    duration: str
    actions: AiLogActions
    details: AiLogDetails


@router.get("/users", response_model=List[AdminUserProject])
async def get_admin_users(db: Session = Depends(get_db)) -> List[AdminUserProject]:
    """
    Return every user/project pair that has uploaded documents.
    """
    documents = (
        db.query(Document)
        .order_by(
            Document.user_id.asc(),
            Document.project_id.asc(),
            Document.created_at.desc(),
        )
        .all()
    )

    projects = []
    seen = set()
    for document in documents:
        user_id = to_response_text(document.user_id)
        project_id = to_response_text(document.project_id)
        key = (user_id, project_id)
        if key in seen:
            continue

        seen.add(key)
        projects.append(
            AdminUserProject(
                user_id=user_id,
                project_id=project_id,
                project_name=to_response_text(document.project_name),
            )
        )

    return projects


@router.get("/ai-logs", response_model=List[AiLogItem])
async def get_ai_logs(db: Session = Depends(get_db)) -> List[AiLogItem]:
    """
    Return AI processing logs shaped for the admin log table and detail modal.
    """
    documents = db.query(Document).order_by(Document.created_at.desc()).all()
    return [build_ai_log_item(document) for document in documents]


def build_ai_log_item(document: Document) -> AiLogItem:
    status_value = get_log_status(document)
    error_message = get_log_error_message(document, status_value)
    stage = get_log_stage(document)
    timestamp = format_log_timestamp(document.created_at)
    duration = get_log_duration(document)
    actions = AiLogActions(
        view=True,
        retry=status_value == "Failed",
        review_and_fix=status_value == "Failed",
    )
    user_id = to_response_text(document.user_id)
    project_id = to_response_text(document.project_id)
    file_name = to_response_text(document.filename) or "Unnamed Document"
    project_name = to_response_text(document.project_name)

    return AiLogItem(
        file_name=file_name,
        project_name=project_name,
        stage=stage,
        status=status_value,
        error_message=error_message,
        timestamp=timestamp,
        duration=duration,
        actions=actions,
        details=AiLogDetails(
            document=AiLogDocumentDetails(
                file_name=file_name,
                document_id=f"DOC-{document.id}",
                file_type=document.file_type or "",
                source_type=document.source_type or "document",
            ),
            project=AiLogProjectDetails(
                user_id=user_id,
                user_owner=user_id,
                project_id=project_id,
                project_name=project_name,
            ),
            processing=AiLogProcessingDetails(
                stage=stage,
                status=status_value,
                error_message=error_message,
                divisions_selected=normalize_divisions(document.divisions),
                instructions_given=document.instructions or "",
            ),
            timing=AiLogTimingDetails(
                timestamp=timestamp,
                duration=duration,
            ),
            output=AiLogOutputDetails(
                output_version="v2",
                total_chunks=document.total_chunks or 0,
                page_count=document.page_count,
                file_size=document.file_size,
            ),
        ),
    )


def get_log_stage(document: Document) -> str:
    if (document.source_type or "").lower() == "addendum":
        return "Analysis"
    return "Extraction"


def get_log_status(document: Document) -> str:
    if document.total_chunks and document.total_chunks > 0:
        return "Success"
    return "Failed"


def get_log_error_message(document: Document, status_value: str) -> Optional[str]:
    if status_value == "Success":
        return None
    return "No AI chunks were created for this document"


def get_log_duration(document: Document) -> str:
    if document.total_chunks and document.total_chunks > 0:
        return f"{max(1, document.total_chunks)}s"
    return "0s"


def format_log_timestamp(value: Optional[datetime]) -> str:
    if not value:
        return ""
    return value.strftime("%Y-%m-%d %H:%M:%S")


def to_response_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)
