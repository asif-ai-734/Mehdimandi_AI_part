"""
API route for source/resource documents screen.
"""

import os
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.database import get_db
from app.models import Document
from app.services.file_extractor import (
    get_file_type,
    get_page_count as get_file_page_count,
)
from app.utils.scopes import (
    is_valid_scope_value,
    normalize_scope_value,
)


router = APIRouter(prefix="/resources", tags=["analysis"])


# -------------------------------------------------------------------
# Schemas
# -------------------------------------------------------------------


class ResourceItem(BaseModel):
    id: int

    filename: str

    pages: Optional[int] = None

    uploaded_at: Optional[str] = None

    file_size: Optional[int] = None

    document_type: Optional[str] = None


class ResourcesResponse(BaseModel):
    title: str = "Source Documents"

    subtitle: str = "All documents analyzed by AI"

    total_items: int

    items: List[ResourceItem]


# -------------------------------------------------------------------
# Route
# -------------------------------------------------------------------


@router.get("", response_model=ResourcesResponse)
async def get_resources(
    user_id: str,
    project_id: str,
    db: Session = Depends(get_db),
) -> ResourcesResponse:
    """
    Return uploaded source documents for the project.
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

    documents = (
        db.query(Document)
        .filter(
            Document.project_id == normalized_project_id,
            Document.user_id == normalized_user_id,
        )
        .order_by(Document.created_at.asc())
        .all()
    )

    items = []

    for document in documents:

        uploaded_at = None

        if getattr(document, "created_at", None):
            try:
                uploaded_at = format_uploaded_date(
                    document.created_at
                )
            except Exception:
                uploaded_at = None

        items.append(
            ResourceItem(
                id=document.id,

                filename=get_filename(document),

                pages=get_page_count(document),

                uploaded_at=uploaded_at,

                file_size=get_file_size(document),

                document_type=get_document_type(document),
            )
        )

    return ResourcesResponse(
        total_items=len(items),
        items=items,
    )


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def get_filename(document: Document) -> str:
    """
    Safely resolve filename from document.
    """

    for field in [
        "filename",
        "file_name",
        "original_filename",
        "name",
    ]:
        value = getattr(document, field, None)

        if value:
            return str(value)

    return "Unnamed Document"


def get_page_count(document: Document) -> Optional[int]:
    """
    Safely resolve page count.
    """

    for field in [
        "page_count",
        "pages",
        "total_pages",
    ]:
        value = getattr(document, field, None)

        if isinstance(value, int):
            if value > 0:
                return value

        try:
            if value is not None:
                page_count = int(value)
                if page_count > 0:
                    return page_count
        except Exception:
            continue

    file_path = getattr(document, "file_path", None)
    file_type = getattr(document, "file_type", None) or get_file_type(
        get_filename(document)
    )

    if not file_path or not file_type:
        return None

    try:
        return get_file_page_count(file_path, file_type)
    except Exception:
        return None


def get_file_size(document: Document) -> Optional[int]:
    """
    Resolve stored file size, falling back to the saved file on disk.
    """

    value = getattr(document, "file_size", None)

    if isinstance(value, int) and value > 0:
        return value

    try:
        if value is not None and int(value) > 0:
            return int(value)
    except Exception:
        pass

    file_path = getattr(document, "file_path", None)

    if not file_path:
        return None

    try:
        if os.path.exists(file_path):
            return os.path.getsize(file_path)
    except Exception:
        return None

    return None


def get_document_type(document: Document) -> Optional[str]:
    """
    Infer document type from filename.
    """

    filename = get_filename(document).lower()

    if filename.endswith(".pdf"):
        return "PDF"

    if filename.endswith(".docx"):
        return "DOCX"

    if filename.endswith(".xlsx"):
        return "XLSX"

    if filename.endswith(".csv"):
        return "CSV"

    if filename.endswith(".txt"):
        return "TXT"

    return None


def format_uploaded_date(value: datetime) -> str:
    """
    Format uploaded date for frontend.
    Example:
    Apr 20, 2026
    """

    return value.strftime("%b %d, %Y")
