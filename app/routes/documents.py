"""
API routes for document management and file upload.
Handles uploading, processing, and managing documents.
"""

import os
from typing import List, Optional, Tuple
from fastapi import (
    APIRouter, Depends, HTTPException, status, UploadFile, File, Form
)
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Document
from app.schemas import DocumentResponse, DocumentUploadResponse
from app.services.file_extractor import validate_file, get_file_type
from app.services.rag_service import get_rag_service
from app.utils.analysis_inputs import divisions_to_json, normalize_divisions
from app.utils.scopes import is_valid_scope_value, normalize_scope_value
from app.config import settings
import logging

router = APIRouter(prefix="/documents", tags=["documents"])
logger = logging.getLogger(__name__)

SOURCE_TYPE_DOCUMENT = "document"
SOURCE_TYPE_ADDENDUM = "addendum"


def save_uploaded_file(
    file: UploadFile,
    project_id: str,
    user_id: str,
    source_type: str = SOURCE_TYPE_DOCUMENT,
) -> str:
    """
    Save uploaded file to disk.

    Args:
        file: Uploaded file
        project_id: Project ID
        user_id: User ID

    Returns:
        Path to saved file

    Raises:
        HTTPException: If file size exceeds limit
    """
    upload_base = settings.upload_dir
    user_project_dir = os.path.join(
        upload_base,
        str(user_id),
        str(project_id),
        source_type,
    )
    os.makedirs(user_project_dir, exist_ok=True)

    file_content = file.file.read()
    file_size = len(file_content)

    if file_size > settings.max_file_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds maximum allowed size of {settings.max_file_size / (1024*1024):.1f} MB"
        )

    file_path = os.path.join(user_project_dir, file.filename)
    with open(file_path, 'wb') as f:
        f.write(file_content)

    file.file.seek(0)
    return file_path


def _collect_uploads(
    files: Optional[List[UploadFile]],
    addendum: Optional[List[UploadFile]],
) -> List[Tuple[UploadFile, str]]:
    uploads = []
    for file in files or []:
        uploads.append((file, SOURCE_TYPE_DOCUMENT))
    for file in addendum or []:
        uploads.append((file, SOURCE_TYPE_ADDENDUM))
    return uploads


def _upload_response(
    filename: str,
    file_type: str,
    source_type: str,
    project_name: str,
    project_address: str,
    divisions: List[str],
    instructions: str,
    total_chunks: int,
    status_value: str,
    message: str,
    document_id: int = 0,
) -> DocumentUploadResponse:
    return DocumentUploadResponse(
        document_id=document_id,
        filename=filename,
        file_type=file_type,
        source_type=source_type,
        project_name=project_name,
        project_address=project_address,
        divisions=divisions,
        instructions=instructions,
        total_chunks=total_chunks,
        status=status_value,
        message=message,
    )


def _process_uploaded_file(
    file: UploadFile,
    source_type: str,
    user_id: str,
    project_id: str,
    project_name: str,
    project_address: str,
    divisions: List[str],
    instructions: str,
    db: Session,
    rag_service,
) -> DocumentUploadResponse:
    filename = file.filename or "unknown"
    try:
        if not file.filename:
            return _upload_response(
                filename="unknown",
                file_type="",
                source_type=source_type,
                project_name=project_name,
                project_address=project_address,
                divisions=divisions,
                instructions=instructions,
                total_chunks=0,
                status_value="error",
                message="Invalid filename",
            )

        file_type = get_file_type(file.filename)
        is_valid, validation_message = validate_file(
            file.filename,
            settings.max_file_size,
            settings.allowed_file_types,
        )

        if not is_valid:
            return _upload_response(
                filename=file.filename,
                file_type=file_type or "",
                source_type=source_type,
                project_name=project_name,
                project_address=project_address,
                divisions=divisions,
                instructions=instructions,
                total_chunks=0,
                status_value="error",
                message=validation_message,
            )

        try:
            file_path = save_uploaded_file(file, project_id, user_id, source_type)
        except HTTPException as exc:
            return _upload_response(
                filename=file.filename,
                file_type=file_type,
                source_type=source_type,
                project_name=project_name,
                project_address=project_address,
                divisions=divisions,
                instructions=instructions,
                total_chunks=0,
                status_value="error",
                message=str(exc.detail),
            )

        db_document = Document(
            filename=file.filename,
            file_path=file_path,
            file_type=file_type,
            source_type=source_type,
            project_id=project_id,
            user_id=user_id,
            project_name=project_name,
            project_address=project_address,
            divisions=divisions_to_json(divisions),
            instructions=instructions,
            total_chunks=0,
        )
        db.add(db_document)
        db.commit()
        db.refresh(db_document)

        total_chunks, error = rag_service.process_document(
            file_path=file_path,
            file_type=file_type,
            document_id=db_document.id,
            user_id=user_id,
            project_id=project_id,
            project_name=project_name,
            project_address=project_address,
            filename=file.filename,
            source_type=source_type,
            db=db,
        )

        if error:
            return _upload_response(
                document_id=db_document.id,
                filename=file.filename,
                file_type=file_type,
                source_type=source_type,
                project_name=project_name,
                project_address=project_address,
                divisions=divisions,
                instructions=instructions,
                total_chunks=0,
                status_value="error",
                message=error,
            )

        return _upload_response(
            document_id=db_document.id,
            filename=file.filename,
            file_type=file_type,
            source_type=source_type,
            project_name=project_name,
            project_address=project_address,
            divisions=divisions,
            instructions=instructions,
            total_chunks=total_chunks,
            status_value="success",
            message="Document processed successfully",
        )

    except Exception as exc:
        logger.error(f"Error uploading document {filename}: {str(exc)}")
        return _upload_response(
            filename=filename,
            file_type=get_file_type(filename) or "",
            source_type=source_type,
            project_name=project_name,
            project_address=project_address,
            divisions=divisions,
            instructions=instructions,
            total_chunks=0,
            status_value="error",
            message=f"Error uploading file: {str(exc)}",
        )


@router.post("/upload", response_model=List[DocumentUploadResponse])
async def upload_documents(
    user_id: str = Form(...),
    project_id: str = Form(...),
    project_name: str = Form(""),
    project_address: str = Form(""),
    divisions: Optional[List[str]] = Form(None),
    instructions: str = Form(""),
    files: Optional[List[UploadFile]] = File(None),
    addendum: Optional[List[UploadFile]] = File(None),
    db: Session = Depends(get_db),
) -> List[DocumentUploadResponse]:
    """
    Upload and process tender documents plus optional addendum files.

    Supports: PDF, DOCX, TXT files.
    Processing is synchronous so chat and analysis can use the full knowledge
    base immediately. Addendum files are tagged separately in metadata.
    """
    uploads = _collect_uploads(files, addendum)
    if not uploads:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files or addendum files provided",
        )

    user_id = normalize_scope_value(user_id)
    project_id = normalize_scope_value(project_id)
    project_name = (project_name or "").strip()
    project_address = (project_address or "").strip()
    divisions = normalize_divisions(divisions)
    instructions = (instructions or "").strip()

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

    if len(instructions) > 5000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Instructions are too long (max 5000 characters)",
        )

    rag_service = get_rag_service()
    return [
        _process_uploaded_file(
            file=file,
            source_type=source_type,
            user_id=user_id,
            project_id=project_id,
            project_name=project_name,
            project_address=project_address,
            divisions=divisions,
            instructions=instructions,
            db=db,
            rag_service=rag_service,
        )
        for file, source_type in uploads
    ]


@router.get("", response_model=List[DocumentResponse])
async def get_documents(
    user_id: str,
    project_id: str,
    db: Session = Depends(get_db)
) -> List[DocumentResponse]:
    """
    Get all documents for a user/project scope.

    Args:
        user_id: User ID scope
        project_id: Project ID scope
        db: Database session

    Returns:
        List of documents
    """
    user_id = normalize_scope_value(user_id)
    project_id = normalize_scope_value(project_id)

    if not is_valid_scope_value(user_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="user_id is required"
        )

    if not is_valid_scope_value(project_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="project_id is required"
        )

    documents = db.query(Document).filter(
        Document.project_id == project_id,
        Document.user_id == user_id
    ).all()

    return documents


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: int,
    user_id: str,
    project_id: str,
    db: Session = Depends(get_db)
) -> None:
    """
    Delete a document and its embeddings.

    Args:
        document_id: Document ID
        user_id: User ID scope
        project_id: Project ID scope
        db: Database session

    Raises:
        HTTPException: For various validation errors
    """
    user_id = normalize_scope_value(user_id)
    project_id = normalize_scope_value(project_id)

    if not is_valid_scope_value(user_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="user_id is required"
        )

    if not is_valid_scope_value(project_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="project_id is required"
        )

    document = db.query(Document).filter(
        Document.id == document_id,
        Document.project_id == project_id,
        Document.user_id == user_id
    ).first()

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )

    rag_service = get_rag_service()
    try:
        rag_service.qdrant_service.delete_by_document(
            document_id=document_id,
            user_id=user_id,
            project_id=project_id
        )
    except Exception as e:
        logger.error(f"Error deleting document from Qdrant: {str(e)}")

    try:
        if document.file_path and os.path.exists(document.file_path):
            os.remove(document.file_path)
    except Exception as e:
        logger.error(f"Error deleting file from disk: {str(e)}")

    db.delete(document)
    db.commit()
