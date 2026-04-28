"""
API routes for document management and file upload.
Handles uploading, processing, and managing documents.
"""

import os
from typing import List, Annotated
from fastapi import (
    APIRouter, Depends, HTTPException, status, UploadFile, File, Form
)
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Document
from app.schemas import DocumentResponse, DocumentUploadResponse
from app.services.file_extractor import validate_file, get_file_type
from app.services.rag_service import get_rag_service
from app.config import settings
import logging

router = APIRouter(prefix="/documents", tags=["documents"])
logger = logging.getLogger(__name__)


def save_uploaded_file(file: UploadFile, project_id: int, user_id: int) -> str:
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
    user_project_dir = os.path.join(upload_base, str(user_id), str(project_id))
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


@router.post("/upload", response_model=List[DocumentUploadResponse])
async def upload_documents(
    user_id: int = Form(...),
    project_id: int = Form(...),
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
) -> List[DocumentUploadResponse]:
    """
    Upload and process documents for a user/project scope.

    Supports: PDF, DOCX, TXT files.
    Processing is synchronous so chat can use the knowledge base immediately.

    Args:
        user_id: User ID scope
        project_id: Project ID scope
        files: List of uploaded files
        db: Database session

    Returns:
        List of upload responses with document IDs and status

    Raises:
        HTTPException: For various validation errors
    """
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided"
        )

    if user_id <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="user_id must be a positive integer"
        )

    if project_id <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="project_id must be a positive integer"
        )

    responses = []
    rag_service = get_rag_service()

    for file in files:
        try:
            if not file.filename:
                responses.append(DocumentUploadResponse(
                    document_id=0,
                    filename="unknown",
                    file_type="",
                    total_chunks=0,
                    status="error",
                    message="Invalid filename"
                ))
                continue

            file_type = get_file_type(file.filename)
            is_valid, validation_message = validate_file(
                file.filename,
                settings.max_file_size,
                settings.allowed_file_types
            )

            if not is_valid:
                responses.append(DocumentUploadResponse(
                    document_id=0,
                    filename=file.filename,
                    file_type=file_type or "",
                    total_chunks=0,
                    status="error",
                    message=validation_message
                ))
                continue

            try:
                file_path = save_uploaded_file(file, project_id, user_id)
            except HTTPException as e:
                responses.append(DocumentUploadResponse(
                    document_id=0,
                    filename=file.filename,
                    file_type=file_type,
                    total_chunks=0,
                    status="error",
                    message=e.detail
                ))
                continue

            db_document = Document(
                filename=file.filename,
                file_path=file_path,
                file_type=file_type,
                project_id=project_id,
                user_id=user_id,
                total_chunks=0
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
                filename=file.filename,
                db=db
            )

            if error:
                responses.append(DocumentUploadResponse(
                    document_id=db_document.id,
                    filename=file.filename,
                    file_type=file_type,
                    total_chunks=0,
                    status="error",
                    message=error
                ))
                continue

            responses.append(DocumentUploadResponse(
                document_id=db_document.id,
                filename=file.filename,
                file_type=file_type,
                total_chunks=total_chunks,
                status="success",
                message="Document processed successfully"
            ))

        except Exception as e:
            logger.error(f"Error uploading document {file.filename}: {str(e)}")
            responses.append(DocumentUploadResponse(
                document_id=0,
                filename=file.filename,
                file_type=get_file_type(file.filename) or "",
                total_chunks=0,
                status="error",
                message=f"Error uploading file: {str(e)}"
            ))

    return responses


@router.get("", response_model=List[DocumentResponse])
async def get_documents(
    user_id: int,
    project_id: int,
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
    documents = db.query(Document).filter(
        Document.project_id == project_id,
        Document.user_id == user_id
    ).all()

    return documents


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: int,
    user_id: int,
    project_id: int,
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