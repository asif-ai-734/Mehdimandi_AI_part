"""
API routes for chat functionality.
Handles chat requests with document context retrieval.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import ChatHistory
from app.schemas import ChatRequest, ChatResponse, ChatHistoryResponse
from app.services.rag_service import get_rag_service
from app.utils.scopes import is_valid_scope_value, normalize_scope_value
import json
import logging

router = APIRouter(prefix="/chat", tags=["chat"])
logger = logging.getLogger(__name__)

SUPPORTED_CHAT_PAGES = {
    "summary",
    "scope",
    "pricing",
    "risks",
    "clarifications",
    "assumptions",
    "exclusions",
    "addenda",
    "quote_draft",
}


@router.post("", response_model=ChatResponse)
async def chat(
    chat_request: ChatRequest,
    db: Session = Depends(get_db)
) -> ChatResponse:
    """
    Send a chat message and get a response based on project documents.
    
    The assistant will answer only using the uploaded project documents.
    User ID and Project ID are supplied in the request body.
    
    Args:
        chat_request: Chat request with user_id, project_id, and user message
        db: Database session
    
    Returns:
        Chat response with assistant message and sources
    
    Raises:
        HTTPException: For various validation errors
    """
    user_id = normalize_scope_value(chat_request.user_id)
    project_id = normalize_scope_value(chat_request.project_id)
    
    # Validate input
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
    
    if not chat_request.message or len(chat_request.message.strip()) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message cannot be empty"
        )
    
    if len(chat_request.message) > 5000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message is too long (max 5000 characters)"
        )

    page = (chat_request.page or "").strip().lower()
    has_page_json = chat_request.page_json is not None

    if page and page not in SUPPORTED_CHAT_PAGES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported page '{page}'"
        )

    if page and not has_page_json:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="page_json is required when page is provided"
        )

    if has_page_json and not page:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="page is required when page_json is provided"
        )

    if has_page_json and not isinstance(chat_request.page_json, (dict, list)):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="page_json must be a JSON object or array"
        )
    
    try:
        # Get RAG service
        rag_service = get_rag_service()

        if page:
            response_text, sources = rag_service.generate_page_chat_response(
                query=chat_request.message,
                user_id=user_id,
                project_id=project_id,
                page=page,
                page_json=chat_request.page_json,
            )
        else:
            # Generate response with context from documents
            response_text, sources = rag_service.generate_chat_response(
                query=chat_request.message,
                user_id=user_id,
                project_id=project_id,
                db=db
            )
        
        # Save to chat history
        chat_history = rag_service.save_chat_history(
            user_id=user_id,
            project_id=project_id,
            user_message=chat_request.message,
            assistant_response=response_text,
            sources=sources,
            db=db
        )
        
        # Convert sources from JSON if stored as string
        stored_sources = sources
        if chat_history.sources:
            try:
                stored_sources = json.loads(chat_history.sources)
            except (json.JSONDecodeError, TypeError):
                stored_sources = []
        
        return ChatResponse(
            id=chat_history.id,
            user_id=chat_history.user_id,
            project_id=chat_history.project_id,
            user_message=chat_history.user_message,
            assistant_response=chat_history.assistant_response,
            sources=stored_sources,
            created_at=chat_history.created_at
        )
    
    except Exception as e:
        logger.error(f"Error generating chat response: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating response: {str(e)}"
        )


@router.get("/history", response_model=ChatHistoryResponse)
async def get_chat_history(
    user_id: str,
    project_id: str,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db)
) -> ChatHistoryResponse:
    """
    Get chat history for a project.
    
    Args:
        user_id: User ID scope
        project_id: Project ID scope
        limit: Maximum number of messages to return (default 50, max 100)
        offset: Number of messages to skip
        db: Database session
    
    Returns:
        Chat history with total count and messages
    
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
    
    # Validate pagination
    limit = min(limit, 100)
    if limit < 1:
        limit = 10
    if offset < 0:
        offset = 0
    
    # Query chat history
    query = db.query(ChatHistory).filter(
        ChatHistory.project_id == project_id,
        ChatHistory.user_id == user_id
    )
    
    total = query.count()
    messages = query.order_by(ChatHistory.created_at.desc()).offset(offset).limit(limit).all()
    
    # Convert to response format
    chat_responses = []
    for chat in reversed(messages):  # Reverse to show chronological order
        sources = []
        if chat.sources:
            try:
                sources = json.loads(chat.sources)
            except (json.JSONDecodeError, TypeError):
                sources = []
        
        chat_responses.append(ChatResponse(
            id=chat.id,
            user_id=chat.user_id,
            project_id=chat.project_id,
            user_message=chat.user_message,
            assistant_response=chat.assistant_response,
            sources=sources,
            created_at=chat.created_at
        ))
    
    return ChatHistoryResponse(
        total_messages=total,
        messages=chat_responses
    )


@router.delete("/history", status_code=status.HTTP_204_NO_CONTENT)
async def clear_chat_history(
    user_id: str,
    project_id: str,
    db: Session = Depends(get_db)
) -> None:
    """
    Clear all chat history for a project.
    
    Args:
        user_id: User ID scope
        project_id: Project ID scope
        db: Database session
    
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
    
    # Delete all chat history for this project
    db.query(ChatHistory).filter(
        ChatHistory.project_id == project_id,
        ChatHistory.user_id == user_id
    ).delete()
    
    db.commit()
