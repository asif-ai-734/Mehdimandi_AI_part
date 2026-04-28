"""
Pydantic schemas for request/response validation.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict


class DocumentResponse(BaseModel):
    """Schema for document response."""
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    filename: str
    file_type: str
    user_id: int
    project_id: int
    total_chunks: int
    created_at: datetime


class DocumentUploadResponse(BaseModel):
    """Schema for document upload response."""
    document_id: int
    filename: str
    file_type: str
    total_chunks: int
    status: str
    message: Optional[str] = None


class ChatRequest(BaseModel):
    """Schema for chat request."""
    user_id: int
    project_id: int
    message: str


class ChatResponse(BaseModel):
    """Schema for chat response."""
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    user_id: int
    project_id: int
    user_message: str
    assistant_response: str
    sources: Optional[List[str]] = None
    created_at: datetime


class ChatHistoryResponse(BaseModel):
    """Schema for chat history response."""
    total_messages: int
    messages: List[ChatResponse]


class ErrorResponse(BaseModel):
    """Schema for error response."""
    status_code: int
    message: str
    details: Optional[dict] = None
