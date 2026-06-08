"""
SQLAlchemy models for the RAG system.
Defines Document, ChatHistory, and user analysis-rule models.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Document(Base):
    """Document model for storing uploaded files metadata."""
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), index=True)
    file_path = Column(String(500))
    file_type = Column(String(10))  # pdf, docx, txt
    source_type = Column(String(20), default="document", nullable=False)  # document, addendum
    project_id = Column(String(255), index=True, nullable=False)
    user_id = Column(String(255), index=True, nullable=False)
    project_name = Column(String(255), default="", nullable=False)
    project_address = Column(String(500), default="", nullable=False)
    divisions = Column(Text, default="[]", nullable=False)
    instructions = Column(Text, default="", nullable=False)
    file_size = Column(Integer, default=0, nullable=False)
    page_count = Column(Integer, nullable=True)
    total_chunks = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)


class ChatHistory(Base):
    """Chat history model for storing conversations."""
    __tablename__ = "chat_history"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(255), index=True, nullable=False)
    project_id = Column(String(255), index=True, nullable=False)
    user_message = Column(Text)
    assistant_response = Column(Text)
    sources = Column(Text, nullable=True)  # JSON list of source filenames
    created_at = Column(DateTime, default=datetime.utcnow)


class AnalysisRules(Base):
    """User-level AI analysis rules applied across analysis screens."""
    __tablename__ = "analysis_rules"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(255), unique=True, index=True, nullable=False)
    general_instructions = Column(Text, default="", nullable=False)
    pricing_specific_instructions = Column(Text, default="", nullable=False)
    scope_analysis_instructions = Column(Text, default="", nullable=False)
    assumptions_instructions = Column(Text, default="", nullable=False)
    exclusions_instructions = Column(Text, default="", nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
