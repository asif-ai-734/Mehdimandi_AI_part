"""
Pydantic schemas for request/response validation.
"""

from datetime import datetime
from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.utils.analysis_inputs import normalize_divisions


class DocumentResponse(BaseModel):
    """Schema for document response."""
    model_config = ConfigDict(from_attributes=True, coerce_numbers_to_str=True)
    
    id: int
    filename: str
    file_type: str
    source_type: str = "document"
    user_id: str
    project_id: str
    project_name: str = ""
    project_address: str = ""
    divisions: List[str] = Field(default_factory=list)
    instructions: str = ""
    file_size: Optional[int] = None
    page_count: Optional[int] = None
    total_chunks: int
    created_at: datetime

    @field_validator("divisions", mode="before")
    @classmethod
    def normalize_document_divisions(cls, value: Any) -> List[str]:
        return normalize_divisions(value)


class DocumentUploadResponse(BaseModel):
    """Schema for document upload response."""
    model_config = ConfigDict(coerce_numbers_to_str=True)

    document_id: int
    filename: str
    file_type: str
    source_type: str = "document"
    project_name: str = ""
    project_address: str = ""
    divisions: List[str] = Field(default_factory=list)
    instructions: str = ""
    file_size: Optional[int] = None
    page_count: Optional[int] = None
    total_chunks: int
    status: str
    message: Optional[str] = None

    @field_validator("divisions", mode="before")
    @classmethod
    def normalize_upload_divisions(cls, value: Any) -> List[str]:
        return normalize_divisions(value)


class ChatRequest(BaseModel):
    """Schema for chat request."""
    model_config = ConfigDict(coerce_numbers_to_str=True)

    user_id: str
    project_id: str
    message: str
    page: Optional[str] = None
    page_json: Optional[Any] = None


class ChatResponse(BaseModel):
    """Schema for chat response."""
    model_config = ConfigDict(from_attributes=True, coerce_numbers_to_str=True)
    
    id: int
    user_id: str
    project_id: str
    user_message: str
    assistant_response: str
    sources: Optional[List[str]] = None
    created_at: datetime


class ChatHistoryResponse(BaseModel):
    """Schema for chat history response."""
    total_messages: int
    messages: List[ChatResponse]


class TenderAnalysisRequest(BaseModel):
    """Request schema for tender analysis."""
    model_config = ConfigDict(coerce_numbers_to_str=True)

    user_id: str
    project_id: str
    divisions: List[str] = Field(default_factory=list)
    instructions: str = ""

    @field_validator("divisions", mode="before")
    @classmethod
    def normalize_request_divisions(cls, value: Any) -> List[str]:
        return normalize_divisions(value)

    @model_validator(mode="before")
    @classmethod
    def accept_division_alias(cls, data: Any) -> Any:
        """Allow clients to send division, divion, or divisions."""
        if isinstance(data, dict) and "divisions" not in data and (
            "division" in data or "divion" in data
        ):
            division = data.get("division", data.get("divion"))
            data["divisions"] = division if isinstance(division, list) else [division]
        return data


class SelectedDivision(BaseModel):
    """Selected CSI division included in an analysis."""

    code: str
    label: str
    allocation_percent: int


class AnalysisMetrics(BaseModel):
    """Top-level analysis metrics used by the preview cards."""

    estimated_value: str
    duration: str
    labor_hours: str
    complexity: str
    risk_score: int


class SummaryPreview(BaseModel):
    """Text preview card with a badge."""

    title: str
    content: str
    badge: str


class ScopePreview(BaseModel):
    """Scope preview card."""

    title: str
    items: List[str]
    badge: str


class RiskPreviewItem(BaseModel):
    """Risk item with severity."""

    label: str
    severity: str


class RiskPreview(BaseModel):
    """Risk preview card."""

    title: str
    items: List[RiskPreviewItem]
    badge: str


class PricingImpactItem(BaseModel):
    """Pricing impact item with value."""

    label: str
    value: str


class PricingImpactPreview(BaseModel):
    """Pricing impacts preview card."""

    title: str
    items: List[PricingImpactItem]
    badge: str


class TenderAnalysisPreview(BaseModel):
    """Preview cards shown after tender analysis."""

    executive_summary: SummaryPreview
    scope_of_work: ScopePreview
    risk_assessment: RiskPreview
    pricing_impacts: PricingImpactPreview
    addenda_summary: SummaryPreview


class TenderAnalysisResponse(BaseModel):
    """Response schema for tender analysis preview."""
    model_config = ConfigDict(coerce_numbers_to_str=True)

    user_id: str
    project_id: str
    project_name: str = ""
    project_address: str = ""
    status: str
    instructions: str
    selected_divisions: List[SelectedDivision]
    metrics: AnalysisMetrics
    analysis_preview: TenderAnalysisPreview
    sources: List[str] = Field(default_factory=list)


class ProjectSummaryHighlight(BaseModel):
    """Key highlight item for the summary response."""

    title: str
    description: str
    type: str


class ProjectSummaryDivision(BaseModel):
    """Selected division item for the summary response."""

    code: str
    name: str


class ProjectSummaryResponse(BaseModel):
    """Flat summary response for the AI analysis results screen."""
    model_config = ConfigDict(coerce_numbers_to_str=True)

    estimated_value: int
    duration_weeks: int
    labor_hours: int
    total_items: int
    key_highlights: List[ProjectSummaryHighlight]
    selected_divisions: List[ProjectSummaryDivision]


class ErrorResponse(BaseModel):
    """Schema for error response."""
    status_code: int
    message: str
    details: Optional[dict] = None
