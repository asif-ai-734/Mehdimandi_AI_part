"""
Runtime system configuration routes.
"""

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.config import settings
from app.services.openai_service import reset_openai_service


router = APIRouter(prefix="/system", tags=["system"])


class SystemSettingsRequest(BaseModel):
    """Runtime settings supplied by the user."""

    model_config = ConfigDict(coerce_numbers_to_str=True)

    api_key: str = Field(..., min_length=1)
    model_name: str = Field(..., min_length=1)
    upload_size: int = Field(..., gt=0)

    @model_validator(mode="before")
    @classmethod
    def accept_common_aliases(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        data = dict(data)
        if "api_key" not in data:
            for key in ("open_ai_api_key", "open_ai_api_kwy", "openai_api_key"):
                if key in data:
                    data["api_key"] = data[key]
                    break

        if "model_name" not in data:
            for key in ("model_version", "openai_model", "model"):
                if key in data:
                    data["model_name"] = data[key]
                    break

        if "upload_size" not in data:
            for key in ("max_file_size", "max_upload_size"):
                if key in data:
                    data["upload_size"] = data[key]
                    break

        return data

    @field_validator("api_key", "model_name")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Value cannot be empty")
        return value


class SystemSettingsResponse(BaseModel):
    """Response for updated runtime settings."""

    status: str
    api_key_set: bool
    model_name: str
    upload_size: int
    upload_size_mb: float


@router.post("", response_model=SystemSettingsResponse)
async def update_system_settings(
    payload: SystemSettingsRequest,
) -> SystemSettingsResponse:
    """Update runtime settings used by OpenAI calls and document uploads."""
    settings.api_key = payload.api_key
    settings.model_name = payload.model_name
    settings.upload_size = payload.upload_size
    reset_openai_service()

    return SystemSettingsResponse(
        status="success",
        api_key_set=bool(settings.api_key),
        model_name=settings.model_name,
        upload_size=settings.upload_size,
        upload_size_mb=round(settings.upload_size / (1024 * 1024), 2),
    )
