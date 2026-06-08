from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class SectionReanalysisRequest(BaseModel):
    user_id: str
    project_id: str
    tab: str
    ai_instructions: str = ""

    @model_validator(mode="before")
    @classmethod
    def accept_instruction_aliases(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value

        data = dict(value)
        if not data.get("ai_instructions"):
            for alias in ("ai_instruction", "instructions"):
                if data.get(alias):
                    data["ai_instructions"] = data[alias]
                    break

        return data


class ProposedChanges(BaseModel):
    title: str = "Proposed Changes"
    changes_label: str
    changes: List[str] = Field(default_factory=list)
    pricing_impact: Optional[str] = None
    affected_tabs: List[str] = Field(default_factory=list)
    notes: Optional[str] = None


class SectionReanalysisResponse(BaseModel):
    user_id: str
    project_id: str
    tab: str
    ai_instructions: str
    previous: Dict[str, Any]
    updated: Dict[str, Any]
    proposed_changes: ProposedChanges
