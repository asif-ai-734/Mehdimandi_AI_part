from typing import Optional

from pydantic import BaseModel


class AnalysisRulesRequest(BaseModel):
    user_id: str
    general_instructions: Optional[str] = None
    pricing_specific_instructions: Optional[str] = None
    scope_analysis_instructions: Optional[str] = None
    assumptions_instructions: Optional[str] = None
    exclusions_instructions: Optional[str] = None


class AnalysisRulesResponse(BaseModel):
    user_id: str
    general_instructions: str = ""
    pricing_specific_instructions: str = ""
    scope_analysis_instructions: str = ""
    assumptions_instructions: str = ""
    exclusions_instructions: str = ""
    saved_pricing_instructions: str = ""
    saved_scope_instructions: str = ""
    saved_assumptions_instructions: str = ""
    saved_exclusions_instructions: str = ""
