from typing import List, Optional

from pydantic import BaseModel


class RiskReference(BaseModel):
    file: str
    page: Optional[int] = None
    section: Optional[str] = None


class RiskItem(BaseModel):
    id: int

    title: str
    description: str

    category: str

    reference: RiskReference


class RiskFilter(BaseModel):
    code: str
    label: str
    active: bool = False


class RisksResponse(BaseModel):
    title: str = "Risks & Coordination Items"
    subtitle: str = "Issues flagged from tender analysis"

    total_items: int
    showing: str

    filters: List[RiskFilter]

    items: List[RiskItem]