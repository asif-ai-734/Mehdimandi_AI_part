from typing import List, Optional

from pydantic import BaseModel


class AssumptionReference(BaseModel):
    file: str
    page: Optional[int] = None
    section: Optional[str] = None


class AssumptionItem(BaseModel):
    id: int
    text: str
    reference: AssumptionReference


class AssumptionFilter(BaseModel):
    code: str
    label: str
    active: bool = False


class AssumptionsResponse(BaseModel):
    title: str = "Assumptions"
    subtitle: str = "Assumptions made based on tender documents"

    total_items: int
    showing: str

    filters: List[AssumptionFilter]
    items: List[AssumptionItem]