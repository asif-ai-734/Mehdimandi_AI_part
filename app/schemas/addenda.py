from typing import List, Optional

from pydantic import BaseModel


class AddendaReference(BaseModel):
    file: str
    page: Optional[int] = None
    item: Optional[str] = None


class AddendaItem(BaseModel):
    id: int

    addendum_number: str
    issued_date: Optional[str] = None

    title: str
    description: str

    impact_type: str

    affected_divisions: List[str]

    scope_change: str
    pricing_impact: str

    reference: AddendaReference


class AddendaFilter(BaseModel):
    code: str
    label: str
    active: bool = False


class AddendaResponse(BaseModel):
    title: str = "Addenda Changes"
    subtitle: str = "Changes issued through addenda"

    total_items: int
    showing: str

    filters: List[AddendaFilter]

    items: List[AddendaItem]