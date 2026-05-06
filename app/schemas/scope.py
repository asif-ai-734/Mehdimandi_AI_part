from typing import List, Optional

from pydantic import BaseModel


class Quantity(BaseModel):
    value: float
    unit: str


class Reference(BaseModel):
    code: str
    title: str
    page: Optional[int] = None
    division: Optional[str] = None


class ScopeItem(BaseModel):
    id: int
    title: str

    division_code: str
    division_label: str

    quantity: Quantity
    specifications: str

    references: List[Reference]


class ScopeFilter(BaseModel):
    code: str
    label: str
    active: bool = False


class ScopeOfWorkResponse(BaseModel):
    title: str = "Scope of Work"

    total_items: int
    showing: str

    filters: List[ScopeFilter]

    items: List[ScopeItem]