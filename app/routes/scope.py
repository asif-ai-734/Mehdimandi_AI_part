from pydantic import BaseModel
from typing import List, Optional


# --- Reusable الصغيرة blocks ---

class Quantity(BaseModel):
    value: float
    unit: str   # "units", "sq.m", "sets", etc.


class Reference(BaseModel):
    code: str              # "A601", "Spec Section 08 14 16"
    title: str             # "Door Schedule", "Window Systems"
    page: Optional[int]
    division: Optional[str]


# --- Core Item ---

class ScopeItem(BaseModel):
    id: int
    title: str

    division_code: str     # "01", "06", "08", "09"
    division_label: str    # "Division 06"

    quantity: Quantity
    specifications: str

    references: List[Reference]


# --- Filters (Top Pills) ---

class ScopeFilter(BaseModel):
    code: str              # "all", "01", "06"
    label: str             # "All Scope", "Div 01"
    active: bool = False


# --- Main Response ---

class ScopeOfWorkResponse(BaseModel):
    title: str = "Scope of Work"

    total_items: int
    showing: str           # "7 of 7"

    filters: List[ScopeFilter]

    items: List[ScopeItem]