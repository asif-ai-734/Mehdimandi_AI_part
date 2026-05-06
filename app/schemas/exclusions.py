from typing import List, Optional

from pydantic import BaseModel


class ExclusionReference(BaseModel):
    file: str
    page: Optional[int] = None
    section: Optional[str] = None


class ExclusionItem(BaseModel):
    id: int
    text: str
    reference: ExclusionReference


class ExclusionFilter(BaseModel):
    code: str
    label: str
    active: bool = False


class ExclusionsResponse(BaseModel):
    title: str = "Exclusions"
    subtitle: str = "Items explicitly excluded from scope"

    total_items: int
    showing: str

    filters: List[ExclusionFilter]
    items: List[ExclusionItem]