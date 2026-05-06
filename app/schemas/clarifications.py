from typing import List, Optional

from pydantic import BaseModel


class ClarificationReference(BaseModel):
    file: str
    page: Optional[int] = None
    section: Optional[str] = None


class ClarificationItem(BaseModel):
    id: int
    question: str
    reference: ClarificationReference


class ClarificationFilter(BaseModel):
    code: str
    label: str
    active: bool = False


class ClarificationsResponse(BaseModel):
    title: str = "Clarifications Needed"
    subtitle: str = "Items requiring clarification from owner/architect"

    total_items: int
    showing: str

    filters: List[ClarificationFilter]
    items: List[ClarificationItem]