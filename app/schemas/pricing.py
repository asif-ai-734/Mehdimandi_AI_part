# app/schemas/pricing.py

from typing import List, Optional

from pydantic import BaseModel


class PricingReference(BaseModel):
    file: str
    page: Optional[int] = None
    section: Optional[str] = None


class PricingImpactItem(BaseModel):
    id: int
    title: str
    description: str
    impact: str
    amount: str
    reference: PricingReference


class PricingFilter(BaseModel):
    code: str
    label: str
    active: bool = False


class PricingImpactsResponse(BaseModel):
    title: str = "Pricing Impacts"
    subtitle: str = "Cost factors identified from tender documents"

    total_items: int
    showing: str

    filters: List[PricingFilter]
    items: List[PricingImpactItem]