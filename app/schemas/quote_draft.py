from typing import List

from pydantic import BaseModel


class QuoteDivisionBreakdownItem(BaseModel):
    division_code: str
    division_label: str
    amount: str


class QuoteAdditionalCostItem(BaseModel):
    title: str
    amount: str


class QuoteDraftResponse(BaseModel):
    title: str = "Suggested Quote Information"
    subtitle: str = "Preliminary quote data for review"

    recommended_lump_sum: str
    lump_sum_note: str = "Based on scope analysis and pricing factors"

    division_breakdown: List[QuoteDivisionBreakdownItem]
    additional_costs: List[QuoteAdditionalCostItem]

    build_quote_label: str = "Open Build Quote"