from typing import List

from pydantic import BaseModel, Field


class QuoteScopeDivisionItem(BaseModel):
    division_code: str
    division_label: str
    details: List[str] = Field(default_factory=list)


class QuotePricePackage(BaseModel):
    code: str
    title: str
    summary: str = ""
    amount: str
    description: str = ""
    scope_of_work: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    exclusions: List[str] = Field(default_factory=list)


class QuoteUnitPriceItem(BaseModel):
    code: str
    item: str
    description: str = ""
    type: str
    unit_price: str


class QuotePricingSummary(BaseModel):
    base_bid_price: str
    hst: str
    total_quoted_price: str
    currency: str = "CAD"


class QuoteTermsAndConditions(BaseModel):
    payment_terms: str
    holdback: str
    quote_validity: str
    currency: str


class QuoteDraftResponse(BaseModel):
    title: str = "Quote Draft"
    scope_of_work: List[QuoteScopeDivisionItem] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    exclusions: List[str] = Field(default_factory=list)
    separate_prices: List[QuotePricePackage] = Field(default_factory=list)
    alternative_prices: List[QuotePricePackage] = Field(default_factory=list)
    unit_prices: List[QuoteUnitPriceItem] = Field(default_factory=list)
    pricing_summary: QuotePricingSummary
    terms_and_conditions: QuoteTermsAndConditions
