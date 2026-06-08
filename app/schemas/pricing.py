# app/schemas/pricing.py

import re
from typing import Any, List, Optional, Union

from pydantic import BaseModel, model_validator


Number = Union[int, float]


FIXED_PRICING_COST_CATEGORIES = [
    {
        "name": "Bonds",
        "keywords": (
            "bond",
            "bonds",
            "bid bond",
            "performance bond",
            "payment bond",
            "surety",
        ),
        "default_description": "Bond and surety requirements.",
        "default_reasoning": "Based on bond and surety requirements found in the tender documents.",
    },
    {
        "name": "Insurance",
        "keywords": (
            "insurance",
            "insured",
            "liability",
            "builder",
            "builders risk",
            "coverage",
            "policy",
        ),
        "default_description": "Insurance requirements and related premiums.",
        "default_reasoning": "Based on insurance and coverage requirements found in the tender documents.",
    },
    {
        "name": "Coordination",
        "keywords": (
            "coordination",
            "coordinate",
            "meeting",
            "meetings",
            "scheduling",
            "schedule",
            "rfi",
            "rfis",
            "submittal",
            "submittals",
            "logistics",
            "phasing",
            "site access",
            "administration",
        ),
        "default_description": "Coordination, scheduling, RFIs, and project administration.",
        "default_reasoning": "Based on coordination, scheduling, and administration requirements found in the tender documents.",
    },
    {
        "name": "Contingency",
        "keywords": (
            "contingency",
            "risk buffer",
            "risk allowance",
            "allowance",
            "reserve",
            "uncertainty",
            "escalation",
        ),
        "default_description": "Contingency allowance for pricing uncertainty and project risk.",
        "default_reasoning": "Based on pricing uncertainty, risk allowances, and contingency factors found in the tender documents.",
    },
]


class PricingComparison(BaseModel):
    aiDraftEstimate: Optional[Number] = None
    estimatorFinalPrice: Optional[Number] = None
    variance: Optional[Number] = None

    @model_validator(mode="before")
    @classmethod
    def accept_snake_case_aliases(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value

        data = dict(value)
        data["aiDraftEstimate"] = parse_amount(
            data.get("aiDraftEstimate", data.get("ai_draft_estimate"))
        )
        data["estimatorFinalPrice"] = parse_amount(
            data.get("estimatorFinalPrice", data.get("estimator_final_price"))
        )

        variance = data.get("variance")
        if variance is None:
            estimate = data["aiDraftEstimate"]
            final = data["estimatorFinalPrice"]
            if estimate is not None and final is not None:
                variance = final - estimate

        data["variance"] = parse_amount(variance)

        return data


class PricingEstimateBreakdownItem(BaseModel):
    division: str
    name: str
    amount: Optional[Number] = None
    editable: bool = True

    @model_validator(mode="before")
    @classmethod
    def accept_aliases(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value

        data = dict(value)
        division = (
            data.get("division")
            or data.get("division_code")
            or data.get("divisionCode")
        )
        data["division"] = normalize_division_code(division) or stringify(division)
        data["name"] = (
            stringify(data.get("name"))
            or stringify(data.get("division_label"))
            or stringify(data.get("divisionLabel"))
            or (f"Division {data['division']}" if data["division"] else "Unassigned")
        )
        data["amount"] = parse_amount(data.get("amount"))
        data["editable"] = parse_bool(data.get("editable"), default=True)

        return data


class PricingAdditionalCostItem(BaseModel):
    name: str
    description: str
    amount: Optional[Number] = None
    editable: bool = True

    @model_validator(mode="before")
    @classmethod
    def accept_aliases(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value

        data = dict(value)
        name = stringify(data.get("name")) or stringify(data.get("title"))
        description = stringify(data.get("description"))
        impact = stringify(data.get("impact"))

        if not description:
            description = impact or "No description found."
        elif impact and impact not in description:
            description = f"{description} {impact}"

        data["name"] = name or "Untitled Cost Item"
        data["description"] = description
        data["amount"] = parse_amount(data.get("amount"))
        data["editable"] = parse_bool(data.get("editable"), default=True)

        return data


class PricingMissingInformationItem(BaseModel):
    title: str
    description: str
    severity: str

    @model_validator(mode="before")
    @classmethod
    def normalize_item(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value

        data = dict(value)
        data["title"] = stringify(data.get("title")) or "Missing information"
        data["description"] = (
            stringify(data.get("description")) or "Pricing information is not confirmed."
        )
        data["severity"] = normalize_severity(data.get("severity"))

        return data


class PricingBasisReasoningItem(BaseModel):
    title: str
    description: str

    @model_validator(mode="before")
    @classmethod
    def normalize_item(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value

        data = dict(value)
        data["title"] = stringify(data.get("title")) or "Pricing Basis"
        data["description"] = (
            stringify(data.get("description"))
            or "Based on pricing information extracted from tender documents."
        )

        return data


class PricingImpactsResponse(BaseModel):
    comparison: PricingComparison
    aiDraftEstimateBreakdown: List[PricingEstimateBreakdownItem]
    additionalCostItems: List[PricingAdditionalCostItem]
    missingInformation: List[PricingMissingInformationItem]
    pricingBasisAndReasoning: List[PricingBasisReasoningItem]

    @model_validator(mode="before")
    @classmethod
    def accept_legacy_pricing_shapes(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value

        data = dict(value)
        comparison = data.get("comparison")
        if not isinstance(comparison, dict):
            comparison = {}

        if "aiDraftEstimate" not in comparison and "ai_draft_estimate" in data:
            comparison["aiDraftEstimate"] = data.get("ai_draft_estimate")

        if "aiDraftEstimate" not in comparison:
            comparison["aiDraftEstimate"] = data.get("aiDraftEstimate")

        if "estimatorFinalPrice" not in comparison:
            comparison["estimatorFinalPrice"] = data.get("estimatorFinalPrice")

        if "variance" not in comparison:
            comparison["variance"] = data.get("variance")

        breakdown = first_list(
            data.get("aiDraftEstimateBreakdown"),
            data.get("ai_draft_estimate_breakdown"),
            data.get("division_breakdown"),
        )
        raw_additional_costs = first_list(
            data.get("additionalCostItems"),
            data.get("additional_cost_items"),
            data.get("additional_costs"),
            data.get("items"),
        )
        missing_info = first_list(
            data.get("missingInformation"),
            data.get("missing_information"),
        )
        raw_basis = first_list(
            data.get("pricingBasisAndReasoning"),
            data.get("pricing_basis_and_reasoning"),
            data.get("pricing_basis"),
        )
        additional_costs = normalize_fixed_additional_cost_items(raw_additional_costs)
        basis = normalize_fixed_pricing_basis(raw_basis, additional_costs)

        return {
            "comparison": comparison,
            "aiDraftEstimateBreakdown": breakdown,
            "additionalCostItems": additional_costs,
            "missingInformation": missing_info,
            "pricingBasisAndReasoning": basis,
        }


def first_list(*values: Any) -> List[Any]:
    for value in values:
        if isinstance(value, list):
            return value

    return []


def normalize_fixed_additional_cost_items(values: List[Any]) -> List[dict[str, Any]]:
    grouped = {
        category["name"]: {
            "description_parts": [],
            "amounts": [],
        }
        for category in FIXED_PRICING_COST_CATEGORIES
    }

    for value in values:
        if not isinstance(value, dict):
            continue

        category = match_fixed_pricing_category(value)
        if not category:
            continue

        description = stringify(value.get("description"))
        impact = stringify(value.get("impact"))
        if not description:
            description = impact
        elif impact and impact not in description:
            description = f"{description} {impact}"

        if description:
            grouped[category]["description_parts"].append(description)

        amount = parse_amount(value.get("amount"))
        if amount is not None:
            grouped[category]["amounts"].append(amount)

    items = []
    for category in FIXED_PRICING_COST_CATEGORIES:
        name = category["name"]
        description = join_unique(grouped[name]["description_parts"])
        amounts = grouped[name]["amounts"]

        items.append(
            {
                "name": name,
                "description": description or category["default_description"],
                "amount": sum(amounts) if amounts else None,
                "editable": True,
            }
        )

    return items


def normalize_fixed_pricing_basis(
    values: List[Any],
    additional_costs: List[dict[str, Any]],
) -> List[dict[str, str]]:
    descriptions = {
        category["name"]: []
        for category in FIXED_PRICING_COST_CATEGORIES
    }

    for value in values:
        if not isinstance(value, dict):
            continue

        category = match_fixed_pricing_category(value)
        if not category:
            continue

        description = stringify(value.get("description"))
        if description:
            descriptions[category].append(description)

    cost_descriptions = {
        item["name"]: item["description"]
        for item in additional_costs
        if stringify(item.get("description"))
    }

    items = []
    for category in FIXED_PRICING_COST_CATEGORIES:
        name = category["name"]
        description = join_unique(descriptions[name])

        if not description:
            description = cost_descriptions.get(name, "")
            if description == category["default_description"]:
                description = ""

        items.append(
            {
                "title": name,
                "description": description or category["default_reasoning"],
            }
        )

    return items


def match_fixed_pricing_category(value: dict[str, Any]) -> str:
    explicit_values = [
        stringify(value.get("name")),
        stringify(value.get("title")),
        stringify(value.get("category")),
    ]

    for explicit_value in explicit_values:
        text = explicit_value.lower()
        if not text:
            continue

        for category in FIXED_PRICING_COST_CATEGORIES:
            category_name = category["name"].lower()
            if text == category_name or text.startswith(category_name):
                return category["name"]

    text = " ".join(
        [
            *explicit_values,
            stringify(value.get("description")),
            stringify(value.get("impact")),
        ]
    ).lower()

    if not text:
        return ""

    for category in FIXED_PRICING_COST_CATEGORIES:
        if any(keyword in text for keyword in category["keywords"]):
            return category["name"]

    return ""


def join_unique(values: List[str]) -> str:
    parts = []
    seen = set()

    for value in values:
        text = stringify(value)
        key = text.lower()

        if text and key not in seen:
            parts.append(text)
            seen.add(key)

    return " ".join(parts)


def normalize_division_code(value: Any) -> str:
    match = re.search(r"\d{1,2}", str(value or ""))

    if not match:
        return ""

    return match.group(0).zfill(2)


def normalize_severity(value: Any) -> str:
    text = stringify(value).lower()

    if text in {"critical", "high"}:
        return "critical"

    if text in {"medium", "warning"}:
        return "warning"

    if text in {"low", "info", "informational"}:
        return "info"

    return "warning"


def parse_amount(value: Any) -> Optional[Number]:
    if value is None:
        return None

    if isinstance(value, bool):
        return None

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        return int(value) if value.is_integer() else value

    text = stringify(value)

    if not text or text.lower() in {"none", "null", "n/a", "not applicable", "not found"}:
        return None

    negative = text.startswith("(") and text.endswith(")")
    match = re.search(r"-?\d[\d,]*(\.\d+)?", text)

    if not match:
        return None

    numeric_text = match.group(0).replace(",", "")
    amount = float(numeric_text)

    if negative:
        amount = -amount

    return int(amount) if amount.is_integer() else amount


def parse_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value

    text = stringify(value).lower()

    if text in {"true", "1", "yes", "y"}:
        return True

    if text in {"false", "0", "no", "n"}:
        return False

    return default


def stringify(value: Any) -> str:
    if value is None:
        return ""

    return re.sub(r"\s+", " ", str(value)).strip()
