"""
Quote draft generation service.
"""

import json
import re
from typing import Any, Dict, List

from app.services.openai_service import get_openai_service
from app.services.rag_service import get_rag_service


CSI_DIVISIONS = {
    "00": "Procurement and Contracting Requirements",
    "01": "General Requirements",
    "02": "Existing Conditions",
    "03": "Concrete",
    "04": "Masonry",
    "05": "Metals",
    "06": "Wood, Plastics & Composites",
    "07": "Thermal and Moisture Protection",
    "08": "Openings",
    "09": "Finishes",
    "10": "Specialties",
    "11": "Equipment",
    "12": "Furnishings",
    "13": "Special Construction",
    "14": "Conveying Equipment",
    "21": "Fire Suppression",
    "22": "Plumbing",
    "23": "HVAC",
    "26": "Electrical",
    "27": "Communications",
    "28": "Electronic Safety and Security",
    "31": "Earthwork",
    "32": "Exterior Improvements",
    "33": "Utilities",
}


QUOTE_DRAFT_SYSTEM_PROMPT = """You are an expert construction estimator.

Generate preliminary quote draft information from the provided tender context.

The output is for review only. Use supported tender information, scope items,
pricing impacts, allowances, alternates, bonds, schedule impacts, and selected
CSI divisions.

Rules:
- Use only the provided context.
- Do not invent unsupported costs.
- If exact values are not found, provide a reasonable estimator-style placeholder only when the context supports that cost category.
- Scope of work must be detailed by CSI division and must not include references.
- Assumptions must be plain bullet text and must not include references.
- Exclusions must be plain bullet text and must not include references.
- Keep separate_prices and alternative_prices as separate arrays.
- Separate and alternative price packages must include description, scope_of_work, assumptions, and exclusions.
- Do not return empty arrays for separate_prices, alternative_prices, or unit_prices.
- If explicit separate, alternative, or unit price amounts are not found, still create useful estimator review entries from the analyzed scope and use "Not found" for amount/unit_price.
- When exact prices are absent, explain what should be priced and why it belongs in that section.
- Amounts must be display strings, such as "$485,000", "CAD $548,050", "$15,000", or "$1,200".
- Unit prices must include code, item, description, type, and unit_price.
- Pricing summary must include base bid price, HST, total quoted price, and currency.
- Terms and conditions must include payment terms, holdback, quote validity, and currency.
- Return JSON only. Do not include markdown.

Required schema:
{
  "title": "Quote Draft",
  "scope_of_work": [
    {
      "division_code": "01",
      "division_label": "General Requirements",
      "details": [
        "Provide coordination, supervision, temporary facilities, and project administration required for the work."
      ]
    }
  ],
  "assumptions": [
    "Site access provided during normal working hours.",
    "Utilities are available at no cost to contractor."
  ],
  "exclusions": [
    "Building permits and inspection fees.",
    "Work on statutory holidays unless specifically agreed."
  ],
  "separate_prices": [
    {
      "code": "SP-01",
      "title": "Kitchen Appliances Package",
      "summary": "Procurement and installation of standard residential units.",
      "amount": "$15,000",
      "description": "Provide pricing for the complete package.",
      "scope_of_work": ["Refrigerator", "Dishwasher", "Microwave"],
      "assumptions": ["Existing electrical points available"],
      "exclusions": ["Custom cabinetry changes"]
    }
  ],
  "alternative_prices": [
    {
      "code": "ALT-01",
      "title": "Kitchen Appliances Package",
      "summary": "Alternative appliance procurement and installation package.",
      "amount": "$8,000",
      "description": "Provide alternative pricing for the complete package.",
      "scope_of_work": ["Refrigerator", "Dishwasher", "Microwave"],
      "assumptions": ["Existing electrical points available"],
      "exclusions": ["Smart appliances"]
    }
  ],
  "unit_prices": [
    {
      "code": "UP-01",
      "item": "Additional wooden doors",
      "description": "Additional matching doors.",
      "type": "Per unit",
      "unit_price": "$1,200"
    }
  ],
  "pricing_summary": {
    "base_bid_price": "CAD $485,000",
    "hst": "CAD $63,050",
    "total_quoted_price": "CAD $548,050",
    "currency": "CAD"
  },
  "terms_and_conditions": {
    "payment_terms": "Progress payments monthly based on work completed. Net 30 days from invoice date.",
    "holdback": "10% holdback will be retained as per Construction Act requirements.",
    "quote_validity": "30 days from date of issue.",
    "currency": "All prices in CAD."
  }
}
"""


class QuoteDraftService:
    def __init__(self):
        self.rag_service = get_rag_service()
        self.openai_service = get_openai_service()

    def generate_quote_draft(
        self,
        user_id: str,
        project_id: str,
        divisions: List[str],
        instructions: str,
    ) -> Dict[str, Any]:
        selected_divisions = build_selected_divisions(divisions)

        context = build_quote_draft_context(
            rag_service=self.rag_service,
            user_id=user_id,
            project_id=project_id,
            divisions=divisions,
            instructions=instructions,
        )

        if not context.strip():
            raise ValueError("No relevant uploaded document context found for this project")

        payload = self.openai_service.generate_json(
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Selected CSI divisions:\n{format_selected_divisions(selected_divisions)}\n\n"
                        f"Estimator instructions:\n"
                        f"{instructions.strip() or 'No instructions provided.'}\n\n"
                        f"Retrieved tender context:\n{context}"
                    ),
                }
            ],
            system_prompt=QUOTE_DRAFT_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=4000,
        )

        return normalize_quote_draft_response(
            payload=payload,
            selected_divisions=selected_divisions,
        )


def build_quote_draft_context(
    rag_service: Any,
    user_id: str,
    project_id: str,
    divisions: List[str],
    instructions: str,
) -> str:
    division_text = " ".join(str(division) for division in divisions)

    queries = [
        f"estimated value lump sum total bid amount quote pricing breakdown {division_text}",
        f"division breakdown costs CSI divisions pricing estimate {division_text}",
        f"separate prices separate price cash allowances optional prices {division_text}",
        f"alternative prices alternates alternate prices substitutions options {division_text}",
        f"unit prices unit price schedule per unit per sq.m additional work {division_text}",
        f"pricing impacts allowances alternates bonds night work premium escalation {division_text}",
        f"quote assumptions exclusions terms conditions payment holdback validity currency {division_text}",
        f"scope quantities labor hours duration schedule cost factors {division_text}",
        f"addendum addenda revisions changed scope pricing impact {division_text}",
        instructions,
    ]

    contexts = []
    seen = set()

    for query in queries:
        query = stringify(query)

        if not query:
            continue

        context, _sources = rag_service.retrieve_context(
            query=query,
            user_id=user_id,
            project_id=project_id,
            top_k=10,
        )

        for block in split_context_blocks(context):
            key = normalize_dedupe_key(block)

            if key and key not in seen:
                seen.add(key)
                contexts.append(block)

    return "\n\n".join(contexts)[:26000]


def normalize_quote_draft_response(
    payload: Dict[str, Any],
    selected_divisions: List[Dict[str, str]],
) -> Dict[str, Any]:
    scope_of_work = normalize_quote_scope(
        first_value(payload.get("scope_of_work"), payload.get("scopeOfWork")),
        selected_divisions,
    )
    assumptions = normalize_text_list(payload.get("assumptions")) or default_quote_assumptions()
    exclusions = normalize_text_list(payload.get("exclusions")) or default_quote_exclusions()
    separate_prices = normalize_quote_price_packages(
        first_value(
            payload.get("separate_prices"),
            payload.get("separated_prices"),
            payload.get("separatePrices"),
            payload.get("separatedPrices"),
            payload.get("separate_price"),
            payload.get("separated_price"),
        ),
        prefix="SP",
        fallback_title="Separate Price",
    )
    alternative_prices = normalize_quote_price_packages(
        first_value(
            payload.get("alternative_prices"),
            payload.get("alternativePrices"),
            payload.get("alternatives"),
            payload.get("alternate_prices"),
            payload.get("alternatePrices"),
            payload.get("alternate_price"),
        ),
        prefix="ALT",
        fallback_title="Alternative Price",
    )
    unit_prices = normalize_quote_unit_prices(
        first_value(
            payload.get("unit_prices"),
            payload.get("unitPrices"),
            payload.get("unit_price_items"),
            payload.get("unitPriceItems"),
        )
    )

    if not separate_prices:
        separate_prices = build_fallback_separate_prices(
            scope_of_work=scope_of_work,
            assumptions=assumptions,
            exclusions=exclusions,
        )

    if not alternative_prices:
        alternative_prices = build_fallback_alternative_prices(
            scope_of_work=scope_of_work,
            assumptions=assumptions,
            exclusions=exclusions,
        )

    if not unit_prices:
        unit_prices = build_fallback_unit_prices(scope_of_work)

    return {
        "title": stringify(payload.get("title")) or "Quote Draft",
        "scope_of_work": scope_of_work,
        "assumptions": assumptions,
        "exclusions": exclusions,
        "separate_prices": separate_prices,
        "alternative_prices": alternative_prices,
        "unit_prices": unit_prices,
        "pricing_summary": normalize_quote_pricing_summary(payload),
        "terms_and_conditions": normalize_quote_terms_and_conditions(payload),
    }


def normalize_quote_scope(
    value: Any,
    selected_divisions: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    items = []

    for raw_item in ensure_list(value):
        if not isinstance(raw_item, dict):
            continue

        code = normalize_division_code(
            first_value(
                raw_item.get("division_code"),
                raw_item.get("division"),
                raw_item.get("code"),
            )
        )

        if not code:
            continue

        details = normalize_text_list(
            first_value(
                raw_item.get("details"),
                raw_item.get("items"),
                raw_item.get("scope_of_work"),
                raw_item.get("description"),
                raw_item.get("specifications"),
            )
        )

        items.append(
            {
                "division_code": code,
                "division_label": stringify(
                    first_value(
                        raw_item.get("division_label"),
                        raw_item.get("divisionLabel"),
                        raw_item.get("label"),
                    )
                )
                or CSI_DIVISIONS.get(code, f"Division {code}"),
                "details": details,
            }
        )

    if items:
        return items

    return [
        {
            "division_code": division["code"],
            "division_label": division["label"],
            "details": [
                (
                    f"Provide all labor, materials, equipment, coordination, "
                    f"and related work required for Division {division['code']} "
                    f"- {division['label']} as identified in the uploaded tender documents."
                )
            ],
        }
        for division in selected_divisions
    ]


def normalize_quote_price_packages(
    value: Any,
    prefix: str,
    fallback_title: str,
) -> List[Dict[str, Any]]:
    packages = []

    for index, raw_item in enumerate(ensure_list(value), start=1):
        if not isinstance(raw_item, dict):
            continue

        packages.append(
            {
                "code": stringify(raw_item.get("code")) or f"{prefix}-{index:02d}",
                "title": stringify(
                    first_value(raw_item.get("title"), raw_item.get("name"))
                )
                or f"{fallback_title} {index}",
                "summary": stringify(
                    first_value(raw_item.get("summary"), raw_item.get("subtitle"))
                ),
                "amount": stringify(
                    first_value(
                        raw_item.get("amount"),
                        raw_item.get("price"),
                        raw_item.get("value"),
                    )
                )
                or "Not found",
                "description": stringify(raw_item.get("description")),
                "scope_of_work": normalize_text_list(
                    first_value(
                        raw_item.get("scope_of_work"),
                        raw_item.get("scope"),
                        raw_item.get("items"),
                    )
                ),
                "assumptions": normalize_text_list(raw_item.get("assumptions")),
                "exclusions": normalize_text_list(raw_item.get("exclusions")),
            }
        )

    return packages


def normalize_quote_unit_prices(value: Any) -> List[Dict[str, str]]:
    items = []

    for index, raw_item in enumerate(ensure_list(value), start=1):
        if not isinstance(raw_item, dict):
            continue

        item = stringify(first_value(raw_item.get("item"), raw_item.get("name")))
        unit_type = stringify(first_value(raw_item.get("type"), raw_item.get("unit")))
        unit_price = stringify(
            first_value(
                raw_item.get("unit_price"),
                raw_item.get("unitPrice"),
                raw_item.get("price"),
                raw_item.get("amount"),
            )
        )

        if not item and not unit_price:
            continue

        items.append(
            {
                "code": stringify(raw_item.get("code")) or f"UP-{index:02d}",
                "item": item or f"Unit Price {index}",
                "description": stringify(raw_item.get("description")),
                "type": unit_type or "Per unit",
                "unit_price": unit_price or "Not found",
            }
        )

    return items


def build_fallback_separate_prices(
    scope_of_work: List[Dict[str, Any]],
    assumptions: List[str],
    exclusions: List[str],
) -> List[Dict[str, Any]]:
    scope_details = collect_scope_details(scope_of_work, limit=4)
    title = build_scope_package_title(scope_of_work, "Separate Price Review")

    return [
        {
            "code": "SP-01",
            "title": title,
            "summary": "Estimator review item for separately carried pricing based on the selected scope.",
            "amount": "Not found",
            "description": (
                "No explicit separate price amount was found in the retrieved "
                "context. Carry this item for estimator review where the owner "
                "or tender form requires isolated pricing for part of the scope."
            ),
            "scope_of_work": scope_details,
            "assumptions": assumptions[:4] or default_price_package_assumptions(),
            "exclusions": exclusions[:4] or default_price_package_exclusions(),
        }
    ]


def build_fallback_alternative_prices(
    scope_of_work: List[Dict[str, Any]],
    assumptions: List[str],
    exclusions: List[str],
) -> List[Dict[str, Any]]:
    scope_details = collect_scope_details(scope_of_work, limit=4)
    title = build_scope_package_title(scope_of_work, "Value Engineering Alternative")

    return [
        {
            "code": "ALT-01",
            "title": title,
            "summary": "Potential alternative pricing path derived from the analyzed scope.",
            "amount": "Not found",
            "description": (
                "No explicit alternative price amount was found in the retrieved "
                "context. Use this item to review possible substitutions, product "
                "options, or value-engineering alternatives for the selected scope."
            ),
            "scope_of_work": scope_details,
            "assumptions": assumptions[:4] or default_price_package_assumptions(),
            "exclusions": exclusions[:4] or default_price_package_exclusions(),
        }
    ]


def build_fallback_unit_prices(
    scope_of_work: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    scope_details = collect_scope_detail_records(scope_of_work, limit=3)
    items = []

    for index, detail in enumerate(scope_details, start=1):
        item, unit_type = infer_unit_price_item(detail)
        items.append(
            {
                "code": f"UP-{index:02d}",
                "item": item,
                "description": summarize_detail(detail["detail"]),
                "type": unit_type,
                "unit_price": "Not found",
            }
        )

    if items:
        return items

    return [
        {
            "code": "UP-01",
            "item": "Additional selected-scope work",
            "description": "Unit price placeholder for added or deleted quantities within the selected scope.",
            "type": "Per unit",
            "unit_price": "Not found",
        }
    ]


def normalize_quote_pricing_summary(payload: Dict[str, Any]) -> Dict[str, str]:
    summary = payload.get("pricing_summary")
    if not isinstance(summary, dict):
        summary = {}

    recommended_lump_sum = stringify(payload.get("recommended_lump_sum"))

    return {
        "base_bid_price": stringify(
            first_value(
                summary.get("base_bid_price"),
                summary.get("baseBidPrice"),
                payload.get("base_bid_price"),
                recommended_lump_sum,
            )
        )
        or "Not found",
        "hst": stringify(first_value(summary.get("hst"), payload.get("hst")))
        or "Not found",
        "total_quoted_price": stringify(
            first_value(
                summary.get("total_quoted_price"),
                summary.get("totalQuotedPrice"),
                payload.get("total_quoted_price"),
                recommended_lump_sum,
            )
        )
        or "Not found",
        "currency": stringify(first_value(summary.get("currency"), payload.get("currency")))
        or "CAD",
    }


def normalize_quote_terms_and_conditions(payload: Dict[str, Any]) -> Dict[str, str]:
    terms = payload.get("terms_and_conditions")
    if not isinstance(terms, dict):
        terms = {}

    return {
        "payment_terms": stringify(
            first_value(terms.get("payment_terms"), terms.get("paymentTerms"))
        )
        or "Progress payments monthly based on work completed. Net 30 days from invoice date.",
        "holdback": stringify(terms.get("holdback"))
        or "10% holdback will be retained as per Construction Act requirements.",
        "quote_validity": stringify(
            first_value(terms.get("quote_validity"), terms.get("quoteValidity"))
        )
        or "30 days from date of issue.",
        "currency": stringify(terms.get("currency")) or "All prices in CAD.",
    }


def collect_scope_details(
    scope_of_work: List[Dict[str, Any]],
    limit: int,
) -> List[str]:
    return [
        record["detail"]
        for record in collect_scope_detail_records(scope_of_work, limit)
    ] or ["Selected scope items identified from the uploaded tender documents."]


def collect_scope_detail_records(
    scope_of_work: List[Dict[str, Any]],
    limit: int,
) -> List[Dict[str, str]]:
    records = []

    for division in scope_of_work:
        code = stringify(division.get("division_code"))
        label = stringify(division.get("division_label"))

        for detail in normalize_text_list(division.get("details")):
            records.append(
                {
                    "division_code": code,
                    "division_label": label,
                    "detail": detail,
                }
            )

            if len(records) >= limit:
                return records

    return records


def build_scope_package_title(
    scope_of_work: List[Dict[str, Any]],
    fallback: str,
) -> str:
    for division in scope_of_work:
        label = stringify(division.get("division_label"))
        if label:
            return f"{label} {fallback}"

    return fallback


def infer_unit_price_item(scope_record: Dict[str, str]) -> tuple[str, str]:
    detail = scope_record["detail"].lower()
    label = scope_record["division_label"]

    if "door" in detail:
        return "Additional doors", "Per unit"

    if "window" in detail or "glazing" in detail:
        return "Additional windows/glazing", "Per unit"

    if "paint" in detail or "finish" in detail:
        return "Extra painting/finishes", "Per sq.m"

    if "tile" in detail or "floor" in detail:
        return "Additional flooring/tile work", "Per sq.m"

    if "hardware" in detail:
        return "Additional hardware sets", "Per set"

    if label:
        return f"Additional {label} work", "Per unit"

    return "Additional selected-scope work", "Per unit"


def summarize_detail(value: str, max_length: int = 160) -> str:
    text = stringify(value)

    if len(text) <= max_length:
        return text

    return text[:max_length].rsplit(" ", 1)[0] + "..."


def default_price_package_assumptions() -> List[str]:
    return [
        "Final pricing to be confirmed by estimator based on current tender scope.",
        "Quantities and product selections to be validated before submission.",
        "Work can proceed during normal project working hours unless noted otherwise.",
    ]


def default_price_package_exclusions() -> List[str]:
    return [
        "Unsupported owner upgrades or substitutions.",
        "Premiums not identified in the uploaded tender documents.",
        "Changes issued after the reviewed tender documents.",
    ]


def default_quote_assumptions() -> List[str]:
    return [
        "Pricing is based on the uploaded tender documents and selected CSI divisions.",
        "Final quantities, product selections, and site conditions will be confirmed before submission.",
        "Normal working hours and standard site access are assumed unless the tender documents state otherwise.",
    ]


def default_quote_exclusions() -> List[str]:
    return [
        "Work outside the selected tender scope and reviewed CSI divisions.",
        "Owner-directed changes, unsupported substitutions, and future addenda not included in the uploaded documents.",
        "Permits, utility charges, and authority fees unless specifically included in the tender scope.",
    ]


def normalize_text_list(value: Any) -> List[str]:
    items = []
    seen = set()

    for raw_item in ensure_list(value):
        text = normalize_text_item(raw_item)
        key = text.lower()

        if text and key not in seen:
            items.append(text)
            seen.add(key)

    return items


def normalize_text_item(value: Any) -> str:
    if isinstance(value, dict):
        for key in (
            "text",
            "detail",
            "description",
            "title",
            "item",
            "scope_item",
            "scopeItem",
            "assumption",
            "exclusion",
            "work",
            "name",
            "label",
            "specifications",
        ):
            text = stringify(value.get(key))
            if text:
                return text

        return ""

    return stringify(value)


def first_value(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue

        if isinstance(value, str) and not value.strip():
            continue

        if isinstance(value, (list, dict)) and not value:
            continue

        return value

    return None


def build_selected_divisions(divisions: List[str]) -> List[Dict[str, str]]:
    selected = []
    seen = set()

    for division in divisions:
        code = normalize_division_code(division)

        if not code or code in seen:
            continue

        selected.append(
            {
                "code": code,
                "label": CSI_DIVISIONS.get(code, f"Division {code}"),
            }
        )
        seen.add(code)

    return selected


def format_selected_divisions(selected_divisions: List[Dict[str, str]]) -> str:
    if not selected_divisions:
        return "All relevant divisions found in the retrieved context."

    return "\n".join(
        f"Division {division['code']} - {division['label']}"
        for division in selected_divisions
    )


def split_context_blocks(context: str) -> List[str]:
    pieces = re.split(r"\n(?=\[S\d+\]\s)", stringify(context))
    return [piece.strip() for piece in pieces if piece.strip()]


def normalize_dedupe_key(text: str) -> str:
    return re.sub(r"\W+", "", stringify(text).lower())[:400]


def ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []

    if isinstance(value, list):
        return value

    return [value]


def stringify(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        return re.sub(r"\s+", " ", value).strip()

    if isinstance(value, (dict, list)):
        return re.sub(
            r"\s+",
            " ",
            json.dumps(value, ensure_ascii=False),
        ).strip()

    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_division_code(value: Any) -> str:
    text = str(value or "").strip()

    if not text:
        return ""

    match = re.search(r"\d{1,2}", text)

    if not match:
        return ""

    return match.group(0).zfill(2)


_quote_draft_service = None


def get_quote_draft_service() -> QuoteDraftService:
    global _quote_draft_service

    if _quote_draft_service is None:
        _quote_draft_service = QuoteDraftService()

    return _quote_draft_service
