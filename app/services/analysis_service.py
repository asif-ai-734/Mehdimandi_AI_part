"""
Tender analysis service.

Uses the existing project RAG index as context and asks OpenAI for a structured
analysis preview. The full editable report can build on this service later.
"""

import json
import re
from typing import Any, Dict, List

from app.schemas import (
    AnalysisMetrics,
    PricingImpactItem,
    PricingImpactPreview,
    RiskPreview,
    RiskPreviewItem,
    ScopePreview,
    SelectedDivision,
    SummaryPreview,
    TenderAnalysisPreview,
    TenderAnalysisResponse,
)
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


TENDER_ANALYSIS_SYSTEM_PROMPT = """You are an expert construction estimator.
Analyze tender documents using only the provided retrieved project context.
Focus on the selected CSI divisions and estimator instructions.
The context can include original tender documents and addendum files. Addendum
chunks are marked with source type "addendum" in the source label when known.
Summarize the complete project context, including addendum revisions, in the
executive summary.
If estimated value, duration, labor hour count, quantities, or pricing impacts are not explicitly supported by the context, use "Not found" or an empty list.
You may infer complexity and risk_score from documented risks, coordination requirements, schedule constraints, warranties, bonds, liquidated damages, site constraints, and administrative burden.
Do not invent project facts. Keep wording concise and useful for an estimator.
Do not copy placeholder schema values. Fill fields only from retrieved context.
When relevant context exists, write an executive summary from that context instead of returning the default fallback sentence.
Return one JSON object only. Do not include markdown.

Required schema:
{
  "metrics": {
    "estimated_value": "Not found",
    "duration": "Not found",
    "labor_hours": "Not found",
    "complexity": "Not found",
    "risk_score": 0
  },
  "analysis_preview": {
    "executive_summary": {
      "title": "Executive Summary",
      "content": "No supported executive summary found.",
      "badge": "0 key points identified"
    },
    "scope_of_work": {
      "title": "Scope of Work",
      "items": [],
      "badge": "0 items identified"
    },
    "risk_assessment": {
      "title": "Risk Assessment",
      "items": [],
      "badge": "0 risks identified"
    },
    "pricing_impacts": {
      "title": "Pricing Impacts",
      "items": [],
      "badge": "0 cost factors identified"
    },
    "addenda_summary": {
      "title": "Addenda Summary",
      "content": "No supported addendum changes found.",
      "badge": "0 addenda changes identified"
    }
  }
}"""


class AnalysisService:
    """Coordinates RAG retrieval and OpenAI tender analysis."""

    def __init__(self):
        self.rag_service = get_rag_service()
        self.openai_service = get_openai_service()

    def analyze_tender(
        self,
        user_id: str,
        project_id: str,
        divisions: List[str],
        instructions: str,
    ) -> TenderAnalysisResponse:
        """Generate a tender analysis preview from stored project documents."""
        selected_divisions = build_selected_divisions(divisions)
        context, sources, project_name, project_address = build_analysis_context(
            rag_service=self.rag_service,
            user_id=user_id,
            project_id=project_id,
            selected_divisions=selected_divisions,
            instructions=instructions,
        )

        if not context.strip():
            raise ValueError("No relevant uploaded document context found for this project")

        payload = self.openai_service.generate_json(
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Project name: {project_name or 'Not provided'}\n"
                        f"Project address: {project_address or 'Not provided'}\n\n"
                        f"Selected CSI divisions:\n{format_selected_divisions(selected_divisions)}\n\n"
                        f"Estimator instructions:\n{instructions.strip() or 'No additional instructions provided.'}\n\n"
                        f"Retrieved tender document context:\n{context}"
                    ),
                }
            ],
            system_prompt=TENDER_ANALYSIS_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=2200,
        )

        return normalize_tender_analysis_response(
            payload=payload,
            user_id=user_id,
            project_id=project_id,
            project_name=project_name,
            project_address=project_address,
            instructions=instructions,
            selected_divisions=selected_divisions,
            sources=sources,
        )


def build_analysis_queries(selected_divisions: List[SelectedDivision], instructions: str) -> List[str]:
    division_terms = " ".join(
        f"Div {division.code} {division.label}"
        for division in selected_divisions
    )
    base_terms = (
        "scope quantities exclusions alternates allowances warranties bonds insurance "
        "liquidated damages schedule duration labor hours risk pricing impacts "
        "coordination requirements submittals closeout site access temporary facilities "
        "addendum addenda bulletin revision clarification changes"
    )
    queries = [
        " ".join(["Tender estimating analysis", division_terms, instructions, base_terms]).strip(),
        " ".join(["project requirements pricing risks schedule contract conditions", instructions]).strip(),
        " ".join(["addendum addenda revisions bulletins clarifications changed scope pricing schedule", instructions]).strip(),
    ]
    for division in selected_divisions:
        queries.append(
            " ".join(
                [
                    f"Division {division.code}",
                    division.label,
                    "specification section requirements scope submittals products execution",
                    instructions,
                ]
            ).strip()
        )
        if division.code == "01":
            queries.append(
                "General Requirements summary of work allowances alternates unit prices "
                "submittals temporary facilities coordination warranty closeout bonds "
                "insurance liquidated damages project schedule"
            )
    return dedupe_preserve_order(queries)


def build_analysis_context(
    rag_service: Any,
    user_id: str,
    project_id: str,
    selected_divisions: List[SelectedDivision],
    instructions: str,
) -> tuple[str, List[str], str, str]:
    """Retrieve a wider tender-analysis context than normal chat retrieval."""
    blocks = []
    sources = []
    seen_text = set()
    project_name = ""
    project_address = ""

    for query in build_analysis_queries(selected_divisions, instructions):
        context, query_sources = rag_service.retrieve_context(
            query=query,
            user_id=user_id,
            project_id=project_id,
            top_k=8,
        )
        add_context_blocks(blocks, seen_text, context)
        sources.extend(query_sources)

    keyword_candidates = []
    try:
        keyword_candidates = rag_service.qdrant_service.scroll_project_payloads(
            user_id=user_id,
            project_id=project_id,
            limit=750,
        )
    except Exception:
        keyword_candidates = []

    for candidate in keyword_candidates:
        payload = candidate.get("payload", {})
        if not project_name:
            project_name = stringify(payload.get("project_name"))
        if not project_address:
            project_address = stringify(payload.get("project_address"))
        if project_name and project_address:
            break

    keyword_terms = build_keyword_terms(selected_divisions, instructions)
    for candidate in top_keyword_candidates(keyword_candidates, keyword_terms, limit=18):
        payload = candidate.get("payload", {})
        text = stringify(payload.get("text"))
        if not text:
            continue
        source = format_payload_source(payload)
        block = f"{source}\n{text}"
        if add_block(blocks, seen_text, block):
            sources.append(source)

    relabeled_blocks = []
    for index, block in enumerate(blocks, start=1):
        block = re.sub(r"^\[S\d+\]\s*", "", block.strip())
        relabeled_blocks.append(f"[A{index}] {block}")

    return (
        limit_context("\n\n".join(relabeled_blocks), 24000),
        compact_sources(sources),
        project_name,
        project_address,
    )


def build_selected_divisions(divisions: List[str]) -> List[SelectedDivision]:
    codes = []
    seen = set()
    for division in divisions:
        code = normalize_division_code(division)
        if not code or code in seen:
            continue
        codes.append(code)
        seen.add(code)

    allocations = default_allocations(len(codes))
    selected = []
    for index, code in enumerate(codes):
        selected.append(
            SelectedDivision(
                code=code,
                label=CSI_DIVISIONS.get(code, f"Division {code}"),
                allocation_percent=allocations[index],
            )
        )
    return selected


def normalize_division_code(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    match = re.search(r"\d{1,2}", text)
    if not match:
        return ""
    return match.group(0).zfill(2)


def default_allocations(count: int) -> List[int]:
    if count <= 0:
        return []
    base = 100 // count
    allocations = [base] * count
    for index in range(100 - (base * count)):
        allocations[index] += 1
    return allocations


def format_selected_divisions(selected_divisions: List[SelectedDivision]) -> str:
    if not selected_divisions:
        return "All relevant divisions found in the retrieved context."

    return "\n".join(
        f"Div {division.code}: {division.label}"
        for division in selected_divisions
    )


def build_keyword_terms(selected_divisions: List[SelectedDivision], instructions: str) -> List[str]:
    terms = [
        "scope",
        "work",
        "requirements",
        "allowance",
        "allowances",
        "alternate",
        "alternates",
        "unit price",
        "submittal",
        "submittals",
        "warranty",
        "warranties",
        "closeout",
        "temporary",
        "coordination",
        "insurance",
        "bond",
        "bonds",
        "liquidated damages",
        "schedule",
        "completion",
        "site access",
        "cash allowance",
        "addendum",
        "addenda",
        "bulletin",
        "revision",
        "revisions",
        "change",
        "changes",
        "clarification",
        "clarifications",
    ]
    for division in selected_divisions:
        terms.extend(
            [
                division.code,
                f"div {division.code}",
                f"division {division.code}",
                division.label,
            ]
        )
        if division.code == "01":
            terms.extend(
                [
                    "general requirements",
                    "summary of work",
                    "contract requirements",
                    "project requirements",
                    "temporary facilities",
                    "construction facilities",
                    "regulatory requirements",
                ]
            )
    terms.extend(token.lower() for token in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9./-]{2,}", instructions))
    return dedupe_preserve_order([term.lower() for term in terms if stringify(term)])


def top_keyword_candidates(
    candidates: List[Dict[str, Any]],
    terms: List[str],
    limit: int,
) -> List[Dict[str, Any]]:
    scored = []
    for candidate in candidates:
        payload = candidate.get("payload", {})
        haystack = " ".join(
            [
                stringify(payload.get("text")),
                stringify(payload.get("filename")),
                stringify(payload.get("source_text_ref")),
                stringify(payload.get("source_type")),
            ]
        ).lower()
        if not haystack:
            continue

        score = 0
        if stringify(payload.get("source_type")).lower() == "addendum":
            score += 8
        for term in terms:
            if term and term in haystack:
                score += 4 if " " in term or term.isdigit() else 1
        if score > 0:
            scored.append((score, candidate))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [candidate for _, candidate in scored[:limit]]


def add_context_blocks(blocks: List[str], seen_text: set[str], context: str) -> None:
    for block in split_context_blocks(context):
        add_block(blocks, seen_text, block)


def split_context_blocks(context: str) -> List[str]:
    pieces = re.split(r"\n(?=\[S\d+\]\s)", context.strip())
    return [piece.strip() for piece in pieces if piece.strip()]


def add_block(blocks: List[str], seen_text: set[str], block: str) -> bool:
    text = block.split("\n", 1)[1] if "\n" in block else block
    key = normalize_dedupe_key(text)
    if not key or key in seen_text:
        return False
    seen_text.add(key)
    blocks.append(block)
    return True


def normalize_dedupe_key(text: str) -> str:
    return re.sub(r"\W+", "", text.lower())[:400]


def format_payload_source(payload: Dict[str, Any]) -> str:
    parts = [stringify(payload.get("filename")) or "unknown"]
    if payload.get("source_type"):
        parts.append(stringify(payload.get("source_type")))
    if payload.get("project_name"):
        parts.append(f"project {stringify(payload.get('project_name'))}")
    if payload.get("project_address"):
        parts.append(f"address {stringify(payload.get('project_address'))}")
    if payload.get("chunk_index") is not None:
        parts.append(f"chunk {payload['chunk_index']}")
    if payload.get("chunk_type"):
        parts.append(stringify(payload.get("chunk_type")))
    return " | ".join(parts)


def limit_context(context: str, max_chars: int) -> str:
    if len(context) <= max_chars:
        return context
    return context[:max_chars].rsplit("\n\n", 1)[0] + "\n\n[context truncated]"


def dedupe_preserve_order(values: List[str]) -> List[str]:
    deduped = []
    seen = set()
    for value in values:
        text = stringify(value)
        key = text.lower()
        if text and key not in seen:
            deduped.append(text)
            seen.add(key)
    return deduped


def normalize_tender_analysis_response(
    payload: Dict[str, Any],
    user_id: str,
    project_id: str,
    project_name: str,
    project_address: str,
    instructions: str,
    selected_divisions: List[SelectedDivision],
    sources: List[str],
) -> TenderAnalysisResponse:
    metrics = normalize_metrics(payload.get("metrics", {}))
    preview = normalize_preview(payload.get("analysis_preview", {}))
    selected = merge_division_allocations(
        selected_divisions,
        ensure_list(payload.get("selected_divisions")),
    )
    return TenderAnalysisResponse(
        user_id=user_id,
        project_id=project_id,
        project_name=project_name,
        project_address=project_address,
        status="preview",
        instructions=instructions.strip(),
        selected_divisions=selected,
        metrics=metrics,
        analysis_preview=preview,
        sources=compact_sources(sources),
    )


def merge_division_allocations(
    selected_divisions: List[SelectedDivision],
    payload_divisions: List[Any],
) -> List[SelectedDivision]:
    allocation_by_code = {}
    for item in payload_divisions:
        if not isinstance(item, dict):
            continue
        code = normalize_division_code(item.get("code"))
        if code:
            allocation_by_code[code] = clamp_int(item.get("allocation_percent"), 0, 100)

    merged = []
    for division in selected_divisions:
        merged.append(
            SelectedDivision(
                code=division.code,
                label=division.label,
                allocation_percent=allocation_by_code.get(
                    division.code,
                    division.allocation_percent,
                ),
            )
        )
    return merged


def normalize_metrics(value: Any) -> AnalysisMetrics:
    data = value if isinstance(value, dict) else {}
    return AnalysisMetrics(
        estimated_value=clean_placeholder(data.get("estimated_value")) or "Not found",
        duration=clean_placeholder(data.get("duration")) or "Not found",
        labor_hours=clean_placeholder(data.get("labor_hours")) or "Not found",
        complexity=clean_placeholder(data.get("complexity")) or "Not found",
        risk_score=clamp_int(data.get("risk_score"), 0, 100),
    )


def normalize_preview(value: Any) -> TenderAnalysisPreview:
    data = value if isinstance(value, dict) else {}
    return TenderAnalysisPreview(
        executive_summary=normalize_summary_card(
            data.get("executive_summary"),
            "Executive Summary",
            "No executive summary found in the retrieved documents.",
            "0 key points identified",
        ),
        scope_of_work=normalize_scope_card(data.get("scope_of_work")),
        risk_assessment=normalize_risk_card(data.get("risk_assessment")),
        pricing_impacts=normalize_pricing_card(data.get("pricing_impacts")),
        addenda_summary=normalize_summary_card(
            data.get("addenda_summary"),
            "Addenda Summary",
            "No addendum changes found in the retrieved documents.",
            "0 addenda changes identified",
        ),
    )


def normalize_summary_card(
    value: Any,
    default_title: str,
    default_content: str,
    default_badge: str,
) -> SummaryPreview:
    data = value if isinstance(value, dict) else {}
    return SummaryPreview(
        title=stringify(data.get("title")) or default_title,
        content=clean_placeholder(data.get("content")) or default_content,
        badge=stringify(data.get("badge")) or default_badge,
    )


def normalize_scope_card(value: Any) -> ScopePreview:
    data = value if isinstance(value, dict) else {}
    items = [stringify(item) for item in ensure_list(data.get("items"))]
    items = [item for item in items if item]
    return ScopePreview(
        title=stringify(data.get("title")) or "Scope of Work",
        items=items,
        badge=normalize_count_badge(
            data.get("badge"),
            len(items),
            "items identified",
        ),
    )


def normalize_risk_card(value: Any) -> RiskPreview:
    data = value if isinstance(value, dict) else {}
    items = []
    for item in ensure_list(data.get("items")):
        if isinstance(item, dict):
            label = stringify(item.get("label"))
            severity = stringify(item.get("severity")) or "Medium"
        else:
            label = stringify(item)
            severity = "Medium"
        if label:
            items.append(RiskPreviewItem(label=label, severity=severity))
    return RiskPreview(
        title=stringify(data.get("title")) or "Risk Assessment",
        items=items,
        badge=normalize_count_badge(
            data.get("badge"),
            len(items),
            "risks identified",
        ),
    )


def normalize_pricing_card(value: Any) -> PricingImpactPreview:
    data = value if isinstance(value, dict) else {}
    items = []
    for item in ensure_list(data.get("items")):
        if isinstance(item, dict):
            label = stringify(item.get("label"))
            item_value = stringify(item.get("value")) or "Not found"
        else:
            label = stringify(item)
            item_value = "Not found"
        if label:
            items.append(PricingImpactItem(label=label, value=item_value))
    return PricingImpactPreview(
        title=stringify(data.get("title")) or "Pricing Impacts",
        items=items,
        badge=normalize_count_badge(
            data.get("badge"),
            len(items),
            "cost factors identified",
        ),
    )


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
        return re.sub(r"\s+", " ", json.dumps(value, ensure_ascii=False)).strip()
    return re.sub(r"\s+", " ", str(value)).strip()


def clean_placeholder(value: Any) -> str:
    text = stringify(value)
    if text in {"...", "-", "--"}:
        return ""
    return text


def normalize_count_badge(value: Any, count: int, noun: str) -> str:
    badge = stringify(value)
    if count and re.match(r"^0\b", badge):
        return f"{count} {noun}"
    return badge or f"{count} {noun}"


def compact_sources(sources: List[str]) -> List[str]:
    compacted = []
    seen = set()
    for source in sources:
        text = stringify(source)
        if not text:
            continue
        text = re.sub(r"^S\d+:\s*", "", text)
        filename = text.split("|", 1)[0].strip()
        key = filename.lower()
        if filename and key not in seen:
            compacted.append(filename)
            seen.add(key)
    return compacted


def clamp_int(value: Any, minimum: int, maximum: int) -> int:
    try:
        integer = int(value)
    except (TypeError, ValueError):
        integer = minimum
    return max(minimum, min(maximum, integer))


_analysis_service = None


def get_analysis_service() -> AnalysisService:
    """Get or create the global analysis service instance."""
    global _analysis_service
    if _analysis_service is None:
        _analysis_service = AnalysisService()
    return _analysis_service
