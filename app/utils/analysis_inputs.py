"""
Helpers for normalizing tender analysis inputs from JSON, query strings, or
multipart form fields.
"""

import json
import re
from typing import Any, List


def normalize_divisions(value: Any) -> List[str]:
    """Return a clean division list from repeated fields, JSON, or CSV numbers."""
    values = value if isinstance(value, list) else [value]
    divisions = []
    seen = set()

    for item in values:
        for text in _expand_division_item(item):
            key = text.lower()
            if text and key not in seen:
                divisions.append(text)
                seen.add(key)

    return divisions


def divisions_to_json(value: Any) -> str:
    """Serialize normalized divisions for SQLite storage."""
    return json.dumps(normalize_divisions(value))


def has_valid_division_code(value: Any) -> bool:
    """Return True when at least one division-like value contains a CSI code."""
    return any(
        re.search(r"\d{1,2}", str(division or ""))
        for division in normalize_divisions(value)
    )


def _expand_division_item(value: Any) -> List[str]:
    if value is None:
        return []

    if isinstance(value, (list, tuple, set)):
        expanded = []
        for item in value:
            expanded.extend(_expand_division_item(item))
        return expanded

    text = str(value).strip()
    if not text:
        return []

    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if parsed is not None:
            return _expand_division_item(parsed)

    if re.fullmatch(r"[\d,\s]+", text) and "," in text:
        return [part.strip() for part in text.split(",") if part.strip()]

    return [text]
