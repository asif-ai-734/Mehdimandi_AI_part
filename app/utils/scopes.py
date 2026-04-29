"""
Helpers for user/project scope values.
"""


def normalize_scope_value(value: object) -> str:
    """Normalize user_id/project_id inputs to stable string scope values."""
    return str(value or "").strip()


def is_valid_scope_value(value: object) -> bool:
    """Return True when a scope value is usable."""
    return bool(normalize_scope_value(value))
