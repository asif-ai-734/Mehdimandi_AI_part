from typing import Any, List, Optional, Union

from pydantic import BaseModel, Field, model_validator


class Quantity(BaseModel):
    value: Union[int, float] = 0
    unit: str = "unspecified"


class ScopeSource(BaseModel):
    document: str = "Not found"
    page: Optional[int] = None


class ScopeActions(BaseModel):
    canEdit: bool = True
    canDuplicate: bool = True
    canDelete: bool = True

    @model_validator(mode="before")
    @classmethod
    def accept_snake_case_aliases(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value

        data = dict(value)
        aliases = {
            "canEdit": "can_edit",
            "canDuplicate": "can_duplicate",
            "canDelete": "can_delete",
        }

        for field, alias in aliases.items():
            if field not in data and alias in data:
                data[field] = data[alias]

        return data


class ScopeItem(BaseModel):
    division: str
    scopeItem: str
    quantity: Quantity
    include: bool = True
    notes: str = ""
    source: ScopeSource = Field(default_factory=ScopeSource)
    actions: ScopeActions = Field(default_factory=ScopeActions)

    @model_validator(mode="before")
    @classmethod
    def accept_legacy_scope_shape(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value

        data = dict(value)
        references = data.get("references")
        first_reference = {}

        if isinstance(references, list):
            for reference in references:
                if isinstance(reference, dict):
                    first_reference = reference
                    break

        if "division" not in data:
            data["division"] = data.get("division_code") or data.get("divisionCode")
        if not data.get("division"):
            data["division"] = "00"

        if "scopeItem" not in data:
            data["scopeItem"] = data.get("scope_item") or data.get("title")
        if not data.get("scopeItem"):
            data["scopeItem"] = "Untitled Scope Item"

        if "notes" not in data:
            data["notes"] = data.get("specifications") or ""

        if "include" not in data:
            data["include"] = True

        if not isinstance(data.get("quantity"), dict):
            data["quantity"] = {"value": 0, "unit": "unspecified"}

        source = data.get("source") if isinstance(data.get("source"), dict) else {}
        source = dict(source)

        if not source.get("document"):
            source["document"] = (
                data.get("document")
                or first_reference.get("document")
                or first_reference.get("title")
                or first_reference.get("code")
                or "Not found"
            )

        if "page" not in source:
            source["page"] = first_reference.get("page")

        data["source"] = source

        if not isinstance(data.get("actions"), dict):
            data["actions"] = {}

        return data


class ScopeFilter(BaseModel):
    code: str
    label: str
    active: bool = False


class ScopeOfWorkResponse(BaseModel):
    title: str = "Scope of Work"

    total_items: int
    showing: str

    filters: List[ScopeFilter]

    items: List[ScopeItem]
