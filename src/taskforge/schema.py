"""A small JSON-Schema validator covering the subset tool definitions actually use.

Why not depend on ``jsonschema``? Tool argument schemas are tiny and we want the
engine to stay standard-library only. More importantly, the error messages matter:
when a model produces bad arguments the validator's message is fed *back to the
model* as the tool result, so it must point at the exact path that is wrong
(``$.path: expected string, got integer``) instead of dumping a stack trace.

Supported keywords: type (incl. lists of types), enum, const, properties,
required, additionalProperties (bool or schema), items, minItems, maxItems,
minLength, maxLength, minimum, maximum, pattern.
"""

from __future__ import annotations

import re
from typing import Any

__all__ = ["SchemaError", "validate"]


class SchemaError(ValueError):
    """Raised when an instance does not match its schema. ``path`` is a JSONPath-ish locator."""

    def __init__(self, path: str, message: str) -> None:
        super().__init__(f"{path}: {message}")
        self.path = path


def _type_name(value: Any) -> str:
    # Order matters: bool is a subclass of int in Python, but JSON treats them as
    # disjoint types. A model passing `true` for an integer field must be rejected.
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _matches_type(value: Any, expected: str) -> bool:
    actual = _type_name(value)
    if expected == "number":
        # JSON "number" includes integers; "integer" does not include 3.5.
        return actual in ("number", "integer")
    if expected == "integer" and actual == "number":
        return float(value).is_integer()
    return actual == expected


def validate(instance: Any, schema: dict[str, Any], path: str = "$") -> None:
    """Validate ``instance`` against ``schema``; raise :class:`SchemaError` on the first problem."""
    if not schema:
        return

    expected = schema.get("type")
    if expected is not None:
        options = expected if isinstance(expected, list) else [expected]
        if not any(_matches_type(instance, t) for t in options):
            raise SchemaError(path, f"expected {' or '.join(options)}, got {_type_name(instance)}")

    if "const" in schema and instance != schema["const"]:
        raise SchemaError(path, f"expected constant {schema['const']!r}")

    if "enum" in schema and instance not in schema["enum"]:
        raise SchemaError(path, f"expected one of {schema['enum']!r}, got {instance!r}")

    if isinstance(instance, str):
        if "minLength" in schema and len(instance) < schema["minLength"]:
            raise SchemaError(path, f"string shorter than {schema['minLength']}")
        if "maxLength" in schema and len(instance) > schema["maxLength"]:
            raise SchemaError(path, f"string longer than {schema['maxLength']}")
        if "pattern" in schema and not re.search(schema["pattern"], instance):
            raise SchemaError(path, f"string does not match pattern {schema['pattern']!r}")

    if _type_name(instance) in ("integer", "number"):
        if "minimum" in schema and instance < schema["minimum"]:
            raise SchemaError(path, f"value {instance} is below minimum {schema['minimum']}")
        if "maximum" in schema and instance > schema["maximum"]:
            raise SchemaError(path, f"value {instance} is above maximum {schema['maximum']}")

    if isinstance(instance, list):
        if "minItems" in schema and len(instance) < schema["minItems"]:
            raise SchemaError(path, f"array has fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            raise SchemaError(path, f"array has more than {schema['maxItems']} items")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for i, item in enumerate(instance):
                validate(item, item_schema, f"{path}[{i}]")

    if isinstance(instance, dict):
        props: dict[str, Any] = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in instance:
                raise SchemaError(path, f"missing required property {key!r}")
        extra = schema.get("additionalProperties", True)
        for key, value in instance.items():
            if key in props:
                validate(value, props[key], f"{path}.{key}")
            elif extra is False:
                raise SchemaError(path, f"unexpected property {key!r}")
            elif isinstance(extra, dict):
                validate(value, extra, f"{path}.{key}")
