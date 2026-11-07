import pytest

from taskforge.schema import SchemaError, validate

SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "minLength": 1, "pattern": r"^[\w/.]+$"},
        "count": {"type": "integer", "minimum": 0, "maximum": 10},
        "ratio": {"type": "number"},
        "mode": {"enum": ["a", "b"]},
        "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 2},
    },
    "required": ["path"],
    "additionalProperties": False,
}


def test_valid_instance_passes():
    validate({"path": "a/b.txt", "count": 3, "ratio": 1, "mode": "a", "tags": ["x"]}, SCHEMA)


@pytest.mark.parametrize("instance, fragment", [
    ({}, "missing required property 'path'"),
    ({"path": "x", "count": True}, "$.count: expected integer, got boolean"),
    ({"path": "x", "count": 11}, "above maximum"),
    ({"path": "x", "ratio": "1"}, "$.ratio: expected number"),
    ({"path": "x", "mode": "c"}, "expected one of"),
    ({"path": "x", "tags": ["a", 1]}, "$.tags[1]: expected string"),
    ({"path": "x", "tags": ["a", "b", "c"]}, "more than 2 items"),
    ({"path": "x", "extra": 1}, "unexpected property 'extra'"),
    ({"path": "bad path!"}, "does not match pattern"),
    ({"path": ""}, "shorter than 1"),
])
def test_errors_point_at_the_offending_path(instance, fragment):
    with pytest.raises(SchemaError) as err:
        validate(instance, SCHEMA)
    assert fragment in str(err.value)


def test_integer_valued_float_counts_as_integer_and_int_as_number():
    validate(3.0, {"type": "integer"})
    validate(3, {"type": "number"})
    with pytest.raises(SchemaError):
        validate(3.5, {"type": "integer"})


def test_type_union():
    validate(None, {"type": ["string", "null"]})
    with pytest.raises(SchemaError):
        validate(1, {"type": ["string", "null"]})
