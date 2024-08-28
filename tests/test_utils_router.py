from flask import Flask
import pytest
from unittest.mock import patch, MagicMock
from werkzeug.exceptions import BadRequest
from utils.router import validate_token, require_auth, validate_schema

@pytest.mark.parametrize("auth_header, expected", [
    ('Bearer test_token', True),
    ('Bearer wrong_token', False),
    ('bearer test_token', False),
    ('test_token', False),
    ('', False)
])
def test_validate_token(auth_header, expected):
    with patch('utils.router.API_TOKEN', 'test_token'):
        assert validate_token(auth_header) == expected

def test_validate_schema():
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name", "age"]
    }
    valid_data = {"name": "John", "age": 30}
    invalid_data = {"name": "John"}

    validate_schema(valid_data, schema)

    with pytest.raises(BadRequest):
        validate_schema(invalid_data, schema)