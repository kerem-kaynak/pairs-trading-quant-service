from functools import wraps
import os
from typing import Any, Callable, Dict, Tuple
from werkzeug.exceptions import BadRequest
from flask import jsonify, request
import jsonschema


API_TOKEN = os.environ.get("API_TOKEN")

def validate_token(auth_header: str) -> bool:
    """
    Validate the authentication token from the request header.

    :param auth_header: The Authorization header from the request
    :returns: True if the token is valid, False otherwise
    """
    if not auth_header:
        return False
    try:
        auth_type, token = auth_header.split(None, 1)
        return auth_type == 'Bearer' and token.strip() == API_TOKEN
    except ValueError:
        return False

def require_auth(f: Callable) -> Callable:
    """
    Decorator to require authentication for a route.

    :param f: The function to be decorated
    :returns: The decorated function
    """
    @wraps(f)
    def decorated(*args: Any, **kwargs: Any) -> Tuple[Dict[str, Any], int]:
        auth_header = request.headers.get('Authorization')
        if not validate_token(auth_header):
            return jsonify({"error": "Invalid or missing Authorization header"}), 401
        return f(*args, **kwargs)
    return decorated

def validate_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> None:
    """
    Validate the request data against the provided schema.

    :param data: The request data to validate
    :param schema: The schema to validate against
    :raises BadRequest: If the data does not match the schema
    """
    try:
        jsonschema.validate(instance=data, schema=schema)
    except jsonschema.exceptions.ValidationError as validation_error:
        raise BadRequest(f"Invalid request data: {validation_error}")
