from flask import Blueprint, request, jsonify
from werkzeug.exceptions import BadRequest, Unauthorized
import jsonschema
from typing import Dict, Tuple, Union
import os
from functools import wraps
import numpy as np
from scipy.stats import linregress
from datetime import datetime
from schemas.ml import rlrt_schema

ml = Blueprint('ml', __name__)

API_TOKEN = os.environ.get("API_TOKEN")

def validate_token(auth_header: str) -> bool:
    if not auth_header:
        return False
    try:
        auth_type, token = auth_header.split(None, 1)
        return auth_type.lower() == 'bearer' and token.strip() == API_TOKEN
    except ValueError:
        return False

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not validate_token(auth_header):
            return jsonify({"error": "Invalid or missing Authorization header"}), 401
        return f(*args, **kwargs)
    return decorated

def validate_schema(data: Dict, schema: Dict) -> None:
    try:
        jsonschema.validate(instance=data, schema=schema)
    except jsonschema.exceptions.ValidationError as validation_error:
        raise BadRequest(f"Invalid request data: {validation_error}")

@ml.errorhandler(BadRequest)
def handle_bad_request(e):
    return jsonify({"error": str(e)}), 400

@ml.errorhandler(Unauthorized)
def handle_unauthorized(e):
    return jsonify({"error": str(e)}), 401

@ml.route('/rlrt', methods=['POST'])
@require_auth
def compute_daily_rlrt_trend() -> Tuple[Dict[str, Union[str, float]], int]:
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        validate_schema(data, rlrt_schema)
        
        sorted_data = sorted(data['data'], key=lambda x: x['date'])
        
        dates = [datetime.strptime(item['date'], '%Y-%m-%d') for item in sorted_data]
        spreads = [item['spread'] for item in sorted_data]
        
        x = np.array([(date - dates[0]).days for date in dates])
        y = np.array(spreads)
        
        slope, intercept, r_value, _, _ = linregress(x, y)
        r2 = r_value ** 2    
        
        next_day = len(x)
        
        prediction = slope * next_day + intercept
        trend = "positive" if prediction > spreads[-1] else "negative"
        
        return jsonify({
            "trend": trend,
            "confidence": r2
        }), 201
    except BadRequest as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500