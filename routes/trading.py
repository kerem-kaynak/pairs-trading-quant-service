from flask import Blueprint, request, jsonify
from werkzeug.exceptions import BadRequest, Unauthorized
from typing import Any, Dict, Tuple

from schemas.trading import trade_schema
from utils.preprocessing import construct_df_from_ohlc
from utils.router import require_auth, validate_schema
from utils.trading import trade_pair_using_model


trading = Blueprint('trading', __name__)

@trading.errorhandler(BadRequest)
def handle_bad_request(e: BadRequest) -> Tuple[Dict[str, str], int]:
    """
    Error handler for BadRequest exceptions.

    :param e: The BadRequest exception
    :returns: A JSON response with the error message and a 400 status code
    """
    return jsonify({"error": str(e)}), 400

@trading.errorhandler(Unauthorized)
def handle_unauthorized(e: Unauthorized) -> Tuple[Dict[str, str], int]:
    """
    Error handler for Unauthorized exceptions.

    :param e: The Unauthorized exception
    :returns: A JSON response with the error message and a 401 status code
    """
    return jsonify({"error": str(e)}), 401

@trading.route('/trade_with_model', methods=['POST'])
@require_auth
def trade_using_model() -> Tuple[Dict[str, Any], int]:
    """
    Compute backtesting statistics using RLRT for the provided spread data.

    :returns: A JSON response containing daily signals, daily positions, daily returns, total returns, maximum drawdowns, annualized returns.
    """
    try:
        data: Dict[str, Any] = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        if 'data' not in data or len(data['data']) < 30:
            return jsonify({"error": "At least 30 data points are required for clustering"}), 400
        
        validate_schema(data, trade_schema)

        df = construct_df_from_ohlc(data['data'])
        results = trade_pair_using_model(df, df.columns[0], df.columns[1])
        return jsonify(results), 200
    except BadRequest as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500
