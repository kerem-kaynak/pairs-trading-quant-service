from flask import Blueprint, request, jsonify
from werkzeug.exceptions import BadRequest, Unauthorized
from typing import Any, Dict, List, Tuple, Union
import pendulum

from schemas.ml import rlrt_schema, pairs_schema
from utils.ml import apply_optics, apply_pca_and_scaling, calculate_rlrt_trend_and_confidence
from utils.preprocessing import compute_returns, construct_df_from_ohlc
from utils.router import require_auth, validate_schema
from utils.spread_stats import run_statistical_criteria_tests_for_pairs


ml = Blueprint('ml', __name__)

@ml.errorhandler(BadRequest)
def handle_bad_request(e: BadRequest) -> Tuple[Dict[str, str], int]:
    """
    Error handler for BadRequest exceptions.

    :param e: The BadRequest exception
    :returns: A JSON response with the error message and a 400 status code
    """
    return jsonify({"error": str(e)}), 400

@ml.errorhandler(Unauthorized)
def handle_unauthorized(e: Unauthorized) -> Tuple[Dict[str, str], int]:
    """
    Error handler for Unauthorized exceptions.

    :param e: The Unauthorized exception
    :returns: A JSON response with the error message and a 401 status code
    """
    return jsonify({"error": str(e)}), 401

@ml.route('/rlrt', methods=['POST'])
@require_auth
def compute_daily_rlrt_trend() -> Tuple[Dict[str, Union[str, List[Dict[str, Union[str, float]]]]], int]:
    """
    Compute the daily RLRT trend based on the provided data.

    :returns: A tuple containing a dictionary with the results or error message, and the HTTP status code
    """
    try:
        data: Dict[str, Any] = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        if 'data' not in data or len(data['data']) < 10:
            return jsonify({"error": "At least 10 data points are required"}), 400
        
        validate_schema(data, rlrt_schema)
        
        sorted_data: List[Dict[str, Union[str, float]]] = sorted(data['data'], key=lambda x: x['date'])
        
        dates: List[pendulum.DateTime] = [pendulum.parse(item['date']) for item in sorted_data]
        spreads: List[float] = [item['spread'] for item in sorted_data]
        
        window_size: int = 10
        results: List[Dict[str, Union[str, float]]] = []
        
        for i in range(len(sorted_data) - window_size + 1):
            window_dates = dates[i:i+window_size]
            window_spreads = spreads[i:i+window_size]
            result = calculate_rlrt_trend_and_confidence(window_dates, window_spreads)
            results.append(result)
        
        return jsonify(results), 200
    except BadRequest as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500
    

@ml.route('/pairs', methods=['POST'])
@require_auth
def suggest_pairs() -> Tuple[Dict[str, Union[List[Dict[str, Any]], str]], int]:
    """
    Suggest pairs of tickers based on the provided data using machine learning techniques.

    :returns: A tuple containing a dictionary with the suggested pairs or error message, and the HTTP status code
    """
    try:
        data: Dict[str, Any] = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        if 'data' not in data or len(data['data']) < 30:
            return jsonify({"error": "At least 30 data points are required for clustering"}), 400
        
        validate_schema(data, pairs_schema)

        df = construct_df_from_ohlc(data['data'])
        df_returns = compute_returns(df)
        scaled_principal_components = apply_pca_and_scaling(df_returns)
        pairs_to_eval = apply_optics(scaled_principal_components, df_returns)
        print(pairs_to_eval)
        suggested_pairs = run_statistical_criteria_tests_for_pairs(
            pairs_to_eval,
            df
        )

        return jsonify({"suggested_pairs": suggested_pairs}), 200
    except BadRequest as e:
        return jsonify({"error": str(e)}), 400
    except ValueError as e:
        return jsonify({"error": f"Invalid input data: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500