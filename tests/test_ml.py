import pytest
from flask import Flask, json
from flask.testing import FlaskClient
from typing import List, Dict, Union
import json
import pendulum

from routes.ml import ml
from utils.router import API_TOKEN
import random


@pytest.fixture
def app() -> Flask:
    """
    Create and configure a Flask app for testing.

    :returns: A Flask application instance configured for testing
    """
    app = Flask(__name__)
    app.register_blueprint(ml, url_prefix="/ml")
    app.config['TESTING'] = True
    return app

@pytest.fixture
def client(app: Flask) -> FlaskClient:
    """
    Create a test client for the Flask app.

    :param app: The Flask application instance
    :returns: A test client for the Flask application
    """
    return app.test_client()

def generate_data(num_points: int, start_spread: float, end_spread: float) -> List[Dict[str, Union[str, float]]]:
    """
    Generate synthetic data for testing.

    :param num_points: Number of data points to generate
    :param start_spread: Starting value for the spread
    :param end_spread: Ending value for the spread
    :returns: A list of dictionaries containing date and spread values
    """
    data = []
    for i in range(num_points):
        spread = start_spread + (end_spread - start_spread) * i / (num_points - 1)
        data.append({
            "date": f"2024-01-{i+1:02d}",
            "spread": round(spread, 2)
        })
    return data

def test_rlrt_not_enough_data_points(client: FlaskClient) -> None:
    """
    Test the RLRT endpoint with insufficient data points.

    :param client: The test client for the Flask application
    """
    data = {"data": generate_data(9, 0.54, 0.62)}
    headers = {'Authorization': f'Bearer {API_TOKEN}'}
    response = client.post('/ml/rlrt', json=data, headers=headers)
    assert response.status_code == 400
    response_data = json.loads(response.data)
    assert "error" in response_data
    assert "At least 10 data points are required" in response_data["error"]

def test_rlrt_minimum_data_points(client: FlaskClient) -> None:
    """
    Test the RLRT endpoint with the minimum required data points.

    :param client: The test client for the Flask application
    """
    data = {"data": generate_data(10, 0.54, 0.62)}
    headers = {'Authorization': f'Bearer {API_TOKEN}'}
    response = client.post('/ml/rlrt', json=data, headers=headers)
    assert response.status_code == 200
    result = json.loads(response.data)
    assert isinstance(result, list)
    assert len(result) == 1
    assert 'trend' in result[0]
    assert 'confidence' in result[0]

def test_rlrt_more_than_minimum_data_points(client: FlaskClient) -> None:
    """
    Test the RLRT endpoint with more than the minimum required data points.

    :param client: The test client for the Flask application
    """
    data = {"data": generate_data(12, 0.54, 0.62)}
    headers = {'Authorization': f'Bearer {API_TOKEN}'}
    response = client.post('/ml/rlrt', json=data, headers=headers)
    assert response.status_code == 200
    result = json.loads(response.data)
    assert isinstance(result, list)
    assert len(result) == 3
    for item in result:
        assert 'trend' in item
        assert 'confidence' in item

def test_rlrt_positive_trend(client: FlaskClient) -> None:
    """
    Test the RLRT endpoint with data that should result in a positive trend.

    :param client: The test client for the Flask application
    """
    data = {"data": generate_data(12, 0.54, 0.62)}
    headers = {'Authorization': f'Bearer {API_TOKEN}'}
    response = client.post('/ml/rlrt', json=data, headers=headers)
    assert response.status_code == 200
    result = json.loads(response.data)
    assert all(item['trend'] == 'positive' for item in result)
    assert all(0 <= item['confidence'] <= 1 for item in result)

def test_rlrt_negative_trend(client: FlaskClient) -> None:
    """
    Test the RLRT endpoint with data that should result in a negative trend.

    :param client: The test client for the Flask application
    """
    data = {"data": generate_data(12, 0.62, 0.54)}
    headers = {'Authorization': f'Bearer {API_TOKEN}'}
    response = client.post('/ml/rlrt', json=data, headers=headers)
    assert response.status_code == 200
    result = json.loads(response.data)
    assert all(item['trend'] == 'negative' for item in result)
    assert all(0 <= item['confidence'] <= 1 for item in result)

def test_rlrt_no_auth(client: FlaskClient) -> None:
    """
    Test the RLRT endpoint without authentication.

    :param client: The test client for the Flask application
    """
    data = {"data": generate_data(10, 0.54, 0.62)}
    response = client.post('/ml/rlrt', json=data)
    assert response.status_code == 401
    response_data = json.loads(response.data)
    assert "error" in response_data
    assert "Invalid or missing Authorization header" in response_data["error"]

def test_rlrt_invalid_auth(client: FlaskClient) -> None:
    """
    Test the RLRT endpoint with invalid authentication.

    :param client: The test client for the Flask application
    """
    data = {"data": generate_data(10, 0.54, 0.62)}
    headers = {'Authorization': 'Bearer invalid_token'}
    response = client.post('/ml/rlrt', json=data, headers=headers)
    assert response.status_code == 401
    response_data = json.loads(response.data)
    assert "error" in response_data
    assert "Invalid or missing Authorization header" in response_data["error"]

def test_rlrt_empty_json(client: FlaskClient) -> None:
    """
    Test the RLRT endpoint with an empty JSON payload.

    :param client: The test client for the Flask application
    """
    headers = {'Authorization': f'Bearer {API_TOKEN}'}
    response = client.post('/ml/rlrt', json={}, headers=headers)
    assert response.status_code == 400
    response_data = json.loads(response.data)
    assert "error" in response_data
    assert "No JSON data provided" in response_data["error"]

def test_suggest_pairs_no_data(client):
    headers = {'Authorization': f'Bearer {API_TOKEN}'}
    response = client.post('/ml/pairs', json={}, headers=headers)
    assert response.status_code == 400
    response_data = json.loads(response.data)
    assert "error" in response_data
    assert "No JSON data provided" in response_data["error"]

def test_suggest_pairs_insufficient_data(client):
    data = {"data": [{"date": "2023-01-01", "ticker": "AAPL", "price": 100}]}
    headers = {'Authorization': f'Bearer {API_TOKEN}'}
    response = client.post('/ml/pairs', json=data, headers=headers)
    assert response.status_code == 400
    response_data = json.loads(response.data)
    assert "error" in response_data
    assert "At least 30 data points are required for clustering" in response_data["error"]

def test_suggest_pairs_no_auth(client):
    data = {"data": [{"date": "2023-01-01", "ticker": "AAPL", "price": 100}] * 30}
    response = client.post('/ml/pairs', json=data)
    assert response.status_code == 401
    response_data = json.loads(response.data)
    assert "error" in response_data
    assert "Invalid or missing Authorization header" in response_data["error"]

def test_suggest_pairs_invalid_auth(client):
    data = {"data": [{"date": "2023-01-01", "ticker": "AAPL", "price": 100}] * 30}
    headers = {'Authorization': 'Bearer invalid_token'}
    response = client.post('/ml/pairs', json=data, headers=headers)
    assert response.status_code == 401
    response_data = json.loads(response.data)
    assert "error" in response_data
    assert "Invalid or missing Authorization header" in response_data["error"]