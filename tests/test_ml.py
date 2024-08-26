import pytest
from flask import Flask, json
from routes.ml import ml, API_TOKEN

@pytest.fixture
def app():
    app = Flask(__name__)
    app.register_blueprint(ml, url_prefix="/ml")
    app.config['TESTING'] = True
    return app

@pytest.fixture
def client(app):
    return app.test_client()

def generate_data(num_points, start_spread, end_spread):
    data = []
    for i in range(num_points):
        spread = start_spread + (end_spread - start_spread) * i / (num_points - 1)
        data.append({
            "date": f"2024-01-{i+1:02d}",
            "spread": round(spread, 2)
        })
    return data

def test_rlrt_not_enough_data_points(client):
    data = {"data": generate_data(9, 0.54, 0.62)}
    headers = {'Authorization': f'Bearer {API_TOKEN}'}
    response = client.post('/ml/rlrt', json=data, headers=headers)
    assert response.status_code == 400
    response_data = json.loads(response.data)
    assert "error" in response_data
    assert "Invalid request data" in response_data["error"]
    assert "is too short" in response_data["error"]

def test_rlrt_minimum_data_points(client):
    data = {"data": generate_data(10, 0.54, 0.62)}
    headers = {'Authorization': f'Bearer {API_TOKEN}'}
    response = client.post('/ml/rlrt', json=data, headers=headers)
    assert response.status_code == 201
    result = json.loads(response.data)
    assert 'trend' in result
    assert 'confidence' in result

def test_rlrt_incorrect_schema(client):
    data = {
        "data": [
            {"date": "2024-01-01", "spread": "not a number"},
            {"incorrect_key": "incorrect_value"}
        ] + generate_data(8, 0.54, 0.62)
    }
    headers = {'Authorization': f'Bearer {API_TOKEN}'}
    response = client.post('/ml/rlrt', json=data, headers=headers)
    assert response.status_code == 400
    response_data = json.loads(response.data)
    assert "error" in response_data
    assert "Invalid request data" in response_data["error"]

def test_rlrt_positive_trend(client):
    data = {"data": generate_data(10, 0.54, 0.62)}
    headers = {'Authorization': f'Bearer {API_TOKEN}'}
    response = client.post('/ml/rlrt', json=data, headers=headers)
    assert response.status_code == 201
    result = json.loads(response.data)
    assert result['trend'] == 'positive'
    assert 0 <= result['confidence'] <= 1

def test_rlrt_negative_trend(client):
    data = {"data": generate_data(10, 0.62, 0.54)}
    headers = {'Authorization': f'Bearer {API_TOKEN}'}
    response = client.post('/ml/rlrt', json=data, headers=headers)
    assert response.status_code == 201
    result = json.loads(response.data)
    assert result['trend'] == 'negative'
    assert 0 <= result['confidence'] <= 1

def test_rlrt_no_auth(client):
    data = {"data": generate_data(10, 0.54, 0.62)}
    response = client.post('/ml/rlrt', json=data)
    assert response.status_code == 401
    response_data = json.loads(response.data)
    assert "error" in response_data
    assert "Invalid or missing Authorization header" in response_data["error"]

def test_rlrt_invalid_auth(client):
    data = {"data": generate_data(10, 0.54, 0.62)}
    headers = {'Authorization': 'Bearer invalid_token'}
    response = client.post('/ml/rlrt', json=data, headers=headers)
    assert response.status_code == 401
    response_data = json.loads(response.data)
    assert "error" in response_data
    assert "Invalid or missing Authorization header" in response_data["error"]

def test_rlrt_empty_json(client):
    headers = {'Authorization': f'Bearer {API_TOKEN}'}
    response = client.post('/ml/rlrt', json={}, headers=headers)
    assert response.status_code == 400
    response_data = json.loads(response.data)
    assert "error" in response_data
    assert "No JSON data provided" in response_data["error"]