import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

import pytest

SAMPLE_PAYLOAD_NO_DEFAULT = {
    "LIMIT_BAL": 200000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 29,
    "PAY_0": -1, "PAY_2": -1, "PAY_3": -1, "PAY_4": -1, "PAY_5": -1, "PAY_6": -1,
    "BILL_AMT1": 1800, "BILL_AMT2": 1500, "BILL_AMT3": 1200,
    "BILL_AMT4": 1000, "BILL_AMT5": 800,  "BILL_AMT6": 700,
    "PAY_AMT1": 1800, "PAY_AMT2": 1500, "PAY_AMT3": 1200,
    "PAY_AMT4": 1000, "PAY_AMT5": 800,  "PAY_AMT6": 700,
}

SAMPLE_PAYLOAD_DEFAULT = {
    "LIMIT_BAL": 20000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 24,
    "PAY_0": 2, "PAY_2": 2, "PAY_3": 2, "PAY_4": 0, "PAY_5": 0, "PAY_6": 2,
    "BILL_AMT1": 19000, "BILL_AMT2": 18500, "BILL_AMT3": 18000,
    "BILL_AMT4": 17000, "BILL_AMT5": 16000, "BILL_AMT6": 15000,
    "PAY_AMT1": 0, "PAY_AMT2": 0, "PAY_AMT3": 0,
    "PAY_AMT4": 0, "PAY_AMT5": 0, "PAY_AMT6": 0,
}

MISSING_FEATURE_PAYLOAD = {"LIMIT_BAL": 100000, "SEX": 1}


@pytest.fixture
def client():
    from api import app
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code in (200, 503)

    def test_health_response_structure(self, client):
        data = json.loads(client.get("/health").data)
        assert "status" in data
        assert "models_loaded" in data
        assert isinstance(data["models_loaded"], list)


class TestPredictEndpoint:
    def test_predict_no_json(self, client):
        resp = client.post("/predict", data="not json",
                           content_type="text/plain")
        assert resp.status_code == 400

    def test_predict_missing_features(self, client):
        resp = client.post("/predict",
                           data=json.dumps(MISSING_FEATURE_PAYLOAD),
                           content_type="application/json")
        assert resp.status_code in (400, 503)

    def test_predict_invalid_version(self, client):
        resp = client.post("/predict?version=v99",
                           data=json.dumps(SAMPLE_PAYLOAD_NO_DEFAULT),
                           content_type="application/json")
        assert resp.status_code == 400

    def test_predict_success_v1(self, client):
        resp = client.post("/predict?version=v1",
                           data=json.dumps(SAMPLE_PAYLOAD_NO_DEFAULT),
                           content_type="application/json")
        assert resp.status_code in (200, 503)
        if resp.status_code == 200:
            data = json.loads(resp.data)
            assert "prediction" in data
            assert "probability" in data
            assert data["model_version"] == "v1"
            assert data["prediction"] in (0, 1)
            assert 0.0 <= data["probability"] <= 1.0

    def test_predict_success_v2(self, client):
        resp = client.post("/predict?version=v2",
                           data=json.dumps(SAMPLE_PAYLOAD_DEFAULT),
                           content_type="application/json")
        assert resp.status_code in (200, 503)
        if resp.status_code == 200:
            assert json.loads(resp.data)["model_version"] == "v2"