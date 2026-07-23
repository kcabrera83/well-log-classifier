import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app import app, load_models

load_models()
client = app.test_client()


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert "models_loaded" in data
    assert "version" in data


def test_health_models_loaded():
    response = client.get("/api/health")
    data = response.get_json()
    assert data["models_loaded"] is True


def test_models():
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.get_json()
    assert "lithology_classifier" in data
    assert "porosity_estimator" in data
    assert "features" in data
    assert "lithology_classes" in data
    assert data["lithology_classifier"]["loaded"] is True
    assert data["porosity_estimator"]["loaded"] is True


def test_api_docs():
    response = client.get("/api/docs")
    assert response.status_code == 200
    data = response.get_json()
    assert data["openapi"] == "3.0.0"
    assert "paths" in data


def test_classify_valid_input():
    response = client.post("/api/classify", json={
        "features": [40.0, 50.0, 25.0, 22.0, 75.0, 8.5]
    })
    assert response.status_code == 200
    data = response.get_json()
    assert "prediction" in data
    assert "probabilities" in data
    assert data["prediction"] in ["sandstone", "limestone", "shale", "dolomite", "anhydrite"]
    total_prob = sum(data["probabilities"].values())
    assert abs(total_prob - 1.0) < 0.01


def test_classify_missing_features():
    response = client.post("/api/classify", json={})
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data


def test_classify_wrong_feature_count():
    response = client.post("/api/classify", json={
        "features": [1.0, 2.0, 3.0]
    })
    assert response.status_code == 400


def test_porosity_valid_input():
    response = client.post("/api/porosity", json={
        "features": [40.0, 50.0, 25.0, 22.0, 75.0, 8.5]
    })
    assert response.status_code == 200
    data = response.get_json()
    assert "porosity" in data
    assert isinstance(data["porosity"], float)
    assert data["porosity"] > 0


def test_porosity_missing_features():
    response = client.post("/api/porosity", json={})
    assert response.status_code == 400


def test_classify_sandstone_input():
    response = client.post("/api/classify", json={
        "features": [40.0, 50.0, 25.0, 22.0, 75.0, 8.5]
    })
    data = response.get_json()
    assert response.status_code == 200
    assert "prediction" in data
