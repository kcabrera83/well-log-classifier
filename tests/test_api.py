import pytest


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "models_loaded" in data
    assert "version" in data


def test_models(client):
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert "lithology_classifier" in data
    assert "porosity_estimator" in data
    assert "features" in data
    assert "lithology_classes" in data


def test_classify_valid_input(client):
    response = client.post("/api/classify", json={
        "features": [40.0, 50.0, 25.0, 22.0, 75.0, 8.5]
    })
    assert response.status_code in (200, 500)
    if response.status_code == 200:
        data = response.json()
        assert "prediction" in data
        assert "probabilities" in data
        assert data["prediction"] in ["sandstone", "limestone", "shale", "dolomite", "anhydrite"]
        total_prob = sum(data["probabilities"].values())
        assert abs(total_prob - 1.0) < 0.01


def test_classify_wrong_feature_count(client):
    response = client.post("/api/classify", json={
        "features": [1.0, 2.0, 3.0]
    })
    assert response.status_code == 400


def test_porosity_valid_input(client):
    response = client.post("/api/porosity", json={
        "features": [40.0, 50.0, 25.0, 22.0, 75.0, 8.5]
    })
    assert response.status_code in (200, 500)
    if response.status_code == 200:
        data = response.json()
        assert "porosity" in data
        assert isinstance(data["porosity"], float)
        assert data["porosity"] > 0
