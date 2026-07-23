"""API integration tests for the Well Log Classifier FastAPI app."""

import sys
from fastapi.testclient import TestClient

sys.path.insert(0, ".")
from app import app

client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "healthy"
    assert data["models_loaded"] is True
    print("[PASS] /api/health")


def test_models_info():
    r = client.get("/api/models")
    assert r.status_code == 200
    data = r.json()
    assert data["lithology_classifier"]["loaded"] is True
    assert data["porosity_estimator"]["loaded"] is True
    assert len(data["features"]) == 6
    assert len(data["lithology_classes"]) == 5
    print("[PASS] /api/models")


def test_classify():
    features = [40, 50, 25, 22, 75, 8.5]
    r = client.post("/api/classify", json={"features": features})
    assert r.status_code == 200
    data = r.json()
    assert "prediction" in data
    assert "probabilities" in data
    assert data["prediction"] in ["sandstone", "limestone", "shale", "dolomite", "anhydrite"]
    total_prob = sum(data["probabilities"].values())
    assert abs(total_prob - 1.0) < 0.01
    print(f"[PASS] /api/classify -> {data['prediction']}")


def test_classify_shale():
    features = [120, 15, 35, 30, 95, 9.5]
    r = client.post("/api/classify", json={"features": features})
    assert r.status_code == 200
    data = r.json()
    assert data["prediction"] in ["sandstone", "limestone", "shale", "dolomite", "anhydrite"]
    print(f"[PASS] /api/classify (shale) -> {data['prediction']}")


def test_classify_invalid_features():
    r = client.post("/api/classify", json={"features": [1, 2]})
    assert r.status_code == 400
    print("[PASS] /api/classify (invalid features -> 400)")


def test_classify_missing_body():
    r = client.post("/api/classify", json={})
    assert r.status_code == 422
    print("[PASS] /api/classify (missing body -> 422)")


def test_porosity():
    features = [40, 50, 25, 22, 75, 8.5]
    r = client.post("/api/porosity", json={"features": features})
    assert r.status_code == 200
    data = r.json()
    assert "porosity" in data
    assert isinstance(data["porosity"], float)
    assert 0 <= data["porosity"] <= 100
    print(f"[PASS] /api/porosity -> {data['porosity']}")


def test_porosity_invalid():
    r = client.post("/api/porosity", json={"features": [1]})
    assert r.status_code == 400
    print("[PASS] /api/porosity (invalid features -> 400)")


def main():
    print("=" * 50)
    print("  API INTEGRATION TESTS")
    print("=" * 50)

    tests = [
        test_health, test_models_info, test_classify, test_classify_shale,
        test_classify_invalid_features, test_classify_missing_body,
        test_porosity, test_porosity_invalid,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {t.__name__}: {e}")
            failed += 1

    print(f"\n{'=' * 50}")
    print(f"  Results: {passed} passed, {failed} failed, {len(tests)} total")
    print(f"{'=' * 50}")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
