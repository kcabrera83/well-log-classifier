import pytest
import os

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "models")


def test_outputs_directory_exists():
    assert os.path.exists(MODELS_DIR)


def test_model_files_exist():
    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith((".pkl", ".joblib", ".h5", ".pt"))]
    assert len(model_files) > 0


def test_classifier_model_loads():
    from well_log_classifier.models.lithology_classifier import LithologyClassifier
    path = os.path.join(MODELS_DIR, "lithology_classifier.pkl")
    model = LithologyClassifier.load(path)
    assert model is not None


def test_porosity_model_loads():
    from well_log_classifier.models.porosity_estimator import PorosityEstimator
    path = os.path.join(MODELS_DIR, "porosity_estimator.pkl")
    model = PorosityEstimator.load(path)
    assert model is not None


def test_classifier_prediction():
    import numpy as np
    from well_log_classifier.models.lithology_classifier import LithologyClassifier
    model = LithologyClassifier.load(os.path.join(MODELS_DIR, "lithology_classifier.pkl"))
    X = np.array([[40.0, 50.0, 25.0, 22.0, 75.0, 8.5]])
    pred = model.predict(X)
    assert pred is not None
    assert len(pred) == 1


def test_porosity_prediction():
    import numpy as np
    from well_log_classifier.models.porosity_estimator import PorosityEstimator
    model = PorosityEstimator.load(os.path.join(MODELS_DIR, "porosity_estimator.pkl"))
    X = np.array([[40.0, 50.0, 25.0, 22.0, 75.0, 8.5]])
    pred = model.predict(X)
    assert pred is not None
    assert len(pred) == 1
    assert pred[0] > 0
