import pytest
import os
import numpy as np
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "models")


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
    from well_log_classifier.models.lithology_classifier import LithologyClassifier
    model = LithologyClassifier.load(os.path.join(MODELS_DIR, "lithology_classifier.pkl"))
    X = np.array([[40.0, 50.0, 25.0, 22.0, 75.0, 8.5]])
    pred = model.predict(X)
    assert pred is not None
    assert len(pred) == 1


def test_classifier_predict_proba():
    from well_log_classifier.models.lithology_classifier import LithologyClassifier
    model = LithologyClassifier.load(os.path.join(MODELS_DIR, "lithology_classifier.pkl"))
    X = np.array([[40.0, 50.0, 25.0, 22.0, 75.0, 8.5]])
    proba = model.predict_proba(X)
    assert proba is not None
    assert proba.shape[1] == len(model.classes)
    assert abs(proba.sum() - 1.0) < 0.01


def test_porosity_prediction():
    from well_log_classifier.models.porosity_estimator import PorosityEstimator
    model = PorosityEstimator.load(os.path.join(MODELS_DIR, "porosity_estimator.pkl"))
    X = np.array([[40.0, 50.0, 25.0, 22.0, 75.0, 8.5]])
    pred = model.predict(X)
    assert pred is not None
    assert len(pred) == 1
    assert pred[0] > 0


def test_classifier_different_inputs():
    from well_log_classifier.models.lithology_classifier import LithologyClassifier
    model = LithologyClassifier.load(os.path.join(MODELS_DIR, "lithology_classifier.pkl"))
    inputs = [
        np.array([[120.0, 15.0, 35.0, 30.0, 90.0, 9.0]]),
        np.array([[30.0, 200.0, 15.0, 12.0, 55.0, 8.0]]),
        np.array([[50.0, 30.0, 20.0, 18.0, 65.0, 7.5]]),
    ]
    for X in inputs:
        pred = model.predict(X)
        assert pred is not None
        assert len(pred) == 1
