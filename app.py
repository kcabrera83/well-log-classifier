"""FastAPI for well log classification and porosity estimation using TensorFlow/Keras."""

import os
import numpy as np
from typing import Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

from well_log_classifier.models.lithology_classifier import LithologyClassifier
from well_log_classifier.models.porosity_estimator import PorosityEstimator
from well_log_classifier.utils.preprocessor import FEATURE_COLUMNS
from well_log_classifier.data_generator import generate_well_log_data, LITHOLOGY_LABELS

app = FastAPI(
    title="Well Log Classifier",
    description="Lithology classification and porosity estimation from well log features (TensorFlow/Keras)",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "outputs", "models")
classifier = None
porosity_estimator = None


@app.on_event("startup")
async def load_models():
    global classifier, porosity_estimator
    try:
        clf_path = os.path.join(MODELS_DIR, "lithology_classifier.pkl")
        por_path = os.path.join(MODELS_DIR, "porosity_estimator.pkl")
        if os.path.exists(clf_path) and os.path.exists(por_path):
            classifier = LithologyClassifier.load(clf_path)
            porosity_estimator = PorosityEstimator.load(por_path)
        else:
            print("[WARN] Models not found. Run 'python train.py' first.")
    except Exception as e:
        print(f"[WARN] Error loading models: {e}")


class ClassifyRequest(BaseModel):
    features: list[float]


class ClassifyResponse(BaseModel):
    prediction: str
    probabilities: Dict[str, float]


class PorosityRequest(BaseModel):
    features: list[float]


class PorosityResponse(BaseModel):
    porosity: float


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "models_loaded": classifier is not None and porosity_estimator is not None,
        "version": "2.0.0",
        "framework": "tensorflow/keras",
    }


@app.get("/api/models")
async def models_info():
    return {
        "lithology_classifier": {
            "loaded": classifier is not None,
            "type": "Keras Neural Network (Dense layers)",
            "framework": "tensorflow/keras",
        },
        "porosity_estimator": {
            "loaded": porosity_estimator is not None,
            "type": "Keras Neural Network (Dense regression)",
            "framework": "tensorflow/keras",
        },
        "features": FEATURE_COLUMNS,
        "lithology_classes": LITHOLOGY_LABELS,
    }


@app.post("/api/classify", response_model=ClassifyResponse)
async def classify(request: ClassifyRequest):
    if not request.features:
        raise HTTPException(status_code=400, detail="Missing 'features' field")
    features = np.array(request.features).reshape(1, -1)
    if features.shape[1] != len(FEATURE_COLUMNS):
        raise HTTPException(
            status_code=400,
            detail=f"Expected {len(FEATURE_COLUMNS)} features: {FEATURE_COLUMNS}",
        )
    try:
        if classifier.scaler is not None:
            features = classifier.scaler.transform(features)
        raw_pred = classifier.predict(features)[0]
        probabilities = classifier.predict_proba(features)[0]
        classes = classifier.classes
        if hasattr(classifier, "label_encoder") and classifier.label_encoder.classes_.size > 0:
            prediction = str(classifier.label_encoder.inverse_transform([raw_pred])[0])
            class_names = [str(c) for c in classifier.label_encoder.inverse_transform(classes)]
        else:
            prediction = str(raw_pred)
            class_names = [str(c) for c in classes]
        prob_dict = {cls_name: round(float(p), 4) for cls_name, p in zip(class_names, probabilities)}
        return ClassifyResponse(prediction=prediction, probabilities=prob_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/porosity", response_model=PorosityResponse)
async def porosity(request: PorosityRequest):
    if not request.features:
        raise HTTPException(status_code=400, detail="Missing 'features' field")
    features = np.array(request.features).reshape(1, -1)
    if features.shape[1] != len(FEATURE_COLUMNS):
        raise HTTPException(
            status_code=400,
            detail=f"Expected {len(FEATURE_COLUMNS)} features: {FEATURE_COLUMNS}",
        )
    try:
        if porosity_estimator.scaler is not None:
            features = porosity_estimator.scaler.transform(features)
        prediction = float(porosity_estimator.predict(features)[0])
        return PorosityResponse(porosity=round(prediction, 4))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
