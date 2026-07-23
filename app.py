"""Flask API for well log classification and porosity estimation."""

import os
import json
import numpy as np
from flask import Flask, request, Response, render_template

from well_log_classifier.models.lithology_classifier import LithologyClassifier
from well_log_classifier.models.porosity_estimator import PorosityEstimator
from well_log_classifier.utils.preprocessor import FEATURE_COLUMNS
from well_log_classifier.data_generator import generate_well_log_data, LITHOLOGY_LABELS

app = Flask(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "outputs", "models")
classifier = None
porosity_estimator = None


def _json_response(data, status=200):
    return Response(
        json.dumps(data, default=lambda o: float(o) if isinstance(o, (np.floating,)) else int(o) if isinstance(o, (np.integer,)) else o.tolist() if isinstance(o, np.ndarray) else str(o)),
        status=status,
        mimetype="application/json",
    )


def load_models():
    global classifier, porosity_estimator
    clf_path = os.path.join(MODELS_DIR, "lithology_classifier.pkl")
    por_path = os.path.join(MODELS_DIR, "porosity_estimator.pkl")
    if os.path.exists(clf_path) and os.path.exists(por_path):
        classifier = LithologyClassifier.load(clf_path)
        porosity_estimator = PorosityEstimator.load(por_path)
        return True
    return False


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/classify", methods=["POST"])
def classify():
    try:
        data = request.get_json()
        if not data or "features" not in data:
            return _json_response({"error": "Missing 'features' field"}, 400)

        features = np.array(data["features"]).reshape(1, -1)
        if features.shape[1] != len(FEATURE_COLUMNS):
            return _json_response({
                "error": f"Expected {len(FEATURE_COLUMNS)} features: {FEATURE_COLUMNS}"
            }, 400)

        raw_pred = classifier.predict(features)[0]
        probabilities = classifier.predict_proba(features)[0]

        classes = classifier.classes
        # Map encoded labels to lithology names if label encoder is available
        if hasattr(classifier, "label_encoder") and classifier.label_encoder.classes_.size > 0:
            prediction = str(classifier.label_encoder.inverse_transform([raw_pred])[0])
            class_names = [str(c) for c in classifier.label_encoder.inverse_transform(classes)]
        else:
            prediction = str(raw_pred)
            class_names = [str(c) for c in classes]

        prob_dict = {}
        for cls_name, p in zip(class_names, probabilities):
            prob_dict[cls_name] = round(float(p), 4)

        return _json_response({
            "prediction": prediction,
            "probabilities": prob_dict,
        })
    except Exception as e:
        return _json_response({"error": str(e)}, 500)


@app.route("/api/porosity", methods=["POST"])
def porosity():
    try:
        data = request.get_json()
        if not data or "features" not in data:
            return _json_response({"error": "Missing 'features' field"}, 400)

        features = np.array(data["features"]).reshape(1, -1)
        if features.shape[1] != len(FEATURE_COLUMNS):
            return _json_response({
                "error": f"Expected {len(FEATURE_COLUMNS)} features: {FEATURE_COLUMNS}"
            }, 400)

        prediction = float(porosity_estimator.predict(features)[0])

        return _json_response({
            "porosity": round(prediction, 4),
        })
    except Exception as e:
        return _json_response({"error": str(e)}, 500)


@app.route("/api/models", methods=["GET"])
def models_info():
    clf_loaded = classifier is not None
    por_loaded = porosity_estimator is not None
    return _json_response({
        "lithology_classifier": {
            "loaded": clf_loaded,
            "type": "RandomForest + GradientBoosting (VotingClassifier)",
        },
        "porosity_estimator": {
            "loaded": por_loaded,
            "type": "GradientBoostingRegressor",
        },
        "features": FEATURE_COLUMNS,
        "lithology_classes": LITHOLOGY_LABELS,
    })


@app.route("/api/health", methods=["GET"])
def health():
    return _json_response({
        "status": "healthy",
        "models_loaded": classifier is not None and porosity_estimator is not None,
        "version": "1.0.0",
    })


if __name__ == "__main__":
    loaded = load_models()
    if not loaded:
        print("[WARN] Models not found. Run 'python train.py' first.")
    else:
        print("[OK] Models loaded successfully.")
    app.run(host="0.0.0.0", port=5008, debug=False)
