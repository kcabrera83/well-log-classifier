"""Training script for well log classification and porosity estimation models using TensorFlow/Keras."""

import os
import sys
import json
import numpy as np

from well_log_classifier.data_generator import generate_well_log_data, LITHOLOGY_LABELS
from well_log_classifier.utils.preprocessor import WellLogPreprocessor, FEATURE_COLUMNS
from well_log_classifier.models.lithology_classifier import LithologyClassifier
from well_log_classifier.models.porosity_estimator import PorosityEstimator


def main():
    print("=" * 60)
    print("  WELL LOG CLASSIFIER - MODEL TRAINING (TensorFlow/Keras)")
    print("  Elaborado por Ing. Kelvin Cabrera")
    print("=" * 60)

    print("\n[1/5] Generating synthetic well log data...")
    df = generate_well_log_data(n_samples=5000, random_state=42)
    print(f"  Generated {len(df)} samples across {df['lithology'].nunique()} lithologies")
    print(f"  Lithology distribution:\n{df['lithology'].value_counts().to_string()}")

    print("\n[2/5] Preprocessing data...")
    preprocessor = WellLogPreprocessor()
    X, y = preprocessor.fit_transform(df)
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    print(f"  Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
    print(f"  Features: {FEATURE_COLUMNS}")

    print("\n[3/5] Training lithology classifier (Keras Neural Network)...")
    classifier = LithologyClassifier(random_state=42)
    classifier.label_encoder = preprocessor.label_encoder
    classifier.scaler = preprocessor.scaler
    train_metrics = classifier.train(X_train, y_train)
    print(f"  Training accuracy: {train_metrics['train_accuracy']}")

    eval_metrics = classifier.evaluate(X_test, y_test, target_names=preprocessor.classes)
    print(f"  Test accuracy: {eval_metrics['test_accuracy']}")

    report = eval_metrics["classification_report"]
    print("\n  Classification Report:")
    print(f"  {'Class':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("  " + "-" * 55)
    for cls in preprocessor.classes:
        if cls in report:
            m = report[cls]
            print(f"  {cls:<15} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1-score']:>10.4f} {m['support']:>10.0f}")

    print("\n[4/5] Training porosity estimator (Keras Neural Network)...")
    y_porosity = ((df["neutron_porosity"] + df["density_porosity"]) / 2).values
    from sklearn.model_selection import train_test_split
    X_p_train, X_p_test, y_p_train, y_p_test = train_test_split(
        X, y_porosity, test_size=0.2, random_state=42
    )

    porosity_estimator = PorosityEstimator(random_state=42)
    porosity_estimator.scaler = preprocessor.scaler
    p_train_metrics = porosity_estimator.train(X_p_train, y_p_train)
    print(f"  Training RMSE: {p_train_metrics['train_rmse']} | R2: {p_train_metrics['train_r2']}")

    p_eval_metrics = porosity_estimator.evaluate(X_p_test, y_p_test)
    print(f"  Test RMSE: {p_eval_metrics['test_rmse']} | MAE: {p_eval_metrics['test_mae']} | R2: {p_eval_metrics['test_r2']}")

    print("\n[5/5] Saving models...")
    os.makedirs("outputs/models", exist_ok=True)
    classifier.save("outputs/models/lithology_classifier.pkl")
    porosity_estimator.save("outputs/models/porosity_estimator.pkl")
    print("  Saved: outputs/models/lithology_classifier.pkl")
    print("  Saved: outputs/models/porosity_estimator.pkl")

    results = {
        "lithology_classifier": {
            "framework": "tensorflow/keras",
            "architecture": "Dense(128)->Dropout->Dense(64)->Dropout->Dense(32)->Dense(num_classes)",
            "train_accuracy": train_metrics["train_accuracy"],
            "test_accuracy": eval_metrics["test_accuracy"],
            "classification_report": eval_metrics["classification_report"],
        },
        "porosity_estimator": {
            "framework": "tensorflow/keras",
            "architecture": "Dense(128)->Dropout->Dense(64)->Dropout->Dense(32)->Dense(1)",
            "train_rmse": p_train_metrics["train_rmse"],
            "train_r2": p_train_metrics["train_r2"],
            "test_rmse": p_eval_metrics["test_rmse"],
            "test_mae": p_eval_metrics["test_mae"],
            "test_r2": p_eval_metrics["test_r2"],
        },
        "data": {
            "total_samples": len(df),
            "train_samples": X_train.shape[0],
            "test_samples": X_test.shape[0],
            "features": FEATURE_COLUMNS,
            "lithology_classes": preprocessor.classes,
        },
    }
    with open("outputs/training_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("  Saved: outputs/training_results.json")

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
