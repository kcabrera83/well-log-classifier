"""RandomForest + GradientBoosting ensemble for lithology classification."""

import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder


class LithologyClassifier:
    """Ensemble classifier combining RandomForest and GradientBoosting."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.label_encoder = LabelEncoder()
        self._build_model()

    def _build_model(self):
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1,
        )
        gb = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=self.random_state,
        )
        self.model = VotingClassifier(
            estimators=[("rf", rf), ("gb", gb)],
            voting="soft",
            n_jobs=-1,
        )
        self._fitted = False

    def train(self, X_train, y_train) -> dict:
        """Train the ensemble and return training metrics.

        Args:
            X_train: Scaled feature array.
            y_train: Encoded integer labels.
        """
        self.model.fit(X_train, y_train)
        self._fitted = True
        y_pred = self.model.predict(X_train)
        acc = accuracy_score(y_train, y_pred)
        return {"train_accuracy": round(acc, 4)}

    def evaluate(self, X_test, y_test, target_names=None) -> dict:
        """Evaluate on test data and return metrics."""
        if not self._fitted:
            raise RuntimeError("Model not trained. Call train() first.")
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(
            y_test, y_pred, target_names=target_names, output_dict=True
        )
        return {"test_accuracy": round(acc, 4), "classification_report": report}

    def predict(self, X) -> np.ndarray:
        """Predict lithology labels."""
        return self.model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities."""
        return self.model.predict_proba(X)

    @property
    def classes(self):
        """Return class labels from the model."""
        return self.model.classes_

    def save(self, path: str):
        """Save the model to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "label_encoder": self.label_encoder}, f)

    @classmethod
    def load(cls, path: str) -> "LithologyClassifier":
        """Load a saved model from disk."""
        instance = cls.__new__(cls)
        with open(path, "rb") as f:
            data = pickle.load(f)
        instance.model = data["model"]
        instance.label_encoder = data.get("label_encoder", LabelEncoder())
        instance._fitted = True
        instance.random_state = 42
        return instance
