"""TensorFlow/Keras deep learning model for lithology classification."""

import numpy as np
import pickle
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score


class LithologyClassifier:
    """Keras neural network classifier for lithology classification."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.label_encoder = LabelEncoder()
        self.model = None
        self.scaler = None
        self._fitted = False
        self._input_dim = None
        self._num_classes = None

    def _build_model(self, input_dim, num_classes):
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, X_train, y_train) -> dict:
        """Train the Keras model and return training metrics."""
        self._input_dim = X_train.shape[1]
        self._num_classes = len(np.unique(y_train))
        self.model = self._build_model(self._input_dim, self._num_classes)
        self.model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
        self._fitted = True
        loss, acc = self.model.evaluate(X_train, y_train, verbose=0)
        return {"train_accuracy": round(float(acc), 4)}

    def evaluate(self, X_test, y_test, target_names=None) -> dict:
        """Evaluate on test data and return metrics."""
        if not self._fitted:
            raise RuntimeError("Model not trained. Call train() first.")
        y_pred = self.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(
            y_test, y_pred, target_names=target_names, output_dict=True
        )
        return {"test_accuracy": round(acc, 4), "classification_report": report}

    def predict(self, X) -> np.ndarray:
        """Predict lithology labels."""
        probs = self.model.predict(X, verbose=0)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities."""
        return self.model.predict(X, verbose=0)

    @property
    def classes(self):
        """Return class labels from the label encoder."""
        return self.label_encoder.transform(self.label_encoder.classes_)

    def save(self, path: str):
        """Save the model to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model_path = path.replace('.pkl', '_keras.keras')
        self.model.save(model_path)
        with open(path, "wb") as f:
            pickle.dump({
                "label_encoder": self.label_encoder,
                "input_dim": self._input_dim,
                "num_classes": self._num_classes,
                "scaler": self.scaler,
            }, f)

    @classmethod
    def load(cls, path: str) -> "LithologyClassifier":
        """Load a saved model from disk."""
        instance = cls.__new__(cls)
        with open(path, "rb") as f:
            data = pickle.load(f)
        instance.label_encoder = data.get("label_encoder", LabelEncoder())
        instance._input_dim = data.get("input_dim")
        instance._num_classes = data.get("num_classes")
        instance.scaler = data.get("scaler")
        model_path = path.replace('.pkl', '_keras.keras')
        instance.model = keras.models.load_model(model_path)
        instance._fitted = True
        instance.random_state = 42
        return instance
