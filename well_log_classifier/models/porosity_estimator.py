"""TensorFlow/Keras deep learning model for porosity prediction."""

import numpy as np
import pickle
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class PorosityEstimator:
    """Keras neural network regressor for porosity prediction."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self._fitted = False
        self._input_dim = None

    def _build_model(self, input_dim):
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def train(self, X_train, y_train) -> dict:
        """Train the Keras regressor and return training metrics."""
        self._input_dim = X_train.shape[1]
        self.model = self._build_model(self._input_dim)
        self.model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
        self._fitted = True
        y_pred = self.model.predict(X_train, verbose=0).flatten()
        rmse = np.sqrt(mean_squared_error(y_train, y_pred))
        r2 = r2_score(y_train, y_pred)
        return {
            "train_rmse": round(float(rmse), 4),
            "train_r2": round(float(r2), 4),
        }

    def evaluate(self, X_test, y_test) -> dict:
        """Evaluate on test data and return metrics."""
        if not self._fitted:
            raise RuntimeError("Model not trained. Call train() first.")
        y_pred = self.model.predict(X_test, verbose=0).flatten()
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return {
            "test_rmse": round(float(rmse), 4),
            "test_mae": round(float(mae), 4),
            "test_r2": round(float(r2), 4),
        }

    def predict(self, X) -> np.ndarray:
        """Predict porosity values."""
        return self.model.predict(X, verbose=0).flatten()

    def save(self, path: str):
        """Save the model to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model_path = path.replace('.pkl', '_keras.keras')
        self.model.save(model_path)
        with open(path, "wb") as f:
            pickle.dump({"input_dim": self._input_dim, "scaler": self.scaler}, f)

    @classmethod
    def load(cls, path: str) -> "PorosityEstimator":
        """Load a saved model from disk."""
        instance = cls.__new__(cls)
        with open(path, "rb") as f:
            data = pickle.load(f)
        instance._input_dim = data.get("input_dim")
        instance.scaler = data.get("scaler")
        model_path = path.replace('.pkl', '_keras.keras')
        instance.model = keras.models.load_model(model_path)
        instance._fitted = True
        instance.random_state = 42
        return instance
