"""GradientBoosting regressor for porosity prediction."""

import numpy as np
import pickle
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class PorosityEstimator:
    """GradientBoosting-based porosity estimator from well log features."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
        )
        self._fitted = False

    def train(self, X_train, y_train) -> dict:
        """Train the regressor and return training metrics."""
        self.model.fit(X_train, y_train)
        self._fitted = True
        y_pred = self.model.predict(X_train)
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
        y_pred = self.model.predict(X_test)
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
        return self.model.predict(X)

    def save(self, path: str):
        """Save the model to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    @classmethod
    def load(cls, path: str) -> "PorosityEstimator":
        """Load a saved model from disk."""
        instance = cls.__new__(cls)
        with open(path, "rb") as f:
            instance.model = pickle.load(f)
        instance._fitted = True
        instance.random_state = 42
        return instance
