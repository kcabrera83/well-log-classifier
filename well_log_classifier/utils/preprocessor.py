"""Data preprocessing pipeline for well log data."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

FEATURE_COLUMNS = [
    "gamma_ray",
    "resistivity",
    "neutron_porosity",
    "density_porosity",
    "sonic",
    "caliper",
]


class WellLogPreprocessor:
    """Handles scaling, encoding, and train/test splitting for well log data."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "WellLogPreprocessor":
        """Fit the scaler and label encoder on a DataFrame."""
        self.scaler.fit(df[FEATURE_COLUMNS])
        self.label_encoder.fit(df["lithology"])
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> tuple:
        """Transform a DataFrame into scaled features and encoded labels.

        Returns:
            Tuple of (X_scaled, y_encoded).
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        X = self.scaler.transform(df[FEATURE_COLUMNS])
        y = self.label_encoder.transform(df["lithology"])
        return X, y

    def fit_transform(self, df: pd.DataFrame) -> tuple:
        """Fit and transform in one step."""
        self.fit(df)
        return self.transform(df)

    def split_data(
        self, X, y, test_size: float = 0.2, random_state: int = 42
    ) -> tuple:
        """Split into train and test sets."""
        return train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

    def inverse_transform_labels(self, y_encoded: np.ndarray) -> np.ndarray:
        """Convert encoded labels back to original lithology names."""
        return self.label_encoder.inverse_transform(y_encoded)

    @property
    def classes(self):
        """Return the list of lithology classes."""
        return list(self.label_encoder.classes_)
