"""Train/test split and reusable preprocessing (ColumnTransformer) builders.

All transformers are placed inside scikit-learn Pipelines, so imputation,
scaling and encoding are fitted on the training split only and the exact same
transformations are applied at inference time (CLI and API).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src import config


def add_temperature_stress(X):
    """Append |avg_temp_c - 18| as an extra column (agronomic temperature stress).

    Potatoes grow best around 18 C, so distance from that optimum is a
    meaningful input. It uses inputs only, never the target. ``X`` holds the
    numeric feature columns in ``config.NUMERIC_FEATURES`` order.
    """
    X = np.asarray(X, dtype=float)
    temp_idx = config.NUMERIC_FEATURES.index("avg_temp_c")
    stress = np.abs(X[:, temp_idx] - config.OPTIMAL_TEMP_C).reshape(-1, 1)
    return np.hstack([X, stress])


def numeric_feature_names(_transformer=None, input_features=None):
    """Feature names emitted by ``add_temperature_stress``."""
    return np.array(list(config.NUMERIC_FEATURES) + ["temp_stress_c"], dtype=object)


def split_data(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fixed, stratified 80/20 split with ``random_state=seed``.

    The same split is shared by regression, classification and clustering so
    the test rows are never seen by any fitted transformer.
    """
    train, test = train_test_split(
        df,
        test_size=config.TEST_SIZE,
        random_state=seed,
        stratify=df[config.CLASSIFICATION_TARGET],
    )
    return train.sort_values("row_id"), test.sort_values("row_id")


def build_preprocessor(scale: bool = True) -> ColumnTransformer:
    """Build the shared preprocessing step for regression and classification.

    * numeric: median imputation -> temperature-stress feature -> standard scaling
    * season: most-frequent imputation -> one-hot encoding
    * cooperative_support: most-frequent imputation, kept as 0/1
    """
    numeric_steps = [
        ("impute", SimpleImputer(strategy="median")),
        ("temp_stress", FunctionTransformer(add_temperature_stress,
                                            feature_names_out=numeric_feature_names)),
    ]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), config.NUMERIC_FEATURES),
            ("season", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), config.CATEGORICAL_FEATURES),
            ("support", SimpleImputer(strategy="most_frequent"), config.BINARY_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_cluster_preprocessor() -> Pipeline:
    """Imputation + scaling for the clustering features (growing conditions)."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
