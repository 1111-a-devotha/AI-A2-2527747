"""Candidate models and evidence-based model selection (cross-validation).

Final models are chosen by 5-fold cross-validation on the TRAINING split only.
The held-out test set is used once, afterwards, to report final metrics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import make_scorer, precision_score, silhouette_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src import config
from src.preprocessing import build_cluster_preprocessor, build_preprocessor


def regression_candidates(seed: int) -> dict[str, Pipeline]:
    """Mean baseline plus two regression candidates, each with preprocessing."""
    return {
        "mean_baseline": Pipeline([("prep", build_preprocessor()),
                                   ("model", DummyRegressor(strategy="mean"))]),
        "linear_regression": Pipeline([("prep", build_preprocessor()),
                                       ("model", LinearRegression())]),
        "random_forest": Pipeline([("prep", build_preprocessor(scale=False)),
                                   ("model", RandomForestRegressor(
                                       n_estimators=300, min_samples_leaf=2,
                                       random_state=seed, n_jobs=1))]),
    }


def classification_candidates(seed: int) -> dict[str, Pipeline]:
    """Majority baseline plus two classifiers, each with preprocessing."""
    return {
        "majority_baseline": Pipeline([("prep", build_preprocessor()),
                                       ("model", DummyClassifier(strategy="most_frequent"))]),
        "logistic_regression": Pipeline([("prep", build_preprocessor()),
                                         ("model", LogisticRegression(max_iter=2000,
                                                                      random_state=seed))]),
        "random_forest": Pipeline([("prep", build_preprocessor(scale=False)),
                                   ("model", RandomForestClassifier(
                                       n_estimators=300, min_samples_leaf=2,
                                       random_state=seed, n_jobs=1))]),
    }


def cross_validate_regressors(models: dict, X: pd.DataFrame, y: pd.Series, seed: int) -> dict:
    """5-fold CV MAE / RMSE / R2 for each regression candidate (train split only)."""
    cv = KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=seed)
    out = {}
    for name, model in models.items():
        res = cross_validate(model, X, y, cv=cv, scoring={
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
        })
        out[name] = {
            "cv_mae": round(float(-res["test_mae"].mean()), 6),
            "cv_rmse": round(float(-res["test_rmse"].mean()), 6),
            "cv_rmse_std": round(float(res["test_rmse"].std()), 6),
            "cv_r2": round(float(res["test_r2"].mean()), 6),
        }
    return out


def cross_validate_classifiers(models: dict, X: pd.DataFrame, y: pd.Series, seed: int) -> dict:
    """5-fold stratified CV precision / recall / F1 / ROC-AUC (train split only)."""
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=seed)
    out = {}
    for name, model in models.items():
        res = cross_validate(model, X, y, cv=cv, scoring={
            "precision": make_scorer(precision_score, zero_division=0), "recall": "recall", "f1": "f1", "roc_auc": "roc_auc",
        }, error_score=np.nan)
        out[name] = {
            "cv_precision": round(float(np.nanmean(res["test_precision"])), 6),
            "cv_recall": round(float(np.nanmean(res["test_recall"])), 6),
            "cv_f1": round(float(np.nanmean(res["test_f1"])), 6),
            "cv_f1_std": round(float(np.nanstd(res["test_f1"])), 6),
            "cv_roc_auc": round(float(np.nanmean(res["test_roc_auc"])), 6),
        }
    return out


def select_best(cv_results: dict, metric: str, higher_is_better: bool, exclude: str) -> str:
    """Pick the candidate with the best mean CV score, excluding the baseline."""
    pool = {k: v[metric] for k, v in cv_results.items() if k != exclude}
    return (max if higher_is_better else min)(pool, key=pool.get)


def build_kmeans(k: int, seed: int) -> Pipeline:
    """Scaling + KMeans pipeline for the clustering features."""
    return Pipeline([("prep", build_cluster_preprocessor()),
                     ("kmeans", KMeans(n_clusters=k, n_init=20, random_state=seed))])


def compare_k(X: pd.DataFrame, seed: int) -> dict[int, dict]:
    """Fit KMeans for K = 2..6 on scaled training features; return silhouette and inertia."""
    results = {}
    for k in config.K_RANGE:
        pipe = build_kmeans(k, seed).fit(X)
        scaled = pipe.named_steps["prep"].transform(X)
        labels = pipe.named_steps["kmeans"].labels_
        results[k] = {
            "silhouette": round(float(silhouette_score(scaled, labels)), 6),
            "inertia": round(float(pipe.named_steps["kmeans"].inertia_), 4),
            "cluster_sizes": [int(s) for s in np.bincount(labels, minlength=k)],
        }
    return results
