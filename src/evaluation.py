"""Held-out metrics, error analysis, cluster profiles and plots."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless, works offline and on servers
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    confusion_matrix, f1_score, mean_absolute_error, mean_squared_error,
    precision_score, r2_score, recall_score, roc_auc_score,
)

from src import config  # noqa: E402


def regression_metrics(y_true, y_pred) -> dict:
    """MAE, RMSE and R2 on the held-out test set."""
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 6),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 6),
        "r2": round(float(r2_score(y_true, y_pred)), 6),
    }


def classification_metrics(y_true, y_pred, y_prob) -> dict:
    """Precision, recall, F1 (positive class = high risk), confusion matrix, ROC-AUC.

    ROC-AUC is only reported when the model produces scores that vary; a
    constant-score baseline gets ``null`` with an explanation.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    out = {
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        "accuracy": round(float((np.asarray(y_true) == np.asarray(y_pred)).mean()), 6),
        "confusion_matrix": {
            "labels": ["0_not_high_risk", "1_high_risk"],
            "matrix": cm.tolist(),
            "tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
        },
    }
    if len(np.unique(y_prob)) > 1 and len(np.unique(y_true)) > 1:
        out["roc_auc"] = round(float(roc_auc_score(y_true, y_prob)), 6)
        out["roc_auc_note"] = "valid"
    else:
        out["roc_auc"] = None
        out["roc_auc_note"] = "not valid: model outputs a constant probability for every farm (equivalent to 0.5)"
    return out


def describe_row(row: pd.Series) -> str:
    """Short plain description of one farm's inputs (for error analysis)."""
    def fmt(v, spec):
        return "missing" if pd.isna(v) else format(v, spec)
    return (
        f"rainfall {fmt(row['rainfall_mm'], '.0f')} mm, temp {fmt(row['avg_temp_c'], '.1f')} C, "
        f"soil moisture {fmt(row['soil_moisture_pct'], '.1f')} %, fertilizer "
        f"{fmt(row['fertilizer_kg'], '.0f')} kg, field {fmt(row['field_size_ha'], '.2f')} ha, "
        f"season {row['season']}, cooperative support {int(row['cooperative_support'])}"
    )


def cluster_profiles(X_train: pd.DataFrame, labels: np.ndarray, train_targets: pd.DataFrame,
                     centers_original: np.ndarray) -> list[dict]:
    """Plain-language profile for every cluster.

    Clusters are fitted on growing-condition features only. Yield and risk are
    attached *afterwards* to describe each group; they were never used to
    form the clusters.
    """
    overall_mean = X_train[config.CLUSTER_FEATURES].mean()
    overall_std = X_train[config.CLUSTER_FEATURES].std()
    names = {
        "rainfall_mm": "rainfall", "avg_temp_c": "temperature",
        "soil_moisture_pct": "soil moisture", "fertilizer_kg": "fertilizer use",
        "field_size_ha": "field size",
    }
    profiles = []
    for c in range(len(centers_original)):
        mask = labels == c
        center = dict(zip(config.CLUSTER_FEATURES, centers_original[c]))
        traits = []
        for f in config.CLUSTER_FEATURES:
            z = (center[f] - overall_mean[f]) / overall_std[f]
            if z >= 0.5:
                traits.append(f"high {names[f]}")
            elif z <= -0.5:
                traits.append(f"low {names[f]}")
        members = X_train.loc[mask]
        tgt = train_targets.loc[mask]
        mean_yield = float(tgt[config.REGRESSION_TARGET].mean())
        risk_rate = float(tgt[config.CLASSIFICATION_TARGET].mean())
        overall_risk = float(train_targets[config.CLASSIFICATION_TARGET].mean())
        trait_text = ", ".join(traits) if traits else "average conditions on every feature"
        risk_word = ("more often" if risk_rate > overall_risk + 0.05
                     else "less often" if risk_rate < overall_risk - 0.05 else "about as often")
        profiles.append({
            "cluster": c,
            "size_train": int(mask.sum()),
            "center_original_units": {k: round(float(v), 4) for k, v in center.items()},
            "season_A_share": round(float((members["season"] == "A").mean()), 4),
            "cooperative_support_share": round(float(members["cooperative_support"].mean()), 4),
            "mean_yield_tons_after_the_fact": round(mean_yield, 4),
            "high_risk_rate_after_the_fact": round(risk_rate, 4),
            "defining_traits": traits,
            "plain_language_profile": (
                f"Cluster {c} ({int(mask.sum())} training farms): {trait_text}. "
                f"Average harvest {mean_yield:.2f} t; these farms are high-risk {risk_word} "
                f"than the cooperative average ({risk_rate:.0%} vs {overall_risk:.0%})."
            ),
        })
    return profiles


def plot_residuals(y_true, y_pred, model_name: str, path: Path) -> None:
    """Predicted-vs-actual and residual-vs-predicted panels."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    resid = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    axes[0].scatter(y_true, y_pred, s=28, alpha=0.8, color="#2b6cb0", edgecolor="white")
    axes[0].plot([lo, hi], [lo, hi], "--", color="#555", lw=1)
    axes[0].set(xlabel="Actual yield (t)", ylabel="Predicted yield (t)",
                title="Predicted vs actual (test set)")
    axes[1].scatter(y_pred, resid, s=28, alpha=0.8, color="#c05621", edgecolor="white")
    axes[1].axhline(0, color="#555", lw=1, ls="--")
    worst = int(np.argmax(np.abs(resid)))
    axes[1].annotate("largest error", (y_pred[worst], resid[worst]),
                     xytext=(8, 8), textcoords="offset points", fontsize=9)
    axes[1].set(xlabel="Predicted yield (t)", ylabel="Residual = actual - predicted (t)",
                title="Residuals (test set)")
    fig.suptitle(f"Regression: {model_name}  |  {config.VERIFICATION_CODE}", fontsize=11)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_confusion(cm: np.ndarray, model_name: str, path: Path) -> None:
    """Annotated confusion-matrix heatmap."""
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(cm, cmap="Blues")
    labels = ["Not high risk (0)", "High risk (1)"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels, rotation=90, va="center")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=16,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set(xlabel="Predicted", ylabel="Actual",
           title=f"Confusion matrix: {model_name}\n{config.VERIFICATION_CODE} (test set)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_clusters(scaled_train: np.ndarray, labels: np.ndarray, scaled_centers: np.ndarray,
                  k_scores: dict, selected_k: int, seed: int, path: Path) -> None:
    """Silhouette-by-K bar chart and PCA view of the selected clusters."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ks = list(k_scores)
    sil = [k_scores[k]["silhouette"] for k in ks]
    colors = ["#2b6cb0" if k == selected_k else "#a0aec0" for k in ks]
    axes[0].bar([str(k) for k in ks], sil, color=colors)
    for k, s in zip(ks, sil):
        axes[0].text(str(k), s, f"{s:.3f}", ha="center", va="bottom", fontsize=9)
    axes[0].set(xlabel="K", ylabel="Silhouette score", title="Silhouette by K (training farms)")
    axes[0].grid(axis="y", alpha=0.3)

    pca = PCA(n_components=2, random_state=seed).fit(scaled_train)
    pts = pca.transform(scaled_train)
    cts = pca.transform(scaled_centers)
    palette = ["#2b6cb0", "#c05621", "#2f855a", "#805ad5", "#b7791f", "#c53030"]
    for c in range(selected_k):
        m = labels == c
        axes[1].scatter(pts[m, 0], pts[m, 1], s=22, alpha=0.75, color=palette[c],
                        label=f"Cluster {c} (n={m.sum()})", edgecolor="white", lw=0.4)
    axes[1].scatter(cts[:, 0], cts[:, 1], marker="X", s=170, color="black", label="Centers")
    ev = pca.explained_variance_ratio_
    axes[1].set(xlabel=f"PC1 ({ev[0]:.0%} of variance)", ylabel=f"PC2 ({ev[1]:.0%})",
                title=f"Selected K={selected_k} (PCA projection)")
    axes[1].legend(fontsize=8, loc="best")
    axes[1].grid(alpha=0.3)
    fig.suptitle(f"Farm clustering  |  {config.VERIFICATION_CODE}", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
