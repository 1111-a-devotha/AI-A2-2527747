"""End-to-end pipeline: generate -> validate/clean -> split -> train -> evaluate -> save."""
from __future__ import annotations

import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from src import config
from src.data_generation import write_dataset
from src.evaluation import (
    classification_metrics, cluster_profiles, describe_row, plot_clusters,
    plot_confusion, plot_residuals, regression_metrics,
)
from src.modeling import (
    build_kmeans, classification_candidates, compare_k, cross_validate_classifiers,
    cross_validate_regressors, regression_candidates, select_best,
)
from src.preprocessing import split_data
from src.validation import validate_and_clean, validate_student_id


def _write_json(path: Path, obj: dict) -> None:
    """Write JSON with stable key order and Unix line endings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8", newline="\n")


def run_regression(train, test, seed):
    """Compare baseline and candidates by CV, refit the winner, score the test set."""
    X_tr, y_tr = train[config.INPUT_FEATURES], train[config.REGRESSION_TARGET]
    X_te, y_te = test[config.INPUT_FEATURES], test[config.REGRESSION_TARGET]
    models = regression_candidates(seed)
    cv = cross_validate_regressors(models, X_tr, y_tr, seed)
    test_scores, fitted = {}, {}
    for name, model in models.items():
        fitted[name] = model.fit(X_tr, y_tr)
        test_scores[name] = regression_metrics(y_te, model.predict(X_te))
    best = select_best(cv, "cv_rmse", higher_is_better=False, exclude="mean_baseline")
    runner_up = sorted((k for k in cv if k not in ("mean_baseline", best)),
                       key=lambda k: cv[k]["cv_rmse"])[0]
    reason = (
        f"{best} has the lowest mean 5-fold cross-validated RMSE on the training split "
        f"({cv[best]['cv_rmse']:.4f} t vs {cv[runner_up]['cv_rmse']:.4f} t for {runner_up} and "
        f"{cv['mean_baseline']['cv_rmse']:.4f} t for the mean baseline). Selection used "
        f"training data only; the test set was used once to report final metrics."
    )
    return fitted[best], best, cv, test_scores, reason


def run_classification(train, test, seed):
    """Compare majority baseline and classifiers by CV F1, refit the winner, score test."""
    X_tr, y_tr = train[config.INPUT_FEATURES], train[config.CLASSIFICATION_TARGET]
    X_te, y_te = test[config.INPUT_FEATURES], test[config.CLASSIFICATION_TARGET]
    models = classification_candidates(seed)
    cv = cross_validate_classifiers(models, X_tr, y_tr, seed)
    test_scores, fitted = {}, {}
    for name, model in models.items():
        fitted[name] = model.fit(X_tr, y_tr)
        prob = model.predict_proba(X_te)[:, 1]
        test_scores[name] = classification_metrics(y_te, model.predict(X_te), prob)
    best = select_best(cv, "cv_f1", higher_is_better=True, exclude="majority_baseline")
    runner_up = [k for k in cv if k not in ("majority_baseline", best)][0]
    reason = (
        f"{best} has the highest mean 5-fold stratified cross-validated F1 for the high-risk "
        f"class on the training split ({cv[best]['cv_f1']:.4f} vs {cv[runner_up]['cv_f1']:.4f} "
        f"for {runner_up}; majority baseline {cv['majority_baseline']['cv_f1']:.4f}). F1 was "
        f"chosen because missing a high-risk farm and raising a false alarm both matter."
    )
    return fitted[best], best, cv, test_scores, reason


def run_clustering(train, test, seed):
    """Choose K by silhouette (K=2..6) on training farms and profile each cluster."""
    X_tr = train[config.CLUSTER_FEATURES]
    k_scores = compare_k(X_tr, seed)
    selected_k = max(k_scores, key=lambda k: k_scores[k]["silhouette"])
    pipe = build_kmeans(selected_k, seed).fit(X_tr)
    prep, km = pipe.named_steps["prep"], pipe.named_steps["kmeans"]
    labels = km.labels_
    centers_orig = prep.named_steps["scale"].inverse_transform(km.cluster_centers_)
    profiles = cluster_profiles(train, labels, train[config.TARGET_COLUMNS], centers_orig)
    test_labels = pipe.predict(test[config.CLUSTER_FEATURES])
    plot_clusters(prep.transform(X_tr), labels, km.cluster_centers_, k_scores,
                  selected_k, seed, config.PLOTS_DIR / "clusters.png")
    return pipe, selected_k, k_scores, profiles, test_labels


def run(student_id: str, seed: int) -> dict:
    """Run the whole reproducible pipeline and return a summary for printing."""
    validate_student_id(student_id, seed)
    for d in (config.MODELS_DIR, config.ARTIFACTS_DIR, config.PLOTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Data -----------------------------------------------------------------
    data_summary = write_dataset(seed)
    raw = pd.read_csv(config.DATA_CSV)
    clean, validation = validate_and_clean(raw)
    validation["dataset"] = data_summary
    train, test = split_data(clean, seed)
    validation["split"] = {
        "method": "train_test_split, stratified on high_risk",
        "test_size": config.TEST_SIZE, "random_state": seed,
        "train_rows": int(len(train)), "test_rows": int(len(test)),
        "test_row_ids": [int(r) for r in test["row_id"]],
    }
    _write_json(config.ARTIFACTS_DIR / "data_validation_report.json", validation)

    # 2. Regression -----------------------------------------------------------
    reg_model, reg_best, reg_cv, reg_test, reg_reason = run_regression(train, test, seed)
    reg_pred = reg_model.predict(test[config.INPUT_FEATURES])
    y_true = test[config.REGRESSION_TARGET].to_numpy()
    resid = y_true - reg_pred
    worst = int(np.argmax(np.abs(resid)))
    worst_row = test.iloc[worst]
    regression_report = {
        "verification_code": config.VERIFICATION_CODE,
        "target": config.REGRESSION_TARGET,
        "evaluation": "held-out test set",
        "test_rows": int(len(test)),
        "baseline": {"mean_baseline": reg_test["mean_baseline"]},
        "candidates": {k: v for k, v in reg_test.items() if k != "mean_baseline"},
        "cross_validation_train_only": reg_cv,
        "selected_model": reg_best,
        "selection_reason": reg_reason,
        "selected_model_test_metrics": reg_test[reg_best],
        "residual_summary": {
            "mean_residual": round(float(resid.mean()), 6),
            "std_residual": round(float(resid.std()), 6),
            "share_within_1_ton": round(float((np.abs(resid) <= 1.0).mean()), 6),
        },
        "largest_error": {
            "row_id": int(worst_row["row_id"]),
            "actual_yield_tons": round(float(y_true[worst]), 4),
            "predicted_yield_tons": round(float(reg_pred[worst]), 4),
            "error_tons": round(float(resid[worst]), 4),
            "inputs": describe_row(worst_row),
        },
    }
    _write_json(config.ARTIFACTS_DIR / "regression_metrics.json", regression_report)
    plot_residuals(y_true, reg_pred, reg_best, config.PLOTS_DIR / "residuals.png")

    # 3. Classification -------------------------------------------------------
    clf_model, clf_best, clf_cv, clf_test, clf_reason = run_classification(train, test, seed)
    X_te = test[config.INPUT_FEATURES]
    clf_prob = clf_model.predict_proba(X_te)[:, 1]
    clf_pred = clf_model.predict(X_te)
    c_true = test[config.CLASSIFICATION_TARGET].to_numpy()
    wrong = np.flatnonzero(clf_pred != c_true)
    error_example = None
    if len(wrong):
        conf = np.abs(clf_prob[wrong] - 0.5)
        idx = int(wrong[np.argmax(conf)])
        r = test.iloc[idx]
        error_example = {
            "row_id": int(r["row_id"]),
            "type": "false negative (missed high-risk farm)" if c_true[idx] == 1
                    else "false positive (false alarm)",
            "actual_high_risk": int(c_true[idx]),
            "predicted_high_risk": int(clf_pred[idx]),
            "high_risk_probability": round(float(clf_prob[idx]), 4),
            "actual_yield_tons": round(float(r[config.REGRESSION_TARGET]), 4),
            "actual_yield_per_ha": round(float(r[config.REGRESSION_TARGET] / r["field_size_ha"]), 4),
            "inputs": describe_row(r),
        }
    ypha_train = train[config.REGRESSION_TARGET] / train["field_size_ha"]
    classification_report = {
        "verification_code": config.VERIFICATION_CODE,
        "target": config.CLASSIFICATION_TARGET,
        "positive_class": "1 = high risk (yield per hectare below the median)",
        "decision_threshold": 0.5,
        "features_used": config.INPUT_FEATURES,
        "excluded_for_leakage": config.FORBIDDEN_FEATURES,
        "train_yield_per_ha_median_reference": round(float(ypha_train.median()), 4),
        "evaluation": "held-out test set",
        "test_rows": int(len(test)),
        "test_class_balance": {"0": int((c_true == 0).sum()), "1": int((c_true == 1).sum())},
        "baseline": {"majority_baseline": clf_test["majority_baseline"]},
        "candidates": {k: v for k, v in clf_test.items() if k != "majority_baseline"},
        "cross_validation_train_only": clf_cv,
        "selected_model": clf_best,
        "selection_reason": clf_reason,
        "selected_model_test_metrics": clf_test[clf_best],
        "misclassified_test_rows": int(len(wrong)),
        "most_confident_error": error_example,
    }
    _write_json(config.ARTIFACTS_DIR / "classification_metrics.json", classification_report)
    plot_confusion(np.array(clf_test[clf_best]["confusion_matrix"]["matrix"]), clf_best,
                   config.PLOTS_DIR / "confusion_matrix.png")

    # 4. Clustering -----------------------------------------------------------
    clu_model, k, k_scores, profiles, test_clusters = run_clustering(train, test, seed)
    best_sil = k_scores[k]["silhouette"]
    clustering_report = {
        "verification_code": config.VERIFICATION_CODE,
        "algorithm": "KMeans (n_init=20) on median-imputed, standardised features",
        "features_used": config.CLUSTER_FEATURES,
        "targets_used_for_fitting": False,
        "fitted_on": "training split only; test farms assigned with predict()",
        "k_comparison": {str(kk): v for kk, v in k_scores.items()},
        "selected_k": k,
        "selection_reason": (
            f"K={k} has the highest silhouette score among K=2..6 ({best_sil:.4f})."
        ),
        "interpretation": (
            f"The best silhouette ({best_sil:.3f}) is modest, which is expected because the "
            "synthetic inputs are drawn independently from uniform distributions: farms form "
            "a continuum rather than sharply separated groups. "
            + ("The score keeps rising up to the largest K tested, so K is chosen at the "
               "edge of the allowed range; the gain per extra cluster is small. "
               if k == max(config.K_RANGE) else "")
            + "The clusters are therefore useful as practical segments for planning field "
            "visits, not as natural farm types."
        ),
        "profiles": profiles,
    }
    _write_json(config.ARTIFACTS_DIR / "clustering_report.json", clustering_report)

    # 5. Test predictions -----------------------------------------------------
    preds = pd.DataFrame({
        "row_id": test["row_id"].astype(int).to_numpy(),
        "actual_yield_tons": y_true,
        "predicted_yield_tons": reg_pred,
        "yield_error_tons": resid,
        "abs_yield_error_tons": np.abs(resid),
        "actual_high_risk": c_true,
        "predicted_high_risk": clf_pred.astype(int),
        "high_risk_probability": clf_prob,
        "classification_correct": (clf_pred == c_true).astype(int),
        "cluster": test_clusters.astype(int),
    })
    preds.to_csv(config.ARTIFACTS_DIR / "test_predictions.csv", index=False,
                 float_format=config.FLOAT_FORMAT, lineterminator="\n")

    # 6. Save models ----------------------------------------------------------
    joblib.dump(reg_model, config.REGRESSION_MODEL_PATH)
    joblib.dump(clf_model, config.CLASSIFICATION_MODEL_PATH)
    joblib.dump(clu_model, config.CLUSTERING_MODEL_PATH)
    joblib.dump(reg_model.named_steps["prep"], config.PREPROCESSOR_PATH)
    metadata = {
        "verification_code": config.VERIFICATION_CODE,
        "student_name": config.STUDENT_NAME,
        "registration_number": config.REGISTRATION_NUMBER,
        "institutional_username": config.INSTITUTIONAL_USERNAME,
        "seed": seed,
        "data_sha256": data_summary["sha256"],
        "rows_after_cleaning": int(len(clean)),
        "input_features": config.INPUT_FEATURES,
        "valid_ranges": {k2: list(v) for k2, v in config.RANGES.items()},
        "season_values": list(config.SEASON_VALUES),
        "cooperative_support_values": list(config.SUPPORT_VALUES),
        "regression": {"selected_model": reg_best, "test_metrics": reg_test[reg_best],
                       "file": config.REGRESSION_MODEL_PATH.name},
        "classification": {"selected_model": clf_best,
                           "test_metrics": {m: clf_test[clf_best][m]
                                            for m in ("precision", "recall", "f1", "roc_auc")},
                           "threshold": 0.5, "file": config.CLASSIFICATION_MODEL_PATH.name},
        "clustering": {"selected_k": k, "silhouette": best_sil,
                       "features": config.CLUSTER_FEATURES,
                       "profiles": [{"cluster": p["cluster"],
                                     "plain_language_profile": p["plain_language_profile"]}
                                    for p in profiles],
                       "file": config.CLUSTERING_MODEL_PATH.name},
        "intended_use": "Classroom decision-support prototype for a Musanze potato cooperative.",
        "data_status": "synthetic",
    }
    _write_json(config.MODEL_METADATA_PATH, metadata)

    summary = {
        "verification_code": config.VERIFICATION_CODE,
        "student": f"{config.STUDENT_NAME} ({config.REGISTRATION_NUMBER})",
        "seed": seed,
        "base_rows": data_summary["base_rows"],
        "rows_in_csv": data_summary["total_rows_in_csv"],
        "rows_after_cleaning": int(len(clean)),
        "duplicates_removed": validation["exact_duplicate_rows_found"],
        "missing_values": {c: v for c, v in validation["missing_values_before_cleaning"].items() if v},
        "train_rows": int(len(train)), "test_rows": int(len(test)),
        "sha256": data_summary["sha256"],
        "regression": {"selected": reg_best, **reg_test[reg_best]},
        "classification": {"selected": clf_best,
                           **{m: clf_test[clf_best][m] for m in ("precision", "recall", "f1", "roc_auc")}},
        "clustering": {"selected_k": k, "silhouette": best_sil},
        "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                        "pandas": pd.__version__, "numpy": np.__version__},
    }
    _write_json(config.ARTIFACTS_DIR / "run_summary.json", summary)
    return summary


def artifact_list() -> list[str]:
    """Relative paths of every file under data/, models/ and artifacts/."""
    files = []
    for folder in (config.DATA_DIR, config.MODELS_DIR, config.ARTIFACTS_DIR):
        for p in sorted(folder.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                files.append(p.relative_to(config.ROOT).as_posix())
    return files
