# Musanze Harvest Intelligence Service

| | |
|---|---|
| **Student** | TUYISINGIZE Devotha |
| **Registration number** | 25/27747 |
| **Institutional username** | ug2527747@ines.ac.rw |
| **Verification code** | **AI-A2-7747** |
| **Course** | SWE 3513 Artificial Intelligence, Individual Assignment 2 (INES-Ruhengeri) |
| **Repository URL** | TO FILL: https://github.com/&lt;your-account&gt;/AI_A2_2527747 |
| **Final commit hash** | `18d2a89a49647a83f546bbfd649600ff660454df` |
| **Dataset SHA-256** | `82120e732b1ac8c8b2acd4d7e70b332ebf2a0e18fe402bcef29a751a94768205` |

> **The dataset is synthetic.** It is generated from the registration digits by
> `data/generate_data.py` and does not describe any real farm, farmer or cooperative.

## Problem

A potato cooperative in Musanze receives farm records in inconsistent
spreadsheets. Its field officer needs an offline, evidence-based tool that:

1. **Regression:** estimates the harvest (`yield_tons`) of a farm.
2. **Classification:** flags farms at **high production risk** (yield per hectare below the median).
3. **Clustering:** groups farms with similar growing conditions to plan field visits.

The service is a classroom **decision-support prototype**. It does not replace an
agronomist and must not be used to decide access to finance, insurance or public services.

## Personal data signature

| Item | Value |
|---|---|
| Seed (LLLL) | 7747, used as `numpy.random.default_rng(7747)` and every `random_state` |
| Base rows | 240 + (7747 mod 41) = 240 + 39 = **279** |
| Missing values | m = round(0.05 x 279) = **14** in `soil_moisture_pct` and 14 in `fertilizer_kg` |
| Duplicates | 3 exact duplicates appended, giving **282** rows in the CSV |
| CSV | `data/AI_A2_2527747.csv`, 4 decimal places, Unix line endings |
| Fingerprint | `artifacts/data_fingerprint.txt` (SHA-256 of the CSV bytes) |

## Setup and run

Tested with **Python 3.13.16 on Ubuntu 24.04.5 LTS** (Linux x86_64). Any Python 3.10+ should work.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python run_all.py --student-id ug2527747@ines.ac.rw --seed 7747
python predict.py --json samples/valid.json
python predict.py --json samples/invalid.json     # shows validation errors, exit code 2
pytest -q
uvicorn app:app --host 127.0.0.1 --port 8000      # docs at http://127.0.0.1:8000/docs
```

`run_all.py` recreates the CSV, validates and cleans it, trains all three tasks and
regenerates every artifact in about 10 seconds. A wrong seed or a different
student ID is rejected:

```text
$ python run_all.py --student-id ug2527747@ines.ac.rw --seed 1234
ERROR: Seed 1234 does not match the last four digits of 'ug2527747@ines.ac.rw' (7747).
```

Running it twice produces byte-identical CSV, JSON and prediction files.

## Results (held-out test set, 56 farms)

| Task | Selected model | Key test metrics | Baseline |
|---|---|---|---|
| Regression | Linear regression | MAE 0.726 t, RMSE 0.938 t, R² 0.865 | Mean: MAE 2.142, RMSE 2.594, R² -0.034 |
| Classification | Logistic regression | Precision 0.962, Recall 0.893, F1 0.926, ROC-AUC 0.995 | Majority: F1 0.000 |
| Clustering | KMeans, K = 6 | Silhouette 0.186 (highest of K = 2..6) | n/a |

Models were selected by **5-fold cross-validation on the training split only**
(lowest CV RMSE for regression, highest CV F1 for classification). The test set was
used once, to report the figures above.

## Project structure

```text
AI_A2_2527747/
  run_all.py              one-command reproducible pipeline
  predict.py              offline CLI prediction for one JSON record
  app.py                  FastAPI service (/health, /model-info, /predict)
  data/generate_data.py   individual data generator (seed 7747)
  src/
    config.py             identity, paths, column schema, valid ranges
    data_generation.py    generator, CSV writer, SHA-256 fingerprint
    validation.py         seed/ID, schema, range, missing, duplicate and leakage checks
    preprocessing.py      stratified split + ColumnTransformer pipelines
    modeling.py           baselines, candidates, cross-validation, K comparison
    evaluation.py         metrics, error analysis, cluster profiles, plots
    pipeline.py           orchestration and artifact writing
    schema.py             typed request/response models (pydantic)
    inference.py          loads saved models and scores one record
  tests/                  32 pytest tests (happy and failure paths)
  samples/valid.json, samples/invalid.json
  docs/                   technical report PDF (+ script that builds it)
  evidence/               test log, declaration, AI-use disclosure, demo video
```

## Generated artifacts

| Path | Content |
|---|---|
| `data/AI_A2_2527747.csv` | Raw individual dataset (with deliberate missing values and duplicates) |
| `artifacts/data_fingerprint.txt` | Verification code, row count, SHA-256 |
| `artifacts/data_validation_report.json` | Missing values, duplicate rows, range checks, leakage check, split |
| `artifacts/regression_metrics.json` | Baseline and candidate MAE/RMSE/R², CV scores, selected model and reason, largest error |
| `artifacts/classification_metrics.json` | Baseline and candidate precision/recall/F1, confusion matrices, ROC-AUC, most confident error |
| `artifacts/clustering_report.json` | Silhouette and inertia for K = 2..6, selected K, centres and profiles |
| `artifacts/test_predictions.csv` | Row ID, actual/predicted yield, errors, risk, probability, cluster |
| `artifacts/run_summary.json` | One-page summary of the run |
| `artifacts/plots/` | `residuals.png`, `confusion_matrix.png`, `clusters.png` |
| `models/` | `regression_model.joblib`, `classification_model.joblib`, `clustering_model.joblib`, `preprocessor.joblib`, `model_metadata.json` |

## Leakage and preprocessing controls

* `yield_tons`, `high_risk` and any yield-per-hectare value are never inputs; `validation.py`
  checks this, and the API rejects them as unknown fields.
* Imputation (median), scaling and one-hot encoding live inside scikit-learn `Pipeline` /
  `ColumnTransformer` objects, fitted on the training split only (`random_state=7747`, stratified 80/20).
* One engineered feature, `temp_stress_c = |avg_temp_c - 18|`, measures distance from the
  potato optimum temperature. It uses an input only.
* Duplicates are removed before the split, so no farm appears in both train and test.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | `{"status": "ok", "models_loaded": true, "verification_code": "AI-A2-7747"}` |
| GET | `/model-info` | Identity, dataset SHA-256, selected models, test metrics, features, valid ranges, cluster profiles |
| POST | `/predict` | Yield estimate, high-risk prediction, probability, cluster and verification code |

Valid ranges: rainfall 600-1400 mm, temperature 12-26 °C, soil moisture 20-80 %,
fertilizer 0-250 kg, field size 0.2-4.0 ha, season `A`/`B`, cooperative support `0`/`1`.
Types are strict (no strings for numbers, no `true`/`false` for 0/1) and unknown fields are refused.

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d @samples/valid.json
```

```json
{"verification_code": "AI-A2-7747", "yield_estimate_tons": 12.9108, "high_risk_prediction": 0,
 "high_risk_label": "not high risk", "high_risk_probability": 0.0282, "cluster": 5, "...": "..."}
```

Invalid input returns **422** with one entry per bad field:

```json
{"verification_code": "AI-A2-7747", "status": "invalid_input",
 "message": "The farm record failed validation. Fix the fields below and retry.",
 "errors": [{"field": "rainfall_mm", "problem": "Input should be less than or equal to 1400", "received": 2500.0},
            {"field": "avg_temp_c", "problem": "Input should be a valid number", "received": "warm"},
            {"field": "soil_moisture_pct", "problem": "Field required", "received": null},
            {"field": "season", "problem": "Input should be 'A' or 'B'", "received": "C"}]}
```

If the models have not been trained, `/predict` returns **503** with the command to run.

## Changing a rule

All valid ranges are in `src/config.py` (`RANGES`). Changing one value there updates data
validation, the CLI and the API together. Response fields are defined in
`src/schema.py` (`PredictionResponse`) and filled in `src/inference.py`.

## Dependencies

Only the packages named in the assignment, plus the libraries they need to run:
`pydantic` (FastAPI's typed validation), `uvicorn` (server that runs the FastAPI app) and
`httpx` (required by FastAPI's `TestClient` in the API tests). The report PDF is built by
`docs/build_report.py` with `reportlab`; this is optional and not needed to run or test the system.

## Known limitations

* The data is synthetic and generated from a known formula with independent uniform inputs.
  Real farm data would be noisier, correlated and possibly biased.
* `high_risk` is a relative label (below the median yield per hectare), so half the farms are
  always "high risk" whatever the absolute harvest. It depends strongly on field size.
* 279 farms is a small sample; metrics would vary with a different split.
* Silhouette scores are low (0.16 to 0.19). The clusters are practical segments, not natural farm types.
* Inputs outside the training ranges are rejected rather than extrapolated.

## Responsible use

Intended for a field officer to prioritise visits and discussions. Every output must be
reviewed by a person who knows the farm. Do not use it to decide credit, insurance,
subsidies or access to any public service. See the model card in the report (page 7).
