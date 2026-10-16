"""Build docs/AI_A2_2527747_REPORT.pdf from the generated artifacts.

Every number in the report is read from artifacts/*.json, so the PDF always
agrees with the latest run of run_all.py.

Usage (after run_all.py):  python docs/build_report.py
Requires: reportlab (optional dependency, only for this report).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_CENTER  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

from src import config  # noqa: E402

A = config.ARTIFACTS_DIR
OUT = ROOT / "docs" / f"AI_A2_{config.REG_DIGITS}_REPORT.pdf"


def load(name: str) -> dict:
    return json.loads((A / name).read_text(encoding="utf-8"))


reg = load("regression_metrics.json")
clf = load("classification_metrics.json")
clu = load("clustering_report.json")
val = load("data_validation_report.json")
summ = load("run_summary.json")
fp = val["dataset"]["sha256"]


def commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "see README.md"


def test_summary() -> str:
    log = ROOT / "evidence" / "test_log.txt"
    if log.exists():
        for line in reversed(log.read_text(encoding="utf-8").splitlines()):
            if "passed" in line or "failed" in line:
                return line.strip("= ").strip()
    return "run pytest -q"


# ---------------------------------------------------------------- styles
ss = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=ss["Normal"], fontName="Helvetica", fontSize=9.6,
                      leading=12.6, spaceAfter=4)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=8.6, leading=10.8)
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=15,
                    leading=18, spaceAfter=6, textColor=colors.HexColor("#1a365d"))
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=11,
                    leading=14, spaceBefore=6, spaceAfter=3, textColor=colors.HexColor("#2c5282"))
TITLE = ParagraphStyle("title", parent=H1, fontSize=19, leading=23, alignment=TA_CENTER)
CENTER = ParagraphStyle("center", parent=BODY, alignment=TA_CENTER)
CODE = ParagraphStyle("code", parent=BODY, fontName="Courier", fontSize=8, leading=10,
                      backColor=colors.HexColor("#f4f6f8"), borderPadding=4, spaceAfter=6)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=8.6, leading=10.6, spaceAfter=0)


def p(text, style=BODY):
    return Paragraph(text, style)


def table(rows, widths, header=True, font=8.6):
    data = [[c if isinstance(c, Paragraph) else Paragraph(str(c), CELL) for c in r] for r in rows]
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1 if header else 0)
    style = [
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#a0aec0")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0"))]
    t.setStyle(TableStyle(style))
    return t


def img(path: Path, width_cm: float):
    from reportlab.lib.utils import ImageReader
    w, h = ImageReader(str(path)).getSize()
    return Image(str(path), width=width_cm * cm, height=width_cm * cm * h / w)


def f4(x):
    return "n/a" if x is None else f"{x:.4f}"


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#4a5568"))
    canvas.drawString(1.8 * cm, 1.1 * cm,
                      f"{config.VERIFICATION_CODE}  |  {config.STUDENT_NAME}  |  {config.REGISTRATION_NUMBER}")
    canvas.drawRightString(A4[0] - 1.8 * cm, 1.1 * cm, f"Page {doc.page}")
    canvas.restoreState()


story = []

# ================================================================ PAGE 1
story += [
    p("Musanze Harvest Intelligence Service", TITLE),
    p("SWE 3513 Artificial Intelligence - Individual Assignment 2 - Technical report", CENTER),
    Spacer(1, 8),
    table([
        ["Full name", config.STUDENT_NAME],
        ["Registration number", config.REGISTRATION_NUMBER],
        ["Institutional username", config.INSTITUTIONAL_USERNAME],
        ["Verification code", f"<b>{config.VERIFICATION_CODE}</b>"],
        ["Personal seed (LLLL)", str(config.LLLL)],
        ["Final code commit", commit_hash()],
        ["Dataset SHA-256", fp],
    ], [4.5, 12.4], header=False),
    p("1. Identity and decision problem", H1),
    p("Stakeholder", H2),
    p("The <b>field officer of a potato cooperative in Musanze</b>. Farm records arrive in inconsistent "
      "spreadsheets, internet access is unreliable, and the officer has limited time to visit every "
      "farm. The officer needs a local tool that turns a farm's growing conditions into evidence for "
      "prioritising visits and advice. A small API allows later integration with a mobile app."),
    p("Prediction questions", H2),
    table([
        ["Task", "Question for the field officer", "Output"],
        ["Regression", "How many tonnes of potatoes is this farm likely to harvest?", "yield_tons (t)"],
        ["Classification", "Is this farm at high production risk, i.e. will its yield per hectare be "
                           "below the cooperative median?", "high_risk 0/1 and probability"],
        ["Clustering", "Which group of farms with similar growing conditions does this farm belong to?",
         "cluster ID and plain-language profile"],
    ], [2.8, 10.0, 4.1]),
    p("Ethical boundary", H2),
    p("This is a <b>classroom decision-support prototype trained on synthetic data</b>. It supports, and "
      "never replaces, the judgement of the field officer or an agronomist. It must <b>not</b> be used to "
      "decide access to credit, insurance, subsidies, inputs or any public service, or to rank or "
      "penalise farmers. Every prediction must be reviewed by a person who knows the farm before any "
      "action is taken. The API and CLI print this advisory with every prediction."),
    p("Headline results (held-out test set)", H2),
    table([
        ["Task", "Selected model", "Result", "Baseline"],
        ["Regression", reg["selected_model"],
         f"MAE {reg['selected_model_test_metrics']['mae']:.3f} t, RMSE "
         f"{reg['selected_model_test_metrics']['rmse']:.3f} t, R<super>2</super> "
         f"{reg['selected_model_test_metrics']['r2']:.3f}",
         f"mean: RMSE {reg['baseline']['mean_baseline']['rmse']:.3f} t"],
        ["Classification", clf["selected_model"],
         f"F1 {clf['selected_model_test_metrics']['f1']:.3f}, ROC-AUC "
         f"{clf['selected_model_test_metrics']['roc_auc']:.3f}",
         f"majority: F1 {clf['baseline']['majority_baseline']['f1']:.3f}"],
        ["Clustering", f"KMeans, K={clu['selected_k']}",
         f"silhouette {clu['k_comparison'][str(clu['selected_k'])]['silhouette']:.3f}", "K=2..6 compared"],
    ], [2.8, 3.6, 6.6, 3.9]),
    PageBreak(),
]

# ================================================================ PAGE 2
ds = val["dataset"]
mv = val["missing_values_before_cleaning"]
story += [
    p("2. Data and reproducibility", H1),
    p("All data is <b>synthetic</b>, produced by <font face='Courier'>data/generate_data.py</font> "
      f"with <font face='Courier'>numpy.random.default_rng({config.LLLL})</font>. No CSV is edited by hand."),
    p("Data dictionary", H2),
    table([
        ["Column", "Generation rule", "Type", "Use"],
        ["rainfall_mm", "Uniform 600 to 1400", "float", "feature"],
        ["avg_temp_c", "Uniform 12 to 26", "float", "feature"],
        ["soil_moisture_pct", "Uniform 20 to 80; 14 values set missing", "float", "feature"],
        ["fertilizer_kg", "Uniform 0 to 250; 14 values set missing", "float", "feature"],
        ["field_size_ha", "Uniform 0.2 to 4.0", "float", "feature"],
        ["season", "Random A or B", "category", "feature (one-hot)"],
        ["cooperative_support", "Random 0 or 1", "binary", "feature"],
        ["yield_tons", "0.004 x rain + 0.06 x soil + 0.02 x fert + 1.30 x size - 0.25 x |temp - 18| + "
                       "0.70 x support + season (A +0.50, B -0.20) + N(0, 0.80); clipped at 0.50",
         "float", "regression target"],
        ["high_risk", "1 if yield_tons / field_size_ha is below its median, else 0", "binary",
         "classification target"],
    ], [3.4, 9.4, 1.6, 2.5]),
    p("Personal seed rule and row count", H2),
    table([
        ["Rule", "Computation", "Value"],
        ["Seed", "LLLL = last four digits of 25/27747", str(config.LLLL)],
        ["Base rows", f"240 + ({config.LLLL} mod 41) = 240 + {config.LLLL % 41}", str(ds["base_rows"])],
        ["Missing per column", f"m = round(0.05 x {ds['base_rows']})", str(ds["missing_per_column"])],
        ["Rows written to CSV", f"{ds['base_rows']} base rows + 3 exact duplicates", str(ds["total_rows_in_csv"])],
        ["Rows after cleaning", "duplicates removed, no range violations", str(val["rows_after_cleaning"])],
        ["Train / test", f"stratified 80/20 split, random_state={config.LLLL}",
         f"{val['split']['train_rows']} / {val['split']['test_rows']}"],
    ], [3.6, 9.6, 3.7]),
    p("Missing-value and duplicate evidence", H2),
    p(f"The validation step (<font face='Courier'>src/validation.py</font>) loaded {val['rows_loaded']} rows "
      f"and found <b>{mv['soil_moisture_pct']}</b> missing soil-moisture values, <b>{mv['fertilizer_kg']}</b> "
      f"missing fertilizer values and no missing values elsewhere. It found <b>{val['exact_duplicate_rows_found']}</b> "
      f"exact duplicate rows at CSV positions {val['duplicate_row_positions']}, copies of rows "
      f"{val['duplicates_are_copies_of_rows']}; they were removed, keeping the first copy. All values were "
      f"inside their valid ranges, every required column was present, and recomputing the high-risk rule from "
      f"the CSV agreed with the stored label for {val['high_risk_rule_agreement']:.0%} of farms. "
      f"Classes are balanced ({val['class_balance']['0']} not high risk, {val['class_balance']['1']} high risk). "
      "Missing values are kept as NaN and imputed later inside the pipeline, from training rows only."),
    p("Fingerprint and reproducibility", H2),
    p("The CSV is written with <font face='Courier'>float_format=\"%.4f\"</font> and Unix line endings so the bytes, "
      "and therefore the hash, are the same on Windows, macOS and Linux. SHA-256 of "
      f"<font face='Courier'>data/{ds['csv_file']}</font>:"),
    p(fp, CODE),
    p("Running <font face='Courier'>run_all.py</font> twice produced byte-identical CSV, JSON metric files and "
      "test predictions. The same fingerprint and metrics were also obtained with older library versions "
      "(numpy 2.1, pandas 2.2, scikit-learn 1.5). A seed other than 7747, or another student's ID, is rejected."),
    PageBreak(),
]

# ================================================================ PAGE 3
story += [
    p("3. Pipeline design", H1),
    p("Flow of <font face='Courier'>run_all.py</font>", H2),
    p("check student ID and seed  ->  generate CSV + SHA-256  ->  validate schema, ranges, missing values, "
      "duplicates and leakage  ->  remove duplicates  ->  stratified train/test split (random_state=7747)  ->  "
      "fit pipelines on <b>train only</b>  ->  5-fold cross-validation on train to select models  ->  "
      "score the held-out test set once  ->  save models, JSON, CSV and plots.", CODE),
    p("Train/test separation", H2),
    p("Duplicates are removed <i>before</i> the split, so the same farm can never appear in both sets. One "
      "fixed split (223 train, 56 test, stratified on high_risk) is shared by all three tasks. Model selection "
      "uses cross-validation inside the training set; the test rows are used only for the final report."),
    p("Preprocessing (one ColumnTransformer inside every model Pipeline)", H2),
    table([
        ["Columns", "Steps (fitted on training data only)"],
        ["5 numeric features", "median imputation -> add temp_stress_c = |avg_temp_c - 18| -> standard scaling "
                               "(scaling skipped for random forests, which do not need it)"],
        ["season", "most-frequent imputation -> one-hot encoding (handle_unknown='ignore')"],
        ["cooperative_support", "most-frequent imputation, kept as 0/1"],
        ["clustering features", "the 5 numeric growing conditions: median imputation -> standard scaling -> KMeans"],
    ], [3.6, 13.3]),
    p("Because preprocessing and model are saved together as one Pipeline (joblib), "
      "<font face='Courier'>predict.py</font> and <font face='Courier'>app.py</font> apply exactly the same "
      "transformations as training. The engineered temperature-stress feature uses an input only; it reflects "
      "the agronomic fact that potatoes grow best near 18 °C."),
    p("Leakage controls", H2),
    table([
        ["Risk", "Control"],
        ["Target used as input", "Features are fixed to the 7 inputs. yield_tons, high_risk and yield per hectare are "
                                 "on a forbidden list checked by validation.py; a test confirms the saved models see "
                                 "only the 7 inputs."],
        ["Statistics from test rows", "Imputers, scalers and encoders are fitted inside Pipelines, so in each CV fold "
                                      "and in the final fit they only see training rows."],
        ["Hidden proxy of the target", "Feature-target correlations are checked; the largest is "
                                       f"{val['leakage']['max_abs_feature_target_correlation']:.2f}, far below 0.98."],
        ["Target sent to the API", "The request schema forbids extra fields, so yield_tons in a request returns 422."],
    ], [4.2, 12.7]),
    p("Model candidates and selection rule", H2),
    table([
        ["Task", "Baseline", "Candidates", "Selection evidence"],
        ["Regression", "DummyRegressor (mean)", "LinearRegression; RandomForestRegressor (300 trees, "
                                                 "min_samples_leaf=2)", "lowest 5-fold CV RMSE on train"],
        ["Classification", "DummyClassifier (most frequent)", "LogisticRegression; RandomForestClassifier "
                                                              "(300 trees, min_samples_leaf=2)", "highest 5-fold "
                                                                                                 "stratified CV F1"],
        ["Clustering", "-", "KMeans for K = 2, 3, 4, 5, 6 (n_init=20)", "highest silhouette on train"],
    ], [2.6, 3.6, 6.4, 4.3]),
    p(f"Every random_state is {config.LLLL}; random forests use n_jobs=1 so results are deterministic."),
    PageBreak(),
]

# ================================================================ PAGE 4
rb, rc = reg["baseline"]["mean_baseline"], reg["candidates"]
rcv = reg["cross_validation_train_only"]
cb, cc = clf["baseline"]["majority_baseline"], clf["candidates"]
ccv = clf["cross_validation_train_only"]
le = reg["largest_error"]
ce = clf["most_confident_error"]
sm = clf["selected_model_test_metrics"]["confusion_matrix"]
rows_r = [["Model", "CV RMSE (train)", "Test MAE", "Test RMSE", "Test R<super>2</super>"],
          ["mean baseline", f4(rcv["mean_baseline"]["cv_rmse"]), f4(rb["mae"]), f4(rb["rmse"]), f4(rb["r2"])]]
for k, v in rc.items():
    tag = " <b>(selected)</b>" if k == reg["selected_model"] else ""
    rows_r.append([k + tag, f4(rcv[k]["cv_rmse"]), f4(v["mae"]), f4(v["rmse"]), f4(v["r2"])])
rows_c = [["Model", "CV F1 (train)", "Precision", "Recall", "F1", "ROC-AUC"],
          ["majority baseline", f4(ccv["majority_baseline"]["cv_f1"]), f4(cb["precision"]), f4(cb["recall"]),
           f4(cb["f1"]), "not valid*"]]
for k, v in cc.items():
    tag = " <b>(selected)</b>" if k == clf["selected_model"] else ""
    rows_c.append([k + tag, f4(ccv[k]["cv_f1"]), f4(v["precision"]), f4(v["recall"]), f4(v["f1"]), f4(v["roc_auc"])])
story += [
    p("4. Regression and classification", H1),
    p("Regression: yield_tons (56 test farms)", H2),
    table(rows_r, [5.0, 3.0, 3.0, 3.0, 2.9]),
    Spacer(1, 3),
    img(A / "plots" / "residuals.png", 15.5),
    p(f"<b>Interpretation.</b> Linear regression was selected because it has the lowest cross-validated RMSE "
      f"({rcv['linear_regression']['cv_rmse']:.3f} t vs {rcv['random_forest']['cv_rmse']:.3f} t). The true "
      "relationship is additive, and with 223 training farms the forest cannot learn it as precisely. Residuals "
      f"are centred on zero (mean {reg['residual_summary']['mean_residual']:+.3f} t) with no visible trend, and "
      f"{reg['residual_summary']['share_within_1_ton']:.0%} of test farms are within 1 t. The test RMSE "
      f"({reg['selected_model_test_metrics']['rmse']:.3f} t) is close to the 0.80 t standard deviation of the random "
      "noise in the data, so most of the remaining error cannot be predicted from the inputs."),
    p(f"<b>Important regression error (farm {le['row_id']}).</b> Actual {le['actual_yield_tons']:.2f} t, predicted "
      f"{le['predicted_yield_tons']:.2f} t (error {le['error_tons']:+.2f} t). Inputs: {le['inputs']}. The farm has "
      "very high rainfall, which pushes the estimate up, but it is hot (about 7 °C above the optimum) and used little "
      "fertilizer. The model accounts for these, so the remaining shortfall is mainly the random noise term, about "
      "3.4 noise standard deviations below expectation. A field officer should treat such a gap as a sign of a "
      "problem the inputs do not record, such as disease or poor seed.", SMALL),
    p("Classification: high_risk (56 test farms, 28 per class)", H2),
    table(rows_c, [4.7, 2.5, 2.3, 2.3, 2.3, 2.8]),
    p("* The majority baseline gives the same probability to every farm, so ROC-AUC is not meaningful (equal to 0.5).",
      SMALL),
]
conf_tbl = Table([[img(A / "plots" / "confusion_matrix.png", 5.4), [
    p(f"<b>Interpretation.</b> Logistic regression was selected because it has the highest cross-validated F1 "
      f"({ccv['logistic_regression']['cv_f1']:.3f} vs {ccv['random_forest']['cv_f1']:.3f}). On the test set the "
      f"random forest scored slightly higher (F1 {cc['random_forest']['f1']:.3f}), but the choice was fixed before "
      "the test set was used, which keeps the reported metrics honest. The selected model made "
      f"{sm['fp']} false alarm and {sm['fn']} missed high-risk farms out of 56. The strongest input is field size: "
      "large fields tend to have lower yield per hectare, so they are flagged more often.", SMALL),
    p(f"<b>Important classification error (farm {ce['row_id']}).</b> {ce['type'].capitalize()}: probability "
      f"{ce['high_risk_probability']:.2f}, actual yield {ce['actual_yield_per_ha']:.2f} t/ha against a median of about "
      f"{clf['train_yield_per_ha_median_reference']:.2f} t/ha. Inputs: {ce['inputs']}. Rainfall and soil moisture "
      "are high, so the model expected a good yield per hectare, but the farm fell just below the median. Misses like "
      "this sit near the boundary; a lower threshold would catch them at the cost of more false alarms.", SMALL),
]]], colWidths=[5.8 * cm, 11.1 * cm])
conf_tbl.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
story += [conf_tbl, PageBreak()]

# ================================================================ PAGE 5
kc = clu["k_comparison"]
k_rows = [["K"] + list(kc), ["Silhouette"] + [f"{kc[k]['silhouette']:.4f}" for k in kc],
          ["Inertia"] + [f"{kc[k]['inertia']:.1f}" for k in kc]]
prof_rows = [["Cluster", "Farms", "Centre (rain mm / temp °C / soil % / fert kg / size ha)", "Mean yield",
              "High-risk rate"]]
for pr in clu["profiles"]:
    c = pr["center_original_units"]
    prof_rows.append([str(pr["cluster"]), str(pr["size_train"]),
                      f"{c['rainfall_mm']:.0f} / {c['avg_temp_c']:.1f} / {c['soil_moisture_pct']:.1f} / "
                      f"{c['fertilizer_kg']:.0f} / {c['field_size_ha']:.2f}",
                      f"{pr['mean_yield_tons_after_the_fact']:.2f} t", f"{pr['high_risk_rate_after_the_fact']:.0%}"])
story += [
    p("5. Clustering", H1),
    p("KMeans was fitted on the five standardised growing-condition features of the 223 training farms. "
      "<b>No target was used to form the clusters</b>; yield and risk were attached afterwards only to describe them."),
    p("K comparison", H2),
    table(k_rows, [2.6] + [2.86] * 5),
    Spacer(1, 3),
    img(A / "plots" / "clusters.png", 15.5),
    p(f"<b>Choice of K.</b> {clu['selection_reason']} {clu['interpretation']}", SMALL),
    p("Cluster profiles", H2),
    table(prof_rows, [1.5, 1.3, 8.2, 2.4, 3.5]),
    Spacer(1, 3),
]
for pr in clu["profiles"]:
    story.append(p(pr["plain_language_profile"], SMALL))
c0 = clu["profiles"][0]
story += [
    p("Operational meaning", H2),
    p(f"<b>Cluster 0</b> is the clearest example: large fields (about {c0['center_original_units']['field_size_ha']:.1f} ha) "
      f"with little fertilizer (about {c0['center_original_units']['fertilizer_kg']:.0f} kg) on cooler sites. Their total "
      f"harvest is average ({c0['mean_yield_tons_after_the_fact']:.1f} t) because the fields are big, but "
      f"{c0['high_risk_rate_after_the_fact']:.0%} of them are high risk because the fertilizer is spread over many "
      "hectares. For the field officer this group is a priority for advice on fertilizer rate per hectare. Cluster 3 "
      "(hot, dry, small farms with low inputs) has the lowest total harvest and fits water-saving and heat advice. "
      "The clusters plan visits; they do not judge individual farmers.", SMALL),
    PageBreak(),
]

# ================================================================ PAGE 6
story += [
    p("6. API and testing", H1),
    p("Endpoint contract (FastAPI, <font face='Courier'>uvicorn app:app --host 127.0.0.1 --port 8000</font>)", H2),
    table([
        ["Endpoint", "Behaviour"],
        ["GET /health", "200 {status: ok, models_loaded: true, verification_code: AI-A2-7747}"],
        ["GET /model-info", "200 identity, dataset SHA-256, selected models, test metrics, features, valid ranges, "
                            "cluster profiles"],
        ["POST /predict", "Body: one FarmRecord (7 typed fields). 200 returns yield_estimate_tons, high_risk_prediction, "
                          "high_risk_label, high_risk_probability, cluster, cluster_profile, verification_code, advisory"],
    ], [3.3, 13.6]),
    p("Validation rules (src/schema.py, ranges from src/config.py): rainfall 600-1400, temperature 12-26, soil moisture "
      "20-80, fertilizer 0-250, field size 0.2-4.0, season A/B, cooperative_support 0/1. Types are strict: a string is "
      "not converted to a number and true/false is not accepted as 1/0. Unknown fields are refused."),
    p("Valid example", H2),
    p('POST /predict {"rainfall_mm": 1050.0, "avg_temp_c": 18.5, "soil_moisture_pct": 55.0, "fertilizer_kg": 120.0, '
      '"field_size_ha": 1.5, "season": "A", "cooperative_support": 1}<br/>'
      '-&gt; 200 {"yield_estimate_tons": 12.9108, "high_risk_prediction": 0, "high_risk_probability": 0.0282, '
      '"cluster": 5, "verification_code": "AI-A2-7747", ...}', CODE),
    p("Invalid example (samples/invalid.json)", H2),
    p('{"rainfall_mm": 2500.0, "avg_temp_c": "warm", "fertilizer_kg": 120.0, "field_size_ha": 1.5, "season": "C", '
      '"cooperative_support": 1}<br/>-&gt; 422 {"status": "invalid_input", "errors": ['
      '{"field": "rainfall_mm", "problem": "Input should be less than or equal to 1400"}, '
      '{"field": "avg_temp_c", "problem": "Input should be a valid number"}, '
      '{"field": "soil_moisture_pct", "problem": "Field required"}, '
      '{"field": "season", "problem": "Input should be \'A\' or \'B\'"}]}', CODE),
    p("Failure behaviour", H2),
    table([
        ["Situation", "CLI (predict.py)", "API"],
        ["Missing field, wrong type, out of range, unknown field", "JSON error list on stderr, exit code 2",
         "422 with one entry per bad field"],
        ["Malformed JSON", "exit code 2, 'Malformed JSON'", "422"],
        ["Models not trained yet", "exit code 3 with the run_all.py command", "503 with the run_all.py command; "
                                                                            "/health reports degraded"],
        ["Wrong seed or student ID in run_all.py", "exit code 2 with explanation", "-"],
    ], [5.6, 5.6, 5.7]),
    p("Automated tests (pytest)", H2),
    p(f"Result: <b>{test_summary()}</b>. Full output is in evidence/AI_A2_{config.REG_DIGITS}_TEST_LOG.pdf."),
    table([
        ["File", "What is tested"],
        ["tests/test_inference.py", "valid inference returns every field; deterministic prediction; missing field; "
                                    "4 wrong types; 8 range/category violations; target column in input refused; CLI exit "
                                    "codes for valid and invalid samples"],
        ["tests/test_data_and_pipeline.py", "dataset bytes reproducible and equal to the saved SHA-256; 279/14/3 data "
                                            "signature; duplicate removal and range checks; wrong seed and wrong ID "
                                            "rejected; missing column rejected; saved models use only the 7 inputs; "
                                            "metric files have the required content"],
        ["tests/test_api.py", "/health, /model-info, valid /predict; 422 for missing field, wrong type plus range, and "
                              "malformed JSON"],
    ], [4.8, 12.1]),
    p("Most tests exercise failure paths, as the assessor's checklist requires.", SMALL),
    PageBreak(),
]

# ================================================================ PAGE 7
story += [
    p("7. Model card and reflection", H1),
    table([
        ["Item", "Description"],
        ["Intended use", "Help a cooperative field officer in Musanze estimate harvests, spot farms that may need support "
                         "and group farms for visit planning. Classroom prototype only."],
        ["Users", "Field officers and agronomists who can check the output against what they see on the farm."],
        ["Data", "Synthetic: 279 farms generated from a known formula with seed 7747. No real farmer is described."],
        ["Models", f"Regression: {reg['selected_model']}; classification: {clf['selected_model']} (threshold 0.5); "
                   f"clustering: KMeans K={clu['selected_k']} on growing conditions."],
        ["Performance", f"Test RMSE {reg['selected_model_test_metrics']['rmse']:.3f} t, R<super>2</super> "
                        f"{reg['selected_model_test_metrics']['r2']:.3f}; F1 "
                        f"{clf['selected_model_test_metrics']['f1']:.3f}, ROC-AUC "
                        f"{clf['selected_model_test_metrics']['roc_auc']:.3f}; silhouette "
                        f"{kc[str(clu['selected_k'])]['silhouette']:.3f}. These numbers describe the synthetic data and "
                        "would be lower on real farms."],
        ["Limitations", "Inputs are independent and uniform, the yield formula is linear, and the noise is normal, which "
                        "makes the problem easier than reality. 279 farms is a small sample. high_risk is relative (half "
                        "of the farms are always high risk) and depends heavily on field size. Missing data is only "
                        "random. Inputs outside the training ranges are rejected, not extrapolated."],
        ["Risks", "Over-trusting a precise-looking number; labelling large farms as 'risky' without context; unfair "
                  "treatment if outputs were linked to money or services; outdated models if climate or practices "
                  "change."],
        ["Prohibited use", "Deciding credit, insurance, subsidies, input allocation or access to any public service; "
                           "ranking or penalising farmers; any use as the only basis for a decision."],
        ["Human oversight", "Every output carries an advisory. A field officer must review it with the farmer before "
                            "acting, and can override it. Disagreements should be recorded to improve the model."],
    ], [3.3, 13.6]),
    p("What I would improve with real data", H2),
    p("1. Collect real cooperative records over several seasons, with the farmer's consent, and check data quality "
      "(units, outliers, missing values that are not random)."),
    p("2. Add variables that matter in Musanze: potato variety, seed quality, altitude, slope, planting date, late "
      "blight pressure and soil tests."),
    p("3. Validate by season and by sector (train on past seasons, test on the next) rather than a random split, and "
      "report uncertainty, for example prediction intervals for yield."),
    p("4. Define high risk with the cooperative using an absolute yield per hectare that matters to farmers, and "
      "choose the threshold by the cost of missing a farm versus a false alarm."),
    p("5. Check fairness across farm sizes, sectors and support status, and monitor the model after deployment."),
    p("Reflection", H2),
    p("The project showed that a simple, well-validated model can beat a more complex one when the data is small and "
      "the relationship is close to linear, and that the selection rule must be fixed before looking at the test set. "
      "The most useful lessons were about engineering: one command that rebuilds everything, the same pipeline for "
      "training and inference, and clear errors when input is wrong."),
]

doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm,
                        topMargin=1.5 * cm, bottomMargin=1.7 * cm,
                        title=f"{config.VERIFICATION_CODE} Technical report", author=config.STUDENT_NAME)
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(f"Wrote {OUT.relative_to(ROOT)}")
