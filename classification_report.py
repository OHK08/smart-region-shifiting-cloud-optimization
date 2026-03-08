"""
classification_report.py
=========================
Loads the saved model and splits produced by main.py, then
generates a complete evaluation report.

Metrics covered
  Classification : Accuracy, Precision, Recall, F1 (macro / weighted / per-class)
  Regression     : MAE, MSE, RMSE, R2  (label integers used as ordinal proxy)
  Probabilistic  : Log-Loss
  Agreement      : Cohen kappa, MCC
  Discrimination : ROC-AUC (OvR, weighted + macro)

Outputs
  outputs/classification_report.txt   -- plain-text report (UTF-8)
  outputs/classification_report.png   -- two-panel figure

Run:  python classification_report.py
      (requires main.py to have been run first)
"""

import os
import sys
import warnings
import joblib

import numpy  as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors  import LinearSegmentedColormap

from sklearn.metrics import (
    accuracy_score,
    precision_score, recall_score, f1_score,
    classification_report,
    log_loss,
    cohen_kappa_score,
    matthews_corrcoef,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUT_ROOT   = "outputs"
SPLIT_DIR  = os.path.join(OUT_ROOT, "splits")
MODEL_PATH = os.path.join(OUT_ROOT, "model.pkl")
TXT_PATH   = os.path.join(OUT_ROOT, "classification_report.txt")
PNG_PATH   = os.path.join(OUT_ROOT, "classification_report.png")
os.makedirs(OUT_ROOT, exist_ok=True)

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "axes.axisbelow":    True,
    "grid.alpha":        0.28,
    "grid.linewidth":    0.55,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "legend.framealpha": 0.88,
})

RPAL = {
    "ap-south-1":     "#c0392b",
    "ap-southeast-1": "#e67e22",
    "eu-central-1":   "#2980b9",
    "eu-north-1":     "#27ae60",
    "sa-east-1":      "#1abc9c",
    "us-east-1":      "#e74c3c",
    "us-west-2":      "#2ecc71",
}

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
print("=" * 65)
print("  EXTENDED CLASSIFICATION REPORT")
print("=" * 65)

if not os.path.exists(MODEL_PATH):
    sys.exit(f"[ERROR] Model not found at {MODEL_PATH}. Run main.py first.")
if not os.path.exists(os.path.join(SPLIT_DIR, "splits.pkl")):
    sys.exit(f"[ERROR] splits.pkl not found. Run main.py first.")

bundle      = joblib.load(MODEL_PATH)
model       = bundle["model"]
le          = bundle["label_encoder"]
CLASS_NAMES = bundle["classes"]
N_CLS       = len(CLASS_NAMES)
CLS_COLORS  = [RPAL.get(c, "#95a5a6") for c in CLASS_NAMES]

splits = joblib.load(os.path.join(SPLIT_DIR, "splits.pkl"))
X_tr   = splits["X_tr"];  y_tr  = splits["y_tr"]
X_val  = splits["X_val"]; y_val = splits["y_val"]
X_te   = splits["X_te"];  y_te  = splits["y_te"]

print(f"\n  Model loaded   : {MODEL_PATH}")
print(f"  Train   n      : {len(X_tr):,}")
print(f"  Val     n      : {len(X_val):,}")
print(f"  Test    n      : {len(X_te):,}")
print(f"  Classes        : {CLASS_NAMES}\n")


# ---------------------------------------------------------------------------
# Compute metrics for one split
# ---------------------------------------------------------------------------
def compute(X_split, y_split, split_name):
    """
    Returns a dict of all metrics.

    NOTE on MAE / MSE / RMSE / R2
    --------------------------------
    Class integers (0-6) are alphabetically ordered region labels:
      0=ap-south-1, 1=ap-southeast-1, 2=eu-central-1, 3=eu-north-1,
      4=sa-east-1,  5=us-east-1,      6=us-west-2
    Treating them as ordinal values lets MAE/RMSE/R2 quantify
    *how far off* each wrong prediction is (1 rank vs 6 ranks away).
    A perfect model scores MAE=0, RMSE=0, R2=1.
    """
    yp    = model.predict(X_split)
    yprob = model.predict_proba(X_split)
    ybin  = label_binarize(y_split, classes=list(range(N_CLS)))

    mae  = mean_absolute_error(y_split, yp)
    mse  = mean_squared_error(y_split, yp)
    rmse = float(np.sqrt(mse))
    r2   = r2_score(y_split, yp)

    prec_c, rec_c, f1_c, sup_c = precision_recall_fscore_support(
        y_split, yp, zero_division=0)

    return {
        "split":     split_name,
        "y_pred":    yp,
        "y_proba":   yprob,
        "y_bin":     ybin,
        # --- scalar ---
        "acc":       accuracy_score(y_split, yp),
        "prec_mac":  precision_score(y_split, yp, average="macro",    zero_division=0),
        "prec_wt":   precision_score(y_split, yp, average="weighted", zero_division=0),
        "rec_mac":   recall_score(y_split, yp, average="macro",       zero_division=0),
        "rec_wt":    recall_score(y_split, yp, average="weighted",    zero_division=0),
        "f1_mac":    f1_score(y_split, yp, average="macro",           zero_division=0),
        "f1_wt":     f1_score(y_split, yp, average="weighted",        zero_division=0),
        "mae":       mae,
        "mse":       mse,
        "rmse":      rmse,
        "r2":        r2,
        "ll":        log_loss(y_split, yprob),
        "kappa":     cohen_kappa_score(y_split, yp),
        "mcc":       matthews_corrcoef(y_split, yp),
        "auc_w":     roc_auc_score(ybin, yprob, average="weighted", multi_class="ovr"),
        "auc_m":     roc_auc_score(ybin, yprob, average="macro",    multi_class="ovr"),
        # --- per-class ---
        "prec_c":    prec_c,
        "rec_c":     rec_c,
        "f1_c":      f1_c,
        "sup_c":     sup_c,
    }

print("  Computing metrics ...")
tr_m  = compute(X_tr,  y_tr,  "Train")
val_m = compute(X_val, y_val, "Validation")
te_m  = compute(X_te,  y_te,  "Test")
r     = te_m
print("  Done.\n")


# ---------------------------------------------------------------------------
# Build plain-text report  (ASCII-only -- Windows cp1252 safe)
# ---------------------------------------------------------------------------
SEP  = "=" * 68
SEP2 = "-" * 68
DOT  = "." * 68

def section(title):
    return [SEP2, f"  {title}", SEP2, ""]

lines = []
lines += [SEP,
          "  EXTENDED CLASSIFICATION REPORT",
          "  Green-Aware Cloud Region Recommender  |  7-Class Ensemble",
          SEP, ""]

lines.append(f"  Splits  : Train {len(X_tr):,}  |  Val {len(X_val):,}  |  Test {len(X_te):,}")
lines.append(f"  Classes : {CLASS_NAMES}")
lines.append("")

# Section A: scalar metrics
lines += section("A. SCALAR METRICS  (Train / Validation / Test)")

col_w = 34
lines.append(f"  {'Metric':<{col_w}}  {'Train':>10}  {'Val':>10}  {'Test':>10}")
lines.append("  " + "-" * 68)

rows_def = [
    ("Accuracy",                 "acc"),
    ("Precision (Macro)",        "prec_mac"),
    ("Precision (Weighted)",     "prec_wt"),
    ("Recall    (Macro)",        "rec_mac"),
    ("Recall    (Weighted)",     "rec_wt"),
    ("F1-Score  (Macro)",        "f1_mac"),
    ("F1-Score  (Weighted)",     "f1_wt"),
    (None, None),
    ("MAE  (ordinal proxy)",     "mae"),
    ("MSE  (ordinal proxy)",     "mse"),
    ("RMSE (ordinal proxy)",     "rmse"),
    ("R2   (ordinal proxy)",     "r2"),
    (None, None),
    ("Log-Loss",                 "ll"),
    ("Cohen kappa",              "kappa"),
    ("Matthews CC (MCC)",        "mcc"),
    ("ROC-AUC (Weighted OvR)",   "auc_w"),
    ("ROC-AUC (Macro OvR)",      "auc_m"),
]

for label, key in rows_def:
    if key is None:
        lines.append("  " + "- " * 34)
        continue
    lines.append(
        f"  {label:<{col_w}}  "
        f"{tr_m[key]:>10.4f}  "
        f"{val_m[key]:>10.4f}  "
        f"{te_m[key]:>10.4f}"
    )
lines.append("")

# Section B: sklearn classification report
lines += section("B. PER-CLASS REPORT  (Test Set -- sklearn format)")
lines.append(
    classification_report(y_te, r["y_pred"],
                          target_names=CLASS_NAMES, digits=4)
)

# Section C: per-class detail table
lines += section("C. DETAILED PER-CLASS TABLE  (Test Set)")
lines.append(f"  {'Class':<18}  {'Precision':>10}  {'Recall':>10}"
             f"  {'F1-Score':>10}  {'Support':>8}")
lines.append("  " + "-" * 62)
for i, cls in enumerate(CLASS_NAMES):
    lines.append(
        f"  {cls:<18}  {r['prec_c'][i]:>10.4f}  {r['rec_c'][i]:>10.4f}"
        f"  {r['f1_c'][i]:>10.4f}  {r['sup_c'][i]:>8d}"
    )
lines.append("")

# Section D: note on ordinal metrics
lines += section("D. NOTE ON ORDINAL-PROXY METRICS  (MAE / MSE / RMSE / R2)")
lines += [
    "  Class labels are integers assigned alphabetically (0 to 6):",
    "    0=ap-south-1  1=ap-southeast-1  2=eu-central-1  3=eu-north-1",
    "    4=sa-east-1   5=us-east-1       6=us-west-2",
    "",
    "  These integers are used as ordinal ranks. A prediction of class 2",
    "  when the truth is class 3 yields an absolute error of 1; predicting",
    "  class 0 when the truth is 6 yields an error of 6.",
    "",
    "  Perfect model target: MAE=0, MSE=0, RMSE=0, R2=1.",
    "",
    f"  Test MAE  = {r['mae']:.4f}  (avg label-rank error per prediction)",
    f"  Test RMSE = {r['rmse']:.4f}  (penalises large misclassifications more)",
    f"  Test R2   = {r['r2']:.4f}  (1=perfect fit, 0=mean-only baseline)",
    "",
    SEP,
]

report_txt = "\n".join(lines)

# Print to console
print(report_txt)

# Save to file -- always UTF-8 to avoid Windows cp1252 codec errors
with open(TXT_PATH, "w", encoding="utf-8") as f:
    f.write(report_txt)
print(f"\n  Plain-text report saved -> {TXT_PATH}")


# ---------------------------------------------------------------------------
# Figure: two-panel (metrics table + per-class bar chart)
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(26, 18))
fig.patch.set_facecolor("#eef2f7")
fig.suptitle(
    "Extended Classification Report\n"
    "Green-Aware Cloud Region Recommender  |  7-Class Soft-Voting Ensemble  |  "
    f"Train {len(X_tr):,} / Val {len(X_val):,} / Test {len(X_te):,}",
    fontsize=15, fontweight="bold", y=0.995,
)

import matplotlib.gridspec as gridspec
gs = fig.add_gridspec(2, 1, hspace=0.42,
                      top=0.96, bottom=0.04, left=0.03, right=0.97)

# ---- TOP: metrics table ---------------------------------------------------
ax_tbl = fig.add_subplot(gs[0])
ax_tbl.axis("off")
ax_tbl.set_facecolor("#eef2f7")

col_labels_tbl = [
    "Metric", "Category",
    "Train", "Validation", "Test",
    "Dir", "Note"
]

def f4(v):
    return f"{v:.4f}"

tbl_rows = [
    # Classification
    ["Accuracy",             "Classification", f4(tr_m["acc"]),      f4(val_m["acc"]),      f4(te_m["acc"]),      "up", "Correct / Total"],
    ["Precision (Macro)",    "Classification", f4(tr_m["prec_mac"]), f4(val_m["prec_mac"]), f4(te_m["prec_mac"]), "up", "Equal class weight"],
    ["Precision (Weighted)", "Classification", f4(tr_m["prec_wt"]),  f4(val_m["prec_wt"]),  f4(te_m["prec_wt"]),  "up", "Freq-weighted"],
    ["Recall (Macro)",       "Classification", f4(tr_m["rec_mac"]),  f4(val_m["rec_mac"]),  f4(te_m["rec_mac"]),  "up", "Equal class weight"],
    ["Recall (Weighted)",    "Classification", f4(tr_m["rec_wt"]),   f4(val_m["rec_wt"]),   f4(te_m["rec_wt"]),   "up", "Freq-weighted"],
    ["F1-Score (Macro)",     "Classification", f4(tr_m["f1_mac"]),   f4(val_m["f1_mac"]),   f4(te_m["f1_mac"]),   "up", "Equal class weight"],
    ["F1-Score (Weighted)",  "Classification", f4(tr_m["f1_wt"]),    f4(val_m["f1_wt"]),    f4(te_m["f1_wt"]),    "up", "Freq-weighted"],
    # Ordinal proxy
    ["MAE  (ordinal proxy)", "Regression",     f4(tr_m["mae"]),      f4(val_m["mae"]),      f4(te_m["mae"]),      "dn", "Avg rank error"],
    ["MSE  (ordinal proxy)", "Regression",     f4(tr_m["mse"]),      f4(val_m["mse"]),      f4(te_m["mse"]),      "dn", "Squared rank error"],
    ["RMSE (ordinal proxy)", "Regression",     f4(tr_m["rmse"]),     f4(val_m["rmse"]),     f4(te_m["rmse"]),     "dn", "sqrt(MSE)"],
    ["R2   (ordinal proxy)", "Regression",     f4(tr_m["r2"]),       f4(val_m["r2"]),       f4(te_m["r2"]),       "up", "1=perfect fit"],
    # Probabilistic / agreement
    ["Log-Loss",             "Probabilistic",  f4(tr_m["ll"]),       f4(val_m["ll"]),       f4(te_m["ll"]),       "dn", "Calibration"],
    ["Cohen kappa",          "Agreement",      f4(tr_m["kappa"]),    f4(val_m["kappa"]),    f4(te_m["kappa"]),    "up", "Beyond-chance agree."],
    ["Matthews CC (MCC)",    "Agreement",      f4(tr_m["mcc"]),      f4(val_m["mcc"]),      f4(te_m["mcc"]),      "up", "Balanced corr."],
    ["ROC-AUC (Weighted)",   "Discrimination", f4(tr_m["auc_w"]),    f4(val_m["auc_w"]),    f4(te_m["auc_w"]),    "up", "Freq-weighted AUC"],
    ["ROC-AUC (Macro)",      "Discrimination", f4(tr_m["auc_m"]),    f4(val_m["auc_m"]),    f4(te_m["auc_m"]),    "up", "Unweighted AUC"],
]

tbl = ax_tbl.table(
    cellText=tbl_rows, colLabels=col_labels_tbl,
    loc="center", cellLoc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9.8)
tbl.scale(1, 1.85)

# Header styling
HDR = "#1a252f"
for j in range(len(col_labels_tbl)):
    c = tbl[0, j]
    c.set_facecolor(HDR)
    c.set_edgecolor("#2c3e50")
    c.set_text_props(color="white", fontweight="bold", fontsize=10.5)

# Category colour bands
CAT_BG = {
    "Classification": "#dbeafe",
    "Regression":     "#fef9c3",
    "Probabilistic":  "#fce7f3",
    "Agreement":      "#dcfce7",
    "Discrimination": "#ffe4e6",
}
for i in range(1, len(tbl_rows) + 1):
    row = tbl_rows[i - 1]
    cat = row[1]
    bg  = CAT_BG.get(cat, "#f0f4f8")
    alt = "#ffffff" if i % 2 == 0 else bg
    for j in range(len(col_labels_tbl)):
        cell = tbl[i, j]
        cell.set_facecolor(bg if j == 1 else alt)
        cell.set_edgecolor("#c8d6e5")
    # Test column bold
    tbl[i, 4].set_text_props(fontweight="bold", color="#1a252f")
    # Direction arrow colour
    d = row[5]
    tbl[i, 5].set_text_props(
        color="#27ae60" if d == "up" else "#e74c3c",
        fontweight="bold", fontsize=12,
    )

tbl.auto_set_column_width(list(range(len(col_labels_tbl))))
ax_tbl.set_title(
    "Section 1 -- Scalar Metrics Across Splits",
    fontsize=12, fontweight="bold", pad=14, loc="left",
)

# ---- BOTTOM: per-class grouped bar chart ----------------------------------
ax_bar = fig.add_subplot(gs[1])
x = np.arange(N_CLS);  w = 0.26

bars_p = ax_bar.bar(x - w, r["prec_c"], w, label="Precision",
                    color="#2980b9", edgecolor="white", alpha=0.88)
bars_r = ax_bar.bar(x,     r["rec_c"],  w, label="Recall",
                    color="#27ae60", edgecolor="white", alpha=0.88)
bars_f = ax_bar.bar(x + w, r["f1_c"],   w, label="F1-Score",
                    color="#e74c3c", edgecolor="white", alpha=0.88)

for bars in [bars_p, bars_r, bars_f]:
    for bar in bars:
        h = bar.get_height()
        if h > 0.03:
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2, h + 0.012,
                f"{h:.2f}", ha="center", va="bottom",
                fontsize=7.5, fontweight="bold", color="#1a252f",
            )

# Support label above F1 bar
for bar, s in zip(bars_f, r["sup_c"]):
    ax_bar.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.055,
        f"n={s}", ha="center", va="bottom",
        fontsize=7, color="#666666", fontstyle="italic",
    )

ax_bar.axhline(r["f1_mac"], color="#7f8c8d", linestyle="--", linewidth=1.4,
               label=f"Macro-F1 = {r['f1_mac']:.4f}")
ax_bar.axhline(r["acc"],    color="#2c3e50", linestyle=":",  linewidth=1.2,
               label=f"Accuracy  = {r['acc']:.4f}")
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(CLASS_NAMES, rotation=30, ha="right", fontsize=10)
ax_bar.set_ylim(0, 1.18)
ax_bar.set_ylabel("Score")
ax_bar.set_title(
    "Section 2 -- Per-Class Precision / Recall / F1 (Test Set)\n"
    "n = support shown above F1 bars",
    pad=10, loc="left",
)
ax_bar.legend(
    fontsize=9.5, loc="upper right",
    handles=[
        Patch(facecolor="#2980b9", label="Precision"),
        Patch(facecolor="#27ae60", label="Recall"),
        Patch(facecolor="#e74c3c", label="F1-Score"),
        plt.Line2D([0], [0], color="#7f8c8d", linestyle="--",
                   linewidth=1.4, label=f"Macro-F1={r['f1_mac']:.4f}"),
        plt.Line2D([0], [0], color="#2c3e50", linestyle=":",
                   linewidth=1.2, label=f"Accuracy={r['acc']:.4f}"),
    ],
)

# Save figure
fig.savefig(PNG_PATH, dpi=300, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close(fig)
print(f"  Figure saved          -> {PNG_PATH}")

print(f"\n{'=' * 65}")
print(f"  Done. Outputs in: {OUT_ROOT}/")
print(f"    classification_report.txt  (UTF-8 plain text)")
print(f"    classification_report.png  (publication figure)")
print(f"{'=' * 65}\n")