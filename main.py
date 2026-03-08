"""
main.py -- Green-Aware Cloud Region Recommender
================================================
Trains a soft-voting ensemble (HGB + RF), saves the model,
saves train/val/test splits, and generates 6 separate
publication-quality diagnostic figures.

Output layout
  outputs/
    model.pkl                  -- trained model bundle
    splits/
      train.csv / val.csv / test.csv
      splits.pkl
    figures/
      fig1_confusion_matrix.png
      fig2_per_class_metrics.png
      fig3_cross_validation.png
      fig4_feature_importance.png
      fig5_roc_curves.png
      fig6_metrics_summary.png

Run:  python main.py
"""

import os
import warnings
import joblib

import numpy  as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors  import LinearSegmentedColormap

from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_validate)
from sklearn.preprocessing   import LabelEncoder, OrdinalEncoder, label_binarize
from sklearn.metrics         import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
    log_loss, cohen_kappa_score, matthews_corrcoef,
    roc_auc_score, roc_curve, auc,
    precision_recall_fscore_support,
)
from sklearn.ensemble import (HistGradientBoostingClassifier,
                               RandomForestClassifier, VotingClassifier)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------
OUT_ROOT   = "outputs"
FIG_DIR    = os.path.join(OUT_ROOT, "figures")
SPLIT_DIR  = os.path.join(OUT_ROOT, "splits")
MODEL_PATH = os.path.join(OUT_ROOT, "model.pkl")

for d in [OUT_ROOT, FIG_DIR, SPLIT_DIR]:
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------------------
# Global plot style
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
    "legend.edgecolor":  "#cccccc",
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


def save_fig(fig, filename):
    path = os.path.join(FIG_DIR, filename)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"    saved -> {path}")


# ===========================================================================
# 1. LOAD
# ===========================================================================
print("=" * 65)
print("  GREEN-AWARE CLOUD REGION RECOMMENDER -- Training Pipeline")
print("=" * 65)

df   = pd.read_csv("cloud_jobs_dataset.csv")
dist = df["recommended_region"].value_counts(normalize=True).mul(100)
print(f"\n  Dataset  : {len(df):,} rows x {df.shape[1]} columns")
print(f"  Classes  : {df['recommended_region'].nunique()}")
print("\n  Class distribution:")
for r, v in dist.items():
    print(f"    {r:<18s}  {v:5.1f}%")


# ===========================================================================
# 2. PREPROCESSING & FEATURE ENGINEERING
# ===========================================================================
df = pd.get_dummies(df, columns=["job_type"], prefix="jt", dtype=int)

REGION_ORDER = ["eu-north-1", "us-west-2", "sa-east-1",
                "eu-central-1", "us-east-1", "ap-southeast-1", "ap-south-1"]
oe = OrdinalEncoder(categories=[REGION_ORDER],
                    handle_unknown="use_encoded_value", unknown_value=-1)
df["cur_region_enc"] = oe.fit_transform(df[["current_region"]])

le          = LabelEncoder()
df["label"] = le.fit_transform(df["recommended_region"])
CLASS_NAMES = list(le.classes_)
N_CLS       = len(CLASS_NAMES)
CLS_COLORS  = [RPAL.get(c, "#95a5a6") for c in CLASS_NAMES]

# Engineered features
eps = 1e-6
df["urgency"]         = df["importance_level"] / (np.log1p(df["deadline_hours"]) + eps)
df["green_pressure"]  = ((1 - df["latency_sensitivity"])
                         * (0.5 + 0.5 * df["carbon_budget_strict"])
                         * (10 - df["importance_level"]) / 9)
df["carbon_load"]     = df["energy_required_kwh"] * df["current_carbon_intensity"] / 1000
df["cost_pressure"]   = df["cost_sensitivity"] * df["num_vcpus"]
df["lat_x_imp"]       = df["latency_sensitivity"] * df["importance_level"]
df["deadline_urg"]    = np.clip(1 - df["deadline_hours"] / 100, 0, 1)
df["energy_per_vcpu"] = df["energy_required_kwh"] / (df["num_vcpus"] + eps)
df["renew_x_green"]   = df["current_renewable_share"] * (1 + df["carbon_budget_strict"])
df["perf_vs_green"]   = (df["importance_level"] - 5.5) * df["latency_sensitivity"]
df["imp_sq"]          = df["importance_level"] ** 2
df["cost_x_imp"]      = df["cost_sensitivity"] * df["importance_level"]
df["dl_lat"]          = df["deadline_hours"] * df["latency_sensitivity"]

DROP     = {"job_id", "current_region", "recommended_region", "label"}
FEATURES = [c for c in df.columns if c not in DROP]
X = df[FEATURES]
y = df["label"]
print(f"\n  Feature count : {len(FEATURES)}")


# ===========================================================================
# 3. THREE-WAY STRATIFIED SPLIT  (60 / 20 / 20)
# ===========================================================================
X_temp, X_te,  y_temp, y_te  = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)
X_tr,   X_val, y_tr,   y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

print(f"\n  Split sizes:")
print(f"    Train      : {len(X_tr):>7,}  ({len(X_tr)/len(X)*100:.1f}%)")
print(f"    Validation : {len(X_val):>7,}  ({len(X_val)/len(X)*100:.1f}%)")
print(f"    Test       : {len(X_te):>7,}  ({len(X_te)/len(X)*100:.1f}%)")

# Save splits as CSV
for split_X, split_y, name in [(X_tr, y_tr, "train"),
                                (X_val, y_val, "val"),
                                (X_te,  y_te,  "test")]:
    out = split_X.copy()
    out["label"]              = split_y.values
    out["recommended_region"] = le.inverse_transform(split_y.values)
    out.to_csv(os.path.join(SPLIT_DIR, f"{name}.csv"), index=False)

# Save splits as joblib bundle (fast reload)
joblib.dump(
    {"X_tr": X_tr, "y_tr": y_tr,
     "X_val": X_val, "y_val": y_val,
     "X_te": X_te, "y_te": y_te,
     "features": FEATURES, "classes": CLASS_NAMES},
    os.path.join(SPLIT_DIR, "splits.pkl"),
)
print(f"\n  Splits saved to  {SPLIT_DIR}/")
print(f"    train.csv ({len(X_tr):,})  val.csv ({len(X_val):,})  test.csv ({len(X_te):,})  splits.pkl")


# ===========================================================================
# 4. MODEL
# ===========================================================================
hgb = HistGradientBoostingClassifier(
    max_iter=500, max_depth=8, learning_rate=0.05,
    min_samples_leaf=15, l2_regularization=0.3,
    max_features=0.85, random_state=42,
)
rf = RandomForestClassifier(
    n_estimators=400, max_depth=20, min_samples_leaf=5,
    max_features="sqrt", class_weight="balanced",
    random_state=42, n_jobs=-1,
)
model = VotingClassifier(
    estimators=[("hgb", hgb), ("rf", rf)],
    voting="soft", n_jobs=-1,
)

print("\n  Training ensemble (HGB + RandomForest) ...")
model.fit(X_tr, y_tr)
print("  Done.")

# Save model
joblib.dump(
    {"model": model, "label_encoder": le, "ordinal_encoder": oe,
     "features": FEATURES, "classes": CLASS_NAMES},
    MODEL_PATH,
)
print(f"  Model saved -> {MODEL_PATH}")


# ===========================================================================
# 5. CROSS-VALIDATION  (5-fold on training data only)
# ===========================================================================
print("\n  Running 5-fold cross-validation on training data ...")
skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_res = cross_validate(
    model, X_tr, y_tr, cv=skf, n_jobs=-1,
    scoring=["accuracy", "f1_macro", "f1_weighted",
             "neg_log_loss", "roc_auc_ovr_weighted"],
    return_train_score=True,
)
CV = {
    "Accuracy":    cv_res["test_accuracy"],
    "Macro-F1":    cv_res["test_f1_macro"],
    "Weighted-F1": cv_res["test_f1_weighted"],
    "Log-Loss":   -cv_res["test_neg_log_loss"],
    "ROC-AUC":     cv_res["test_roc_auc_ovr_weighted"],
    "Train-Acc":   cv_res["train_accuracy"],
}
print("  Done.")


# ===========================================================================
# 6. EVALUATION
# ===========================================================================
def evaluate(Xs, ys, name):
    yp     = model.predict(Xs)
    yprob  = model.predict_proba(Xs)
    ybin   = label_binarize(ys, classes=list(range(N_CLS)))
    return {
        "name":     name,
        "y_pred":   yp,
        "y_proba":  yprob,
        "y_bin":    ybin,
        "acc":      accuracy_score(ys, yp),
        "f1_mac":   f1_score(ys, yp, average="macro"),
        "f1_wt":    f1_score(ys, yp, average="weighted"),
        "prec_mac": precision_score(ys, yp, average="macro",    zero_division=0),
        "rec_mac":  recall_score(ys, yp, average="macro",       zero_division=0),
        "ll":       log_loss(ys, yprob),
        "kappa":    cohen_kappa_score(ys, yp),
        "mcc":      matthews_corrcoef(ys, yp),
        "auc_w":    roc_auc_score(ybin, yprob, average="weighted", multi_class="ovr"),
        "auc_m":    roc_auc_score(ybin, yprob, average="macro",    multi_class="ovr"),
        "prec_c":   precision_recall_fscore_support(ys, yp, zero_division=0)[0],
        "rec_c":    precision_recall_fscore_support(ys, yp, zero_division=0)[1],
        "f1_c":     precision_recall_fscore_support(ys, yp, zero_division=0)[2],
        "sup_c":    precision_recall_fscore_support(ys, yp, zero_division=0)[3],
    }

val_r  = evaluate(X_val, y_val, "Validation")
test_r = evaluate(X_te,  y_te,  "Test")
r      = test_r

gap_acc = abs(CV["Accuracy"].mean() - r["acc"])
gap_f1  = abs(CV["Macro-F1"].mean() - r["f1_mac"])

# Console summary
print(f"\n{'=' * 65}")
print("  EVALUATION RESULTS")
print(f"{'=' * 65}")
hdr = f"  {'Metric':<28s}  {'Val':>10s}  {'Test':>10s}  {'CV Mean':>10s}"
print(hdr)
print("  " + "-" * 62)
for label, vk, tk, ck in [
    ("Accuracy",           val_r["acc"],     r["acc"],     CV["Accuracy"].mean()),
    ("Macro-F1",           val_r["f1_mac"],  r["f1_mac"],  CV["Macro-F1"].mean()),
    ("Weighted-F1",        val_r["f1_wt"],   r["f1_wt"],   CV["Weighted-F1"].mean()),
    ("Macro Precision",    val_r["prec_mac"],r["prec_mac"], None),
    ("Macro Recall",       val_r["rec_mac"], r["rec_mac"],  None),
    ("Log-Loss",           val_r["ll"],      r["ll"],       CV["Log-Loss"].mean()),
    ("ROC-AUC (Weighted)", val_r["auc_w"],   r["auc_w"],    CV["ROC-AUC"].mean()),
    ("ROC-AUC (Macro)",    val_r["auc_m"],   r["auc_m"],    None),
    ("Cohen kappa",        val_r["kappa"],   r["kappa"],    None),
    ("MCC",                val_r["mcc"],     r["mcc"],      None),
]:
    cv_s = f"{ck:.4f}" if ck is not None else "---"
    print(f"  {label:<28s}  {vk:>10.4f}  {tk:>10.4f}  {cv_s:>10s}")

print(f"\n  CV-Test Gap Accuracy : {gap_acc:.4f}  "
      f"{'[OK no overfit]' if gap_acc < 0.025 else '[WARNING check model]'}")
print(f"  CV-Test Gap Macro-F1 : {gap_f1:.4f}  "
      f"{'[OK stable]'     if gap_f1  < 0.025 else '[WARNING check model]'}")
print(f"\n  Classification Report (Test Set):")
print(classification_report(y_te, r["y_pred"], target_names=CLASS_NAMES, digits=4))

# RF feature importances
rf_imp = model.named_estimators_["rf"].feature_importances_


# ===========================================================================
# 7. FIGURES
# ===========================================================================
print(f"\n  Generating figures -> {FIG_DIR}/")

# ---- Figure 1: Confusion Matrix --------------------------------------------
cm      = confusion_matrix(y_te, r["y_pred"])
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(18, 7.5))
fig.suptitle(
    "Figure 1 -- Confusion Matrix\n"
    "Green-Aware Cloud Region Recommender | 7-Class | Test Set",
    fontsize=14, fontweight="bold", y=1.01,
)
for ax, data, fmt, title in zip(
    axes, [cm, cm_norm], ["d", ".2f"],
    ["Raw Counts", "Row-Normalised (Recall per Class)"],
):
    cmap = LinearSegmentedColormap.from_list("cr", ["#f8fbff", "#154360"])
    im   = ax.imshow(data, cmap=cmap, vmin=0,
                     vmax=cm.max() if fmt == "d" else 1.0)
    ax.set_xticks(range(N_CLS));  ax.set_yticks(range(N_CLS))
    ax.set_xticklabels(CLASS_NAMES, rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(CLASS_NAMES, fontsize=9)
    ax.set_xlabel("Predicted Region");  ax.set_ylabel("True Region")
    ax.set_title(title, pad=10)
    for i in range(N_CLS):
        for j in range(N_CLS):
            v = data[i, j]
            color = "white" if v > data.max() * 0.55 else "black"
            ax.text(j, i, f"{v:{fmt}}", ha="center", va="center",
                    fontsize=8.5, fontweight="bold", color=color)
    plt.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
fig.tight_layout()
save_fig(fig, "fig1_confusion_matrix.png")


# ---- Figure 2: Per-Class Precision / Recall / F1 + Support -----------------
fig, axes = plt.subplots(1, 2, figsize=(18, 6.5))
fig.suptitle(
    "Figure 2 -- Per-Class Precision, Recall & F1-Score (Test Set)",
    fontsize=14, fontweight="bold", y=1.01,
)
x = np.arange(N_CLS);  w = 0.26
ax = axes[0]
for vals, lbl, off, clr in [
    (r["prec_c"], "Precision", -w, "#2980b9"),
    (r["rec_c"],  "Recall",     0, "#27ae60"),
    (r["f1_c"],   "F1-Score",   w, "#e74c3c"),
]:
    bars = ax.bar(x + off, vals, w, label=lbl, color=clr,
                  edgecolor="white", alpha=0.88)
    for bar in bars:
        h = bar.get_height()
        if h > 0.03:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.013,
                    f"{h:.2f}", ha="center", va="bottom",
                    fontsize=7.5, fontweight="bold")
ax.axhline(r["f1_mac"], color="#7f8c8d", linestyle="--", linewidth=1.4,
           label=f"Macro-F1 = {r['f1_mac']:.4f}")
ax.set_xticks(x)
ax.set_xticklabels(CLASS_NAMES, rotation=32, ha="right", fontsize=9)
ax.set_ylim(0, 1.18);  ax.set_ylabel("Score")
ax.set_title("Per-Class Metric Values");  ax.legend(fontsize=9)

ax2 = axes[1]
bars = ax2.bar(CLASS_NAMES, r["sup_c"], color=CLS_COLORS,
               edgecolor="white", alpha=0.88)
for bar, s in zip(bars, r["sup_c"]):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
             str(s), ha="center", va="bottom", fontsize=9, fontweight="bold")
ax2.set_xticklabels(CLASS_NAMES, rotation=32, ha="right", fontsize=9)
ax2.set_ylabel("Samples");  ax2.set_title(f"Class Support (Test n={len(X_te):,})")
fig.tight_layout()
save_fig(fig, "fig2_per_class_metrics.png")


# ---- Figure 3: 5-Fold Cross-Validation -------------------------------------
folds   = np.arange(1, 6)
CV_PLOT = [
    ("Accuracy",    (CV["Train-Acc"], CV["Accuracy"])),
    ("Macro-F1",    (None,            CV["Macro-F1"])),
    ("Weighted-F1", (None,            CV["Weighted-F1"])),
    ("ROC-AUC",     (None,            CV["ROC-AUC"])),
    ("Log-Loss",    (None,            CV["Log-Loss"])),
]
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle(
    "Figure 3 -- 5-Fold Stratified Cross-Validation (Training Data)\n"
    "Red dashed = fold mean | Shaded ribbon = mean +/-1 std",
    fontsize=14, fontweight="bold", y=1.01,
)
for ai, (mname, (tv, vv)) in enumerate(CV_PLOT):
    ax = axes.flatten()[ai]
    ax.bar(folds, vv, color="#2980b9", alpha=0.78,
           edgecolor="white", label="Validation")
    if tv is not None:
        ax.bar(folds, tv, color="#e74c3c", alpha=0.22,
               edgecolor="white", label="Train")
    m_, s_ = vv.mean(), vv.std()
    ax.axhline(m_, color="#c0392b", linewidth=2.0, label=f"Mean={m_:.4f}")
    ax.axhspan(m_ - s_, m_ + s_, alpha=0.13, color="#c0392b",
               label=f"+/-std={s_:.4f}")
    for i, v in enumerate(vv):
        ax.text(folds[i], v + s_ * 0.20, f"{v:.4f}",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    pad = abs(m_) * 0.05 if m_ != 0 else 0.03
    ax.set_ylim(max(0, m_ - 6 * s_) - pad, min(2, m_ + 6 * s_) + pad)
    ax.set_xticks(folds)
    ax.set_xticklabels([f"Fold {i}" for i in folds], fontsize=9)
    ax.set_title(mname, pad=8)
    ax.legend(fontsize=8,
              loc="lower right" if "Loss" not in mname else "upper right")

# 6th panel: generalisation gap table
ax_gap = axes.flatten()[5];  ax_gap.axis("off")
gdata = [
    ["Metric",      "CV Mean",                        "Test",           "Gap",          "Status"],
    ["Accuracy",    f"{CV['Accuracy'].mean():.4f}",   f"{r['acc']:.4f}",f"{gap_acc:.4f}","OK" if gap_acc < 0.025 else "!"],
    ["Macro-F1",    f"{CV['Macro-F1'].mean():.4f}",   f"{r['f1_mac']:.4f}",f"{gap_f1:.4f}","OK" if gap_f1  < 0.025 else "!"],
    ["Weighted-F1", f"{CV['Weighted-F1'].mean():.4f}",f"{r['f1_wt']:.4f}","---",          "---"],
    ["ROC-AUC",     f"{CV['ROC-AUC'].mean():.4f}",    f"{r['auc_w']:.4f}","---",          "---"],
    ["Log-Loss",    f"{CV['Log-Loss'].mean():.4f}",    f"{r['ll']:.4f}", "---",            "---"],
]
tbl = ax_gap.table(cellText=gdata[1:], colLabels=gdata[0],
                   loc="center", cellLoc="center")
tbl.auto_set_font_size(False);  tbl.set_fontsize(9.5);  tbl.scale(1, 1.9)
for j in range(5):
    tbl[0, j].set_facecolor("#1a252f")
    tbl[0, j].set_text_props(color="white", fontweight="bold")
for i in range(1, len(gdata)):
    ok = gdata[i][-1] == "OK"
    for j in range(5):
        tbl[i, j].set_facecolor(
            ("#d5f5e3" if ok else "#fde8e8") if j == 4
            else ("#eaf4fb" if i % 2 == 0 else "white")
        )
ax_gap.set_title("Generalisation Gap (CV vs Test)",
                 fontsize=12, fontweight="bold", pad=10)
fig.tight_layout()
save_fig(fig, "fig3_cross_validation.png")


# ---- Figure 4: Feature Importance ------------------------------------------
top_n = 20
idx_o = np.argsort(rf_imp)[-top_n:]
fp    = np.array(FEATURES)[idx_o]
vp    = rf_imp[idx_o]
q75   = np.percentile(rf_imp, 75)
bc    = ["#e74c3c" if v >= q75 else "#3498db" for v in vp]

fig, ax = plt.subplots(figsize=(13, 9))
ax.barh(fp, vp, color=bc, edgecolor="white", linewidth=0.7, height=0.72)
ax.axvline(0, color="black", linewidth=0.9)
ax.axvline(np.median(rf_imp), color="#7f8c8d", linestyle="--",
           linewidth=1.2, label="Median")
for i, v in enumerate(vp):
    ax.text(v + 0.0003, i, f"{v:.4f}", va="center",
            fontsize=8.5, fontweight="bold", color="#1a252f")
ax.set_xlabel("Mean Decrease in Gini Impurity (Random Forest)", fontsize=11)
ax.set_title(f"Figure 4 -- Top-{top_n} Feature Importance\n"
             "Red = top-quartile | from RF Gini splits", pad=10)
ax.legend(handles=[
    Patch(facecolor="#e74c3c", label="Top-quartile"),
    Patch(facecolor="#3498db", label="Below top-quartile"),
], fontsize=9)
fig.tight_layout()
save_fig(fig, "fig4_feature_importance.png")


# ---- Figure 5: ROC Curves --------------------------------------------------
auc_per = []
for i in range(N_CLS):
    fr, tr, _ = roc_curve(r["y_bin"][:, i], r["y_proba"][:, i])
    auc_per.append(auc(fr, tr))

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle(
    f"Figure 5 -- Multi-Class ROC Curves (One-vs-Rest, Test Set)\n"
    f"Weighted AUC={r['auc_w']:.4f} | Macro AUC={r['auc_m']:.4f}",
    fontsize=14, fontweight="bold", y=1.01,
)
ax = axes[0]
for i, cls in enumerate(CLASS_NAMES):
    fr, tr, _ = roc_curve(r["y_bin"][:, i], r["y_proba"][:, i])
    ax.plot(fr, tr, linewidth=2.0, color=CLS_COLORS[i],
            label=f"{cls}  (AUC={auc_per[i]:.3f})")
ax.plot([0, 1], [0, 1], "k--", linewidth=1.1, alpha=0.5, label="Random (0.500)")
ax.fill_between([0, 1], [0, 1], alpha=0.05, color="gray")
ax.set_xlim(-0.01, 1.0);  ax.set_ylim(0, 1.02)
ax.set_xlabel("False Positive Rate");  ax.set_ylabel("True Positive Rate")
ax.set_title("Per-Class ROC Curves");  ax.legend(fontsize=8.5, loc="lower right")

ax2 = axes[1]
bars = ax2.bar(CLASS_NAMES, auc_per, color=CLS_COLORS,
               edgecolor="white", alpha=0.88)
ax2.axhline(r["auc_w"], color="#2c3e50", linestyle="--", linewidth=1.6,
            label=f"Weighted mean={r['auc_w']:.4f}")
ax2.axhline(0.5, color="#e74c3c", linestyle=":", linewidth=1.1,
            alpha=0.7, label="Random (0.500)")
for bar, v in zip(bars, auc_per):
    ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.006,
             f"{v:.3f}", ha="center", va="bottom",
             fontsize=9, fontweight="bold")
ax2.set_ylim(0, 1.12)
ax2.set_xticklabels(CLASS_NAMES, rotation=32, ha="right", fontsize=9)
ax2.set_ylabel("AUC Score");  ax2.set_title("AUC Score per Region Class")
ax2.legend(fontsize=9)
fig.tight_layout()
save_fig(fig, "fig5_roc_curves.png")


# ---- Figure 6: Metrics Summary Table ----------------------------------------
def ci95(arr):
    m, s = arr.mean(), arr.std()
    return f"[{m - 1.96*s:.4f}, {m + 1.96*s:.4f}]"

col_labels = ["Metric", "Validation", "Test Value", "CV Mean",
              "CV +/-std", "95% CI", "Dir", "Interpretation"]

table_data = [
    ["Accuracy",
     f"{val_r['acc']:.4f}",    f"{r['acc']:.4f}",
     f"{CV['Accuracy'].mean():.4f}", f"+/-{CV['Accuracy'].std():.4f}",
     ci95(CV["Accuracy"]),     "up", "Correct / Total"],
    ["Macro-F1",
     f"{val_r['f1_mac']:.4f}", f"{r['f1_mac']:.4f}",
     f"{CV['Macro-F1'].mean():.4f}", f"+/-{CV['Macro-F1'].std():.4f}",
     ci95(CV["Macro-F1"]),     "up", "Equal weight / class"],
    ["Weighted-F1",
     f"{val_r['f1_wt']:.4f}",  f"{r['f1_wt']:.4f}",
     f"{CV['Weighted-F1'].mean():.4f}", f"+/-{CV['Weighted-F1'].std():.4f}",
     ci95(CV["Weighted-F1"]),  "up", "Freq-weighted F1"],
    ["Macro Precision",
     f"{val_r['prec_mac']:.4f}", f"{r['prec_mac']:.4f}",
     "---", "---", "---",      "up", "Avg precision/class"],
    ["Macro Recall",
     f"{val_r['rec_mac']:.4f}", f"{r['rec_mac']:.4f}",
     "---", "---", "---",      "up", "Avg recall/class"],
    ["ROC-AUC (Weighted)",
     f"{val_r['auc_w']:.4f}",  f"{r['auc_w']:.4f}",
     f"{CV['ROC-AUC'].mean():.4f}", f"+/-{CV['ROC-AUC'].std():.4f}",
     ci95(CV["ROC-AUC"]),      "up", "Freq-weighted AUC"],
    ["ROC-AUC (Macro)",
     f"{val_r['auc_m']:.4f}",  f"{r['auc_m']:.4f}",
     "---", "---", "---",      "up", "Unweighted AUC"],
    ["Log-Loss",
     f"{val_r['ll']:.4f}",     f"{r['ll']:.4f}",
     f"{CV['Log-Loss'].mean():.4f}", f"+/-{CV['Log-Loss'].std():.4f}",
     ci95(CV["Log-Loss"]),     "dn", "Prob. calibration"],
    ["Cohen kappa",
     f"{val_r['kappa']:.4f}",  f"{r['kappa']:.4f}",
     "---", "---", "---",      "up", "Agreement > chance"],
    ["MCC",
     f"{val_r['mcc']:.4f}",    f"{r['mcc']:.4f}",
     "---", "---", "---",      "up", "Balanced correlation"],
    ["CV-Test Gap Acc",
     "---", f"{gap_acc:.4f}",
     "---", "---", "---",
     "OK" if gap_acc < 0.025 else "!", "< 0.025 = no overfit"],
    ["CV-Test Gap F1",
     "---", f"{gap_f1:.4f}",
     "---", "---", "---",
     "OK" if gap_f1 < 0.025 else "!", "< 0.025 = stable"],
]

fig = plt.figure(figsize=(28, 9.0))
fig.patch.set_facecolor("#eef2f7")
ax  = fig.add_subplot(111)
ax.set_facecolor("#eef2f7");  ax.axis("off")

tbl = ax.table(cellText=table_data, colLabels=col_labels,
               loc="center", cellLoc="center")
tbl.auto_set_font_size(False);  tbl.set_fontsize(10);  tbl.scale(1, 2.20)

HDR = "#1a252f"
for j in range(len(col_labels)):
    c = tbl[0, j]
    c.set_facecolor(HDR);  c.set_edgecolor("#2c3e50")
    c.set_text_props(color="white", fontweight="bold", fontsize=10.5)

STRIPE_A = "#ddeeff";  STRIPE_B = "#ffffff"
GOOD = "#d5f5e3";      WARN = "#fde8e8"
for i in range(1, len(table_data) + 1):
    row = table_data[i - 1]
    is_gap = "Gap" in row[0]
    for j in range(len(col_labels)):
        cell = tbl[i, j]
        if is_gap:
            cell.set_facecolor(GOOD if row[6] == "OK" else WARN)
        else:
            cell.set_facecolor(STRIPE_A if i % 2 == 0 else STRIPE_B)
        cell.set_edgecolor("#c8d6e5")
    tbl[i, 2].set_text_props(fontweight="bold", color="#1a252f")
    tbl[i, 6].set_text_props(
        color=("#27ae60" if row[6] in ("up", "OK")
               else ("#e74c3c" if row[6] == "!"
                     else "#e67e22")),
        fontweight="bold", fontsize=11,
    )

tbl.auto_set_column_width(list(range(len(col_labels))))
ax.set_title(
    "Figure 6 -- Comprehensive Evaluation Metrics Summary\n"
    f"Train n={len(X_tr):,}  Val n={len(X_val):,}  Test n={len(X_te):,}"
    "  |  7 Regions  |  Soft-Voting Ensemble (HGB + RF)",
    fontsize=13, fontweight="bold", pad=24, y=1.03,
)
fig.tight_layout(pad=2.0)
save_fig(fig, "fig6_metrics_summary.png")


# ===========================================================================
# 8. SAMPLE PREDICTIONS
# ===========================================================================
print(f"\n{'=' * 65}")
print("  SAMPLE PREDICTIONS (Test Set)")
print(f"{'=' * 65}")
sample   = X_te.sample(10, random_state=7)
s_pred   = le.inverse_transform(model.predict(sample))
s_proba  = model.predict_proba(sample)
s_actual = le.inverse_transform(y_te.loc[sample.index])
for i, (_, row) in enumerate(sample.iterrows()):
    conf = s_proba[i].max()
    tick = "CORRECT" if s_pred[i] == s_actual[i] else "WRONG  "
    print(f"  [{tick}]  imp={int(row['importance_level']):2d}  "
          f"green={int(row['carbon_budget_strict'])}  "
          f"lat={row['latency_sensitivity']:.2f}  "
          f"dl={int(row['deadline_hours']):3d}h  "
          f"pred: {s_pred[i]:<16s}  actual: {s_actual[i]:<16s}  "
          f"conf: {conf:.3f}")

print(f"\n{'=' * 65}")
print(f"  All outputs written to:  {OUT_ROOT}/")
print(f"    Model   -> {MODEL_PATH}")
print(f"    Splits  -> {SPLIT_DIR}/")
print(f"    Figures -> {FIG_DIR}/  (6 files)")
print(f"{'=' * 65}\n")