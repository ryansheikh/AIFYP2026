"""
=============================================================================
 PIPELINE 3 — Heavy Rainfall Prediction (Binary Classification)
=============================================================================
 ✓ XGBoost Classifier
 ✓ Uncertainty Analysis  (Calibrated probabilities + entropy)
 ✓ Explainability        (SHAP values)
 ✓ Time-series aware split

 Run:    python pipeline_3_rainfall.py
 Output: models/rainfall/
"""

import pandas as pd
import numpy as np
import pickle, json, os, warnings
warnings.filterwarnings("ignore")

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from preprocessing import full_preprocess, time_split

OUT_DIR = "models/rainfall"
os.makedirs(OUT_DIR, exist_ok=True)

TARGET = "label_heavy_rain"
META_COLS = {"date", "station", "latitude", "longitude",
             "label_heatwave", "label_heavy_rain", "label_storm",
             # Exclude raw rain targets to prevent leakage
             "precipitation_sum", "rain_sum"}


def get_features(df):
    return [c for c in df.columns
            if c not in META_COLS
            and df[c].dtype in (np.float64, np.int64, np.int32, float, int)]


def main():
    print("\n" + "=" * 65)
    print("  PIPELINE 3 — Heavy Rainfall Prediction")
    print("=" * 65)

    df = full_preprocess()
    features = get_features(df)
    print(f"  Features: {len(features)}  |  Heavy rain rate: {df[TARGET].mean()*100:.2f}%")

    train, val, test = time_split(df)
    X_train, y_train = train[features], train[TARGET]
    X_val,   y_val   = val[features],   val[TARGET]
    X_test,  y_test  = test[features],  test[TARGET]

    spw = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
    print(f"  Train: {len(train):,} (pos={y_train.sum()})  "
          f"Val: {len(val):,}  Test: {len(test):,}")

    # ── Train ────────────────────────────────────────────────────────
    print("\n  Training XGBoost Classifier …")
    model = XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw,
        early_stopping_rounds=30, eval_metric="logloss",
        use_label_encoder=False, random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)], verbose=False)

    # ── Evaluate ─────────────────────────────────────────────────────
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy":  round(accuracy_score(y_test, preds), 4),
        "Precision": round(precision_score(y_test, preds, zero_division=0), 4),
        "Recall":    round(recall_score(y_test, preds, zero_division=0), 4),
        "F1":        round(f1_score(y_test, preds, zero_division=0), 4),
        "AUC_ROC":   round(roc_auc_score(y_test, probs) if y_test.nunique() > 1 else 0, 4),
    }
    print(f"\n  TEST RESULTS:")
    for k, v in metrics.items():
        print(f"    {k:12s} = {v}")

    cm = confusion_matrix(y_test, preds)
    print(f"\n  Confusion Matrix:\n{cm}")

    report = classification_report(y_test, preds, target_names=["Normal", "Heavy Rain"])
    with open(f"{OUT_DIR}/classification_report.txt", "w") as f:
        f.write(report)

    # ── UNCERTAINTY ──────────────────────────────────────────────────
    print("\n  Uncertainty Analysis …")
    cal_model = CalibratedClassifierCV(model, cv=3, method="sigmoid")
    cal_model.fit(X_val, y_val)
    cal_probs = cal_model.predict_proba(X_test)[:, 1]

    eps = 1e-10
    entropy = -(cal_probs * np.log2(cal_probs + eps) +
                (1 - cal_probs) * np.log2(1 - cal_probs + eps))

    metrics["Avg_prediction_entropy"] = round(np.mean(entropy), 4)
    metrics["High_confidence_%"] = round(
        np.mean((cal_probs > 0.8) | (cal_probs < 0.2)) * 100, 2
    )

    # Reliability diagram
    if y_test.nunique() > 1:
        frac_pos, mean_pred = calibration_curve(y_test, cal_probs, n_bins=10)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(mean_pred, frac_pos, "o-", label="Calibrated")
        ax.plot([0, 1], [0, 1], "k--", label="Perfect")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title("Heavy Rainfall — Reliability Diagram")
        ax.legend(); plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/reliability_diagram.png", dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(entropy, bins=50, color="#3498db", alpha=0.7, edgecolor="white")
    ax.set_xlabel("Entropy (bits)"); ax.set_ylabel("Count")
    ax.set_title("Rainfall Prediction Uncertainty")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/uncertainty_entropy.png", dpi=150); plt.close()

    # ── SHAP ─────────────────────────────────────────────────────────
    print("\n  Explainability (SHAP) …")
    X_shap = X_test.sample(n=min(2000, len(X_test)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_shap, plot_type="bar", max_display=20, show=False)
    plt.title("SHAP — Heavy Rainfall Prediction")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/shap_importance.png", dpi=150, bbox_inches="tight"); plt.close()

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_shap, max_display=20, show=False)
    plt.title("SHAP Beeswarm — Heavy Rainfall")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/shap_beeswarm.png", dpi=150, bbox_inches="tight"); plt.close()

    # ── Save ─────────────────────────────────────────────────────────
    with open(f"{OUT_DIR}/model.pkl", "wb") as f: pickle.dump(model, f)
    with open(f"{OUT_DIR}/calibrated_model.pkl", "wb") as f: pickle.dump(cal_model, f)
    with open(f"{OUT_DIR}/features.json", "w") as f: json.dump(features, f, indent=2)
    with open(f"{OUT_DIR}/metrics.json", "w") as f: json.dump(metrics, f, indent=2)

    print(f"\n  ✓ Pipeline 3 complete — {OUT_DIR}/")
    print("=" * 65)


if __name__ == "__main__":
    main()
