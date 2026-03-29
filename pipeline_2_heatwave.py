"""
=============================================================================
 PIPELINE 2 — Heatwave Detection (Binary Classification)
=============================================================================
 ✓ XGBoost Classifier
 ✓ Uncertainty Analysis  (Calibrated probability + entropy)
 ✓ Explainability        (SHAP values)
 ✓ Time-series aware split

 Run:    python pipeline_2_heatwave.py
 Output: models/heatwave/
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
from sklearn.calibration import CalibratedClassifierCV
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from preprocessing import full_preprocess, time_split

OUT_DIR = "models/heatwave"
os.makedirs(OUT_DIR, exist_ok=True)

TARGET = "label_heatwave"
META_COLS = {"date", "station", "latitude", "longitude",
             "label_heatwave", "label_heavy_rain", "label_storm",
             # Exclude raw temp targets to prevent leakage
             "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
             "apparent_temperature_max", "apparent_temperature_min"}


def get_features(df):
    return [c for c in df.columns
            if c not in META_COLS
            and df[c].dtype in (np.float64, np.int64, np.int32, float, int)]


def main():
    print("\n" + "=" * 65)
    print("  PIPELINE 2 — Heatwave Detection")
    print("=" * 65)

    df = full_preprocess()
    features = get_features(df)
    print(f"  Features: {len(features)}  |  Heatwave rate: {df[TARGET].mean()*100:.2f}%")

    train, val, test = time_split(df)
    X_train, y_train = train[features], train[TARGET]
    X_val,   y_val   = val[features],   val[TARGET]
    X_test,  y_test  = test[features],  test[TARGET]

    # Handle class imbalance via scale_pos_weight
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    spw = neg_count / max(pos_count, 1)

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

    # Confusion matrix
    cm = confusion_matrix(y_test, preds)
    print(f"\n  Confusion Matrix:\n{cm}")

    # Classification report
    report = classification_report(y_test, preds, target_names=["Normal", "Heatwave"])
    with open(f"{OUT_DIR}/classification_report.txt", "w") as f:
        f.write(report)

    # ── UNCERTAINTY ANALYSIS ─────────────────────────────────────────
    print("\n  Uncertainty Analysis (Calibrated Probabilities) …")

    # Platt scaling calibration
    cal_model = CalibratedClassifierCV(model, cv=3, method="sigmoid")
    cal_model.fit(X_val, y_val)
    cal_probs = cal_model.predict_proba(X_test)[:, 1]

    # Prediction entropy (higher = more uncertain)
    eps = 1e-10
    entropy = -(cal_probs * np.log2(cal_probs + eps) +
                (1 - cal_probs) * np.log2(1 - cal_probs + eps))
    avg_entropy = np.mean(entropy)

    # Confidence: fraction of predictions with prob > 0.8 or < 0.2
    high_confidence = np.mean((cal_probs > 0.8) | (cal_probs < 0.2))

    metrics["Avg_prediction_entropy"] = round(avg_entropy, 4)
    metrics["High_confidence_fraction_%"] = round(high_confidence * 100, 2)
    metrics["Calibrated_AUC"] = round(
        roc_auc_score(y_test, cal_probs) if y_test.nunique() > 1 else 0, 4
    )

    print(f"    Avg entropy: {avg_entropy:.4f} (lower = more certain)")
    print(f"    High-confidence predictions: {high_confidence*100:.1f}%")

    # Reliability diagram
    from sklearn.calibration import calibration_curve
    if y_test.nunique() > 1:
        frac_pos, mean_pred = calibration_curve(y_test, cal_probs, n_bins=10)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(mean_pred, frac_pos, "o-", label="Calibrated model")
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title("Heatwave — Reliability Diagram")
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/reliability_diagram.png", dpi=150)
        plt.close()

    # Uncertainty distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(entropy, bins=50, color="#e74c3c", alpha=0.7, edgecolor="white")
    ax.axvline(avg_entropy, color="black", ls="--", label=f"Mean={avg_entropy:.3f}")
    ax.set_xlabel("Prediction Entropy (bits)")
    ax.set_ylabel("Count")
    ax.set_title("Heatwave Prediction Uncertainty Distribution")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/uncertainty_entropy.png", dpi=150)
    plt.close()

    print(f"    ✓ Uncertainty plots saved")

    # ── EXPLAINABILITY (SHAP) ────────────────────────────────────────
    print("\n  Explainability (SHAP) …")
    X_shap = X_test.sample(n=min(2000, len(X_test)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_shap, plot_type="bar",
                      max_display=20, show=False)
    plt.title("SHAP Feature Importance — Heatwave Detection")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_shap, max_display=20, show=False)
    plt.title("SHAP Beeswarm — Heatwave Detection")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"    ✓ SHAP plots saved")

    # ── Save ─────────────────────────────────────────────────────────
    with open(f"{OUT_DIR}/model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(f"{OUT_DIR}/calibrated_model.pkl", "wb") as f:
        pickle.dump(cal_model, f)
    with open(f"{OUT_DIR}/features.json", "w") as f:
        json.dump(features, f, indent=2)
    with open(f"{OUT_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  ✓ Pipeline 2 complete — {OUT_DIR}/")
    print("=" * 65)


if __name__ == "__main__":
    main()
