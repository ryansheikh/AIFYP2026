"""
=============================================================================
 PIPELINE 1 — Temperature Forecasting (Regression)
=============================================================================
 ✓ XGBoost Regressor
 ✓ Uncertainty Analysis  (Quantile predictions → confidence intervals)
 ✓ Explainability        (SHAP values)
 ✓ Time-series aware split (no data leakage)

 Run:    python pipeline_1_temperature.py
 Output: models/temperature/  (model, SHAP plots, metrics, uncertainty)
"""

import pandas as pd
import numpy as np
import pickle, json, os, warnings
warnings.filterwarnings("ignore")

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from preprocessing import full_preprocess, time_split

OUT_DIR = "models/temperature"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Feature columns (pipeline-specific) ──────────────────────────────────
META_COLS = {"date", "station", "latitude", "longitude",
             "label_heatwave", "label_heavy_rain", "label_storm"}
TARGET = "temperature_2m_mean"

# Exclude the direct target and very correlated raw targets
EXCLUDE_TARGETS = {
    TARGET, "temperature_2m_max", "temperature_2m_min",
    "apparent_temperature_max", "apparent_temperature_min",
}


def get_features(df):
    return [c for c in df.columns
            if c not in META_COLS | EXCLUDE_TARGETS
            and df[c].dtype in (np.float64, np.int64, np.int32, float, int)]


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "=" * 65)
    print("  PIPELINE 1 — Temperature Forecasting")
    print("=" * 65)

    # ── 1. Load & split ──────────────────────────────────────────────
    df = full_preprocess()
    features = get_features(df)
    print(f"  Features: {len(features)}  |  Records: {len(df):,}")

    train, val, test = time_split(df)
    X_train, y_train = train[features], train[TARGET]
    X_val,   y_val   = val[features],   val[TARGET]
    X_test,  y_test  = test[features],  test[TARGET]

    print(f"  Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")

    # ── 2. Train main model (point estimate) ─────────────────────────
    print("\n  Training XGBoost Regressor …")
    model = XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        early_stopping_rounds=30, eval_metric="rmse",
        random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)], verbose=False)

    # ── 3. Evaluate ──────────────────────────────────────────────────
    preds_test = model.predict(X_test)
    metrics = {
        "MAE":  round(mean_absolute_error(y_test, preds_test), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_test, preds_test)), 4),
        "R2":   round(r2_score(y_test, preds_test), 4),
    }
    print(f"\n  TEST RESULTS:")
    print(f"    MAE  = {metrics['MAE']:.2f} °C")
    print(f"    RMSE = {metrics['RMSE']:.2f} °C")
    print(f"    R²   = {metrics['R2']:.4f}")

    # Baseline: persistence model (tomorrow = today)
    baseline_preds = test[f"{TARGET}_lag1"].values
    baseline_mae = mean_absolute_error(y_test, baseline_preds)
    metrics["Baseline_MAE"] = round(baseline_mae, 4)
    metrics["Improvement_over_baseline_%"] = round(
        (1 - metrics["MAE"] / baseline_mae) * 100, 2
    )
    print(f"    Baseline MAE (persistence) = {baseline_mae:.2f} °C")
    print(f"    Improvement = {metrics['Improvement_over_baseline_%']:.1f}%")

    # ── 4. UNCERTAINTY ANALYSIS ──────────────────────────────────────
    print("\n  Uncertainty Analysis (Quantile Regression) …")

    quantile_models = {}
    for alpha in [0.05, 0.25, 0.50, 0.75, 0.95]:
        qm = XGBRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            objective="reg:quantileerror", quantile_alpha=alpha,
            subsample=0.8, random_state=42, n_jobs=-1,
        )
        qm.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        quantile_models[alpha] = qm

    # Generate confidence intervals on test set
    q05  = quantile_models[0.05].predict(X_test)
    q25  = quantile_models[0.25].predict(X_test)
    q50  = quantile_models[0.50].predict(X_test)
    q75  = quantile_models[0.75].predict(X_test)
    q95  = quantile_models[0.95].predict(X_test)

    # Coverage: % of actual values within 90% CI
    coverage_90 = np.mean((y_test.values >= q05) & (y_test.values <= q95))
    coverage_50 = np.mean((y_test.values >= q25) & (y_test.values <= q75))
    avg_ci_width = np.mean(q95 - q05)

    metrics["CI_90_coverage_%"] = round(coverage_90 * 100, 2)
    metrics["CI_50_coverage_%"] = round(coverage_50 * 100, 2)
    metrics["Avg_CI_90_width_C"] = round(avg_ci_width, 2)

    print(f"    90% CI coverage: {coverage_90*100:.1f}%  (target: ~90%)")
    print(f"    50% CI coverage: {coverage_50*100:.1f}%  (target: ~50%)")
    print(f"    Avg 90% CI width: ±{avg_ci_width/2:.1f}°C")

    # Plot uncertainty
    fig, ax = plt.subplots(figsize=(14, 5))
    idx = range(min(365, len(y_test)))
    ax.fill_between(idx, q05[:len(idx)], q95[:len(idx)],
                    alpha=0.15, color="red", label="90% CI")
    ax.fill_between(idx, q25[:len(idx)], q75[:len(idx)],
                    alpha=0.3, color="red", label="50% CI")
    ax.plot(idx, y_test.values[:len(idx)], "k-", lw=0.8, label="Actual")
    ax.plot(idx, preds_test[:len(idx)], "r--", lw=0.8, label="Predicted")
    ax.set_xlabel("Day"); ax.set_ylabel("Temperature (°C)")
    ax.set_title("Temperature Forecast with Uncertainty Bands (Test Set)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/uncertainty_plot.png", dpi=150)
    plt.close()
    print(f"    ✓ Saved uncertainty plot → {OUT_DIR}/uncertainty_plot.png")

    # ── 5. EXPLAINABILITY (SHAP) ─────────────────────────────────────
    print("\n  Explainability (SHAP) …")

    # Use a sample for SHAP (full dataset is too slow)
    X_shap = X_test.sample(n=min(2000, len(X_test)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)

    # Summary plot (bar)
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_shap, plot_type="bar",
                      max_display=20, show=False)
    plt.title("SHAP Feature Importance — Temperature Prediction")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Summary plot (beeswarm)
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_shap, max_display=20, show=False)
    plt.title("SHAP Beeswarm — Temperature Prediction")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"    ✓ SHAP plots saved to {OUT_DIR}/")

    # ── 6. Save everything ───────────────────────────────────────────
    with open(f"{OUT_DIR}/model.pkl", "wb") as f:
        pickle.dump(model, f)
    for alpha, qm in quantile_models.items():
        with open(f"{OUT_DIR}/quantile_{alpha}.pkl", "wb") as f:
            pickle.dump(qm, f)
    with open(f"{OUT_DIR}/features.json", "w") as f:
        json.dump(features, f, indent=2)
    with open(f"{OUT_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  ✓ Pipeline 1 complete — all artifacts in {OUT_DIR}/")
    print("=" * 65)


if __name__ == "__main__":
    main()
