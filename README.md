# 🌊 Pakistan Coastal Climate Early Warning System
### FYP — Environmental Challenges in Pakistan: A Sustainable Development and AI Perspective
**Syed Bilal, Raiyan Sheikh & Numra Amjad** | SMIU Karachi

---

## 🏗️ Architecture: 4 Separate Pipelines

Each extreme weather target has its **own dedicated pipeline** with training, evaluation, 
uncertainty analysis, and explainability — as requested by supervisor.

```
┌─────────────────────────────────────────────────────────┐
│                    01_download_data.py                    │
│         (Open-Meteo API → 8 coastal stations)           │
└──────────────────────┬──────────────────────────────────┘
                       │
              02_preprocessing.py (shared module)
                       │
        ┌──────────────┼──────────────┬───────────────┐
        ▼              ▼              ▼               ▼
  Pipeline 1      Pipeline 2     Pipeline 3      Pipeline 4
  TEMPERATURE     HEATWAVE       RAINFALL        STORM
  (Regression)    (Classif.)     (Classif.)      (Classif.)
  XGBoost Reg.    XGBoost Clf.   XGBoost Clf.    XGBoost Clf.
  Quantile CI     Calibrated P   Calibrated P    Calibrated P
  SHAP            SHAP           SHAP            SHAP
        │              │              │               │
        └──────────────┴──────────────┴───────────────┘
                       │
                  dashboard.py
              (Streamlit — unified UI)
```

## 🚀 Quick Start

```bash
pip install -r requirements.txt

# Step 1: Download data (~2 min)
python 01_download_data.py

# Step 2: Run all 4 pipelines (~10 min total)
python run_all_pipelines.py

# Step 3: Launch dashboard
streamlit run dashboard.py
```

Or run pipelines individually:
```bash
python pipeline_1_temperature.py
python pipeline_2_heatwave.py
python pipeline_3_rainfall.py
python pipeline_4_storm.py
```

## 📊 What Each Pipeline Produces

### Pipeline 1 — Temperature Forecasting
- **Model**: XGBoost Regressor (500 trees, depth 6)
- **Metrics**: MAE, RMSE, R², improvement over persistence baseline
- **Uncertainty**: Quantile regression (5th, 25th, 50th, 75th, 95th percentiles)
  → 90% and 50% confidence intervals with coverage statistics
- **Explainability**: SHAP bar plot + beeswarm plot (top 20 features)

### Pipeline 2 — Heatwave Detection
- **Target**: Temperature ≥ 42°C (Pakistan coastal threshold)
- **Model**: XGBoost Classifier with scale_pos_weight for imbalance
- **Metrics**: Accuracy, Precision, Recall, F1, AUC-ROC, confusion matrix
- **Uncertainty**: Platt-scaled calibrated probabilities + prediction entropy
  → Reliability diagram + entropy distribution
- **Explainability**: SHAP importance + beeswarm

### Pipeline 3 — Heavy Rainfall Prediction
- **Target**: Precipitation ≥ 30 mm/day (PMD heavy rain threshold)
- **Model**: XGBoost Classifier with class weighting
- **Same evaluation structure as Pipeline 2

### Pipeline 4 — Storm Risk Detection
- **Target**: Wind gusts ≥ 60 km/h (tropical storm threshold for coasts)
- **Model**: XGBoost Classifier with class weighting
- **Same evaluation structure as Pipeline 2

## 🔑 Key Design Decisions (for evaluators)

### Why separate pipelines?
Each weather hazard has different physics, different features matter, different 
class distributions, and different operational thresholds. A unified model would 
compromise on all of these. Separate pipelines allow:
- Feature selection tailored to each hazard (e.g., pressure for storms vs radiation for heatwaves)
- Leakage prevention (each pipeline excludes its own target-related raw features)
- Independent uncertainty calibration
- Clear SHAP interpretability per hazard

### Why XGBoost over Transformer?
- Tabular daily features → tree-based models consistently outperform neural networks
- XGBoost supports native quantile regression for uncertainty
- TreeSHAP provides exact (not approximate) SHAP values
- Reproducible, fast, and deployable on low-resource systems

### Time-series split (no data leakage)
- Train: 2000–2019 | Validation: 2020–2021 | Test: 2022–2024
- No random splitting — strictly temporal

### Uncertainty Analysis types
| Pipeline | Method | What it tells you |
|----------|--------|-------------------|
| Temperature | Quantile Regression | "32°C, but could be 30-34°C (90% CI)" |
| Heatwave | Calibrated probabilities + entropy | "78% chance of heatwave, high confidence" |
| Rainfall | Calibrated probabilities + entropy | "45% chance of heavy rain, moderate uncertainty" |
| Storm | Calibrated probabilities + entropy | "92% normal, very confident" |

## 📂 Output Structure
```
models/
├── temperature/
│   ├── model.pkl              (main XGBoost regressor)
│   ├── quantile_0.05.pkl      (5th percentile model)
│   ├── quantile_0.25.pkl      (25th percentile model)
│   ├── quantile_0.50.pkl      (median model)
│   ├── quantile_0.75.pkl      (75th percentile model)
│   ├── quantile_0.95.pkl      (95th percentile model)
│   ├── features.json
│   ├── metrics.json
│   ├── uncertainty_plot.png
│   ├── shap_importance.png
│   └── shap_beeswarm.png
├── heatwave/
│   ├── model.pkl
│   ├── calibrated_model.pkl
│   ├── features.json
│   ├── metrics.json
│   ├── classification_report.txt
│   ├── reliability_diagram.png
│   ├── uncertainty_entropy.png
│   ├── shap_importance.png
│   └── shap_beeswarm.png
├── rainfall/   (same structure)
└── storm/      (same structure)
```
