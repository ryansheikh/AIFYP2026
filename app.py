"""
=============================================================================
 Pakistan Coastal Climate Early Warning System
 Streamlit Cloud Deployment (app.py)
=============================================================================
 4 Separate AI Pipelines: Temperature • Heatwave • Rainfall • Storm
 With Uncertainty Analysis & SHAP Explainability
 
 Authors: Syed Bilal, Raiyan Sheikh & Numra Amjad
 SMIU Karachi — FYP 2025
=============================================================================
"""
 
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle, json, os, io, time, warnings, requests
from datetime import timedelta
 
warnings.filterwarnings("ignore")
 
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
 
# ═══════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Pakistan Coastal Climate Early Warning",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
# ── CSS ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
.stApp { font-family: 'Outfit', sans-serif; }
.hero {
    background: linear-gradient(135deg, #0c2d48 0%, #145374 40%, #2e8bc0 100%);
    padding: 2rem 2.5rem; border-radius: 16px; margin-bottom: 1.5rem;
    color: white; box-shadow: 0 8px 32px rgba(12,45,72,0.4);
}
.hero h1 { margin:0; font-size:1.9rem; font-weight:700; }
.hero p  { margin:0.4rem 0 0; opacity:0.85; font-weight:300; }
.kpi { background:linear-gradient(145deg,#fff,#f0f4f8); border:1px solid #e2e8f0;
       border-radius:12px; padding:1.1rem; text-align:center;
       box-shadow:0 2px 12px rgba(0,0,0,0.05); }
.kpi .val { font-size:1.7rem; font-weight:700; color:#0c2d48; }
.kpi .lbl { font-size:0.82rem; color:#64748b; margin-top:0.2rem; }
.sec { font-size:1.25rem; font-weight:600; color:#0c2d48;
       border-bottom:3px solid #2e8bc0; padding-bottom:0.4rem;
       margin:1.5rem 0 1rem; }
.alert-box { padding:1rem 1.5rem; border-radius:10px; margin:0.5rem 0; font-weight:500; }
.alert-hw  { background:#fdedec; border-left:5px solid #e74c3c; color:#922b21; }
.alert-rn  { background:#eaf2f8; border-left:5px solid #2980b9; color:#1a5276; }
.alert-st  { background:#f4ecf7; border-left:5px solid #8e44ad; color:#6c3483; }
</style>
""", unsafe_allow_html=True)
 
 
def kpi(val, lbl):
    return f'<div class="kpi"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>'
 
 
# ═══════════════════════════════════════════════════════════════════════════
#  DATA: DOWNLOAD + PREPROCESS (cached — runs once)
# ═══════════════════════════════════════════════════════════════════════════
 
STATIONS = {
    "Karachi": (24.86, 67.01), "Thatta": (24.75, 67.92),
    "Badin": (24.63, 68.84),   "Ormara": (25.21, 64.64),
    "Pasni": (25.26, 63.47),   "Gwadar": (25.12, 62.33),
    "Jiwani": (25.05, 61.80),  "Turbat": (26.00, 63.05),
}
 
# ── Only confirmed valid daily variables for Open-Meteo /v1/archive ────────
DAILY_VARS = [
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "apparent_temperature_max", "apparent_temperature_min",
    "precipitation_sum", "rain_sum",
    "windspeed_10m_max", "windgusts_10m_max", "winddirection_10m_dominant",
    "shortwave_radiation_sum", "et0_fao_evapotranspiration",
]
 
# Year chunks to avoid API timeouts on large requests
YEAR_CHUNKS = [
    ("2000-01-01", "2005-12-31"),
    ("2006-01-01", "2010-12-31"),
    ("2011-01-01", "2015-12-31"),
    ("2016-01-01", "2020-12-31"),
    ("2021-01-01", "2024-12-31"),
]
 
 
@st.cache_data(show_spinner="🌐 Downloading 25 years of coastal weather data (this runs once) …")
def download_data():
    """Download from Open-Meteo Historical Archive API in year chunks."""
    BASE = "https://archive-api.open-meteo.com/v1/archive"
    all_frames = []
 
    for name, (lat, lon) in STATIONS.items():
        station_frames = []
        for start, end in YEAR_CHUNKS:
            for attempt in range(3):  # retry up to 3 times
                try:
                    r = requests.get(BASE, params={
                        "latitude": lat, "longitude": lon,
                        "start_date": start, "end_date": end,
                        "daily": ",".join(DAILY_VARS),
                        "timezone": "Asia/Karachi",
                    }, timeout=180)
                    r.raise_for_status()
                    d = pd.DataFrame(r.json()["daily"])
                    d.rename(columns={"time": "date"}, inplace=True)
                    d["date"] = pd.to_datetime(d["date"])
                    station_frames.append(d)
                    break  # success — exit retry loop
                except Exception as e:
                    if attempt == 2:
                        st.warning(f"⚠️ Failed to download {name} ({start}–{end}): {e}")
                    time.sleep(2 * (attempt + 1))  # backoff
 
            time.sleep(0.3)  # polite gap between chunks
 
        if station_frames:
            sdf = pd.concat(station_frames, ignore_index=True)
            sdf["station"] = name
            sdf["latitude"], sdf["longitude"] = lat, lon
            all_frames.append(sdf)
 
    if not all_frames:
        st.error("❌ Could not download any data. Check your internet connection.")
        st.stop()
 
    return pd.concat(all_frames, ignore_index=True)
 
 
@st.cache_data(show_spinner="⚙️ Engineering features …")
def preprocess(df):
    """Clean, feature-engineer, and add labels."""
    df = df.sort_values(["station", "date"]).reset_index(drop=True)
    
    # Interpolate
    num = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ("latitude","longitude")]
    for col in num:
        df[col] = df.groupby("station")[col].transform(
            lambda s: s.interpolate("linear", limit=7).ffill().bfill())
    df.dropna(subset=["temperature_2m_mean", "precipitation_sum"], inplace=True)
    
    # Temporal
    df["year"]      = df["date"].dt.year
    df["month"]     = df["date"].dt.month
    df["day"]       = df["date"].dt.day
    df["dayofyear"] = df["date"].dt.dayofyear
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["month_sin"] = np.sin(2*np.pi*df["month"]/12)
    df["month_cos"] = np.cos(2*np.pi*df["month"]/12)
    df["doy_sin"]   = np.sin(2*np.pi*df["dayofyear"]/365)
    df["doy_cos"]   = np.cos(2*np.pi*df["dayofyear"]/365)
    
    # Lags
    lag_cols = ["temperature_2m_mean","temperature_2m_max","precipitation_sum",
                "windspeed_10m_max","windgusts_10m_max"]
    for col in lag_cols:
        if col not in df.columns: continue
        for lag in [1,3,7,14,30]:
            df[f"{col}_lag{lag}"] = df.groupby("station")[col].shift(lag)
    
    # Rolling
    roll_cols = ["temperature_2m_mean","precipitation_sum","windspeed_10m_max"]
    for col in roll_cols:
        if col not in df.columns: continue
        for w in [7,14,30]:
            g = df.groupby("station")[col]
            df[f"{col}_roll{w}_mean"] = g.transform(lambda s: s.rolling(w, min_periods=1).mean())
            df[f"{col}_roll{w}_std"]  = g.transform(lambda s: s.rolling(w, min_periods=1).std())
    
    # Derived
    if {"temperature_2m_max","temperature_2m_min"}.issubset(df.columns):
        df["temp_range"] = df["temperature_2m_max"] - df["temperature_2m_min"]
    
    # Labels
    df["label_heatwave"]   = (df["temperature_2m_max"] >= 42).astype(int)
    df["label_heavy_rain"] = (df["precipitation_sum"] >= 30).astype(int)
    # Storm: use windgusts if available, otherwise windspeed
    gust_col = "windgusts_10m_max" if "windgusts_10m_max" in df.columns else "windspeed_10m_max"
    df["label_storm"]      = (df[gust_col] >= 60).astype(int)
    
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df
 
 
# ═══════════════════════════════════════════════════════════════════════════
#  MODEL TRAINING (cached — runs once per session)
# ═══════════════════════════════════════════════════════════════════════════
 
META = {"date","station","latitude","longitude","label_heatwave","label_heavy_rain","label_storm"}
 
PIPELINE_CONFIG = {
    "temperature": {
        "target": "temperature_2m_mean",
        "type": "regression",
        "exclude": {"temperature_2m_mean","temperature_2m_max","temperature_2m_min",
                     "apparent_temperature_max","apparent_temperature_min"},
    },
    "heatwave": {
        "target": "label_heatwave",
        "type": "classification",
        "exclude": {"temperature_2m_max","temperature_2m_min","temperature_2m_mean",
                     "apparent_temperature_max","apparent_temperature_min"},
    },
    "rainfall": {
        "target": "label_heavy_rain",
        "type": "classification",
        "exclude": {"precipitation_sum","rain_sum"},
    },
    "storm": {
        "target": "label_storm",
        "type": "classification",
        "exclude": {"windspeed_10m_max","windgusts_10m_max"},
    },
}
 
 
def get_features(df, pipeline_name):
    excl = META | PIPELINE_CONFIG[pipeline_name]["exclude"]
    return [c for c in df.columns if c not in excl
            and df[c].dtype in (np.float64, np.int64, np.int32, float, int)]
 
 
def time_split(df):
    train = df[df["date"].dt.year <= 2019]
    val   = df[(df["date"].dt.year > 2019) & (df["date"].dt.year <= 2021)]
    test  = df[df["date"].dt.year > 2021]
    return train, val, test
 
 
@st.cache_resource(show_spinner="🤖 Training all 4 ML pipelines (this takes a few minutes on first run) …")
def train_all_pipelines(_df):
    """Train all 4 pipelines, return dict of results."""
    df = _df.copy()
    results = {}
    train, val, test = time_split(df)
 
    for pname, cfg in PIPELINE_CONFIG.items():
        feats = get_features(df, pname)
        target = cfg["target"]
        X_tr, y_tr = train[feats], train[target]
        X_vl, y_vl = val[feats],   val[target]
        X_te, y_te = test[feats],  test[target]
 
        res = {"features": feats, "target": target, "type": cfg["type"]}
 
        if cfg["type"] == "regression":
            # ── Main model ───────────────────────────────────────────
            model = XGBRegressor(
                n_estimators=400, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                early_stopping_rounds=25, eval_metric="rmse",
                random_state=42, n_jobs=-1)
            model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)
            preds = model.predict(X_te)
            
            baseline = X_te[f"{target}_lag1"].values if f"{target}_lag1" in feats else y_te.shift(1).bfill().values
            bl_mae = mean_absolute_error(y_te, baseline)
            
            res["model"] = model
            res["metrics"] = {
                "MAE": round(mean_absolute_error(y_te, preds), 3),
                "RMSE": round(np.sqrt(mean_squared_error(y_te, preds)), 3),
                "R2": round(r2_score(y_te, preds), 4),
                "Baseline_MAE": round(bl_mae, 3),
                "Improvement_%": round((1 - mean_absolute_error(y_te, preds)/bl_mae)*100, 1),
            }
 
            # ── Quantile models (uncertainty) ────────────────────────
            qmodels = {}
            for alpha in [0.05, 0.25, 0.50, 0.75, 0.95]:
                qm = XGBRegressor(
                    n_estimators=250, max_depth=5, learning_rate=0.05,
                    objective="reg:quantileerror", quantile_alpha=alpha,
                    subsample=0.8, random_state=42, n_jobs=-1)
                qm.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)
                qmodels[alpha] = qm
            
            q05 = qmodels[0.05].predict(X_te)
            q95 = qmodels[0.95].predict(X_te)
            q25 = qmodels[0.25].predict(X_te)
            q75 = qmodels[0.75].predict(X_te)
            
            res["metrics"]["CI90_coverage_%"] = round(np.mean((y_te.values >= q05) & (y_te.values <= q95))*100, 1)
            res["metrics"]["CI50_coverage_%"] = round(np.mean((y_te.values >= q25) & (y_te.values <= q75))*100, 1)
            res["metrics"]["Avg_CI90_width"] = round(np.mean(q95-q05), 2)
            res["quantile_models"] = qmodels
            res["test_data"] = {"X": X_te, "y": y_te, "preds": preds,
                                "q05": q05, "q25": q25, "q75": q75, "q95": q95,
                                "dates": test["date"].values}
 
        else:
            # ── Classification ───────────────────────────────────────
            spw = (len(y_tr)-y_tr.sum()) / max(y_tr.sum(), 1)
            model = XGBClassifier(
                n_estimators=400, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                early_stopping_rounds=25, eval_metric="logloss",
                use_label_encoder=False, random_state=42, n_jobs=-1)
            model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)
            
            preds = model.predict(X_te)
            probs = model.predict_proba(X_te)[:, 1]
            
            res["model"] = model
            has_both = y_te.nunique() > 1
            res["metrics"] = {
                "Accuracy":  round(accuracy_score(y_te, preds), 4),
                "Precision": round(precision_score(y_te, preds, zero_division=0), 4),
                "Recall":    round(recall_score(y_te, preds, zero_division=0), 4),
                "F1":        round(f1_score(y_te, preds, zero_division=0), 4),
                "AUC_ROC":   round(roc_auc_score(y_te, probs), 4) if has_both else 0,
            }
            
            # Calibration (uncertainty)
            try:
                cal = CalibratedClassifierCV(model, cv=3, method="sigmoid")
                cal.fit(X_vl, y_vl)
                cal_probs = cal.predict_proba(X_te)[:, 1]
            except Exception:
                cal_probs = probs
            
            eps = 1e-10
            entropy = -(cal_probs*np.log2(cal_probs+eps) + (1-cal_probs)*np.log2(1-cal_probs+eps))
            res["metrics"]["Avg_entropy"] = round(np.mean(entropy), 4)
            res["metrics"]["High_confidence_%"] = round(np.mean((cal_probs>0.8)|(cal_probs<0.2))*100, 1)
            
            res["test_data"] = {"y": y_te, "preds": preds, "probs": probs,
                                "cal_probs": cal_probs, "entropy": entropy}
            res["confusion"] = confusion_matrix(y_te, preds)
            res["report"] = classification_report(y_te, preds, 
                target_names=["Normal", pname.title()], zero_division=0)
 
        # ── SHAP (subsample for speed) ───────────────────────────────
        X_shap = X_te.sample(n=min(800, len(X_te)), random_state=42)
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_shap)
        res["shap"] = {"values": sv, "X": X_shap}
 
        results[pname] = res
 
    return results
 
 
# ═══════════════════════════════════════════════════════════════════════════
#  SHAP PLOT HELPERS
# ═══════════════════════════════════════════════════════════════════════════
 
def shap_bar_fig(shap_vals, X_shap, title=""):
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.summary_plot(shap_vals, X_shap, plot_type="bar", max_display=15, show=False)
    plt.title(title, fontsize=12)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return buf
 
def shap_beeswarm_fig(shap_vals, X_shap, title=""):
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.summary_plot(shap_vals, X_shap, max_display=15, show=False)
    plt.title(title, fontsize=12)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return buf
 
 
# ═══════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ═══════════════════════════════════════════════════════════════════════════
 
def main():
    # ── Header ────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero">
        <h1>🌊 Pakistan Coastal Climate Early Warning System</h1>
        <p>4 Separate AI Pipelines — Temperature • Heatwave • Rainfall • Storm | Uncertainty & Explainability (SHAP)</p>
    </div>""", unsafe_allow_html=True)
 
    # ── Load data ─────────────────────────────────────────────────────
    raw = download_data()
    df  = preprocess(raw)
    results = train_all_pipelines(df)
 
    stations = sorted(raw["station"].unique())
 
    # ── Sidebar ───────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Controls")
        station = st.selectbox("📍 Station", stations)
        yr = st.slider("📅 Years", 2000, 2024, (2000, 2024))
        st.markdown("---")
        page = st.radio("Navigate", [
            "🏠 Overview & Alerts",
            "🌡️ Pipeline 1: Temperature",
            "🔥 Pipeline 2: Heatwave",
            "🌧️ Pipeline 3: Rainfall",
            "🌀 Pipeline 4: Storm",
            "📊 All Models Comparison",
        ], label_visibility="collapsed")
        st.markdown("---")
        st.caption("Syed Bilal • Raiyan Sheikh • Numra Amjad\nDept. of AI & Math Sciences\nSMIU Karachi — FYP 2025")
 
    mask = (raw["station"]==station) & (raw["date"].dt.year.between(*yr))
    dfs = raw[mask].copy()
 
    # ══════════════════════════════════════════════════════════════════
    #  PAGE: OVERVIEW
    # ══════════════════════════════════════════════════════════════════
    if page == "🏠 Overview & Alerts":
        st.markdown('<div class="sec">Station Dashboard</div>', unsafe_allow_html=True)
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.markdown(kpi(f"{dfs['temperature_2m_mean'].mean():.1f}°C","Avg Temp"), True)
        c2.markdown(kpi(f"{dfs['temperature_2m_max'].max():.1f}°C","Record High"), True)
        c3.markdown(kpi(f"{dfs['precipitation_sum'].sum():.0f}mm","Total Rain"), True)
        wg = dfs["windgusts_10m_max"].max() if "windgusts_10m_max" in dfs.columns else (dfs["windspeed_10m_max"].max() if "windspeed_10m_max" in dfs.columns else 0)
        c4.markdown(kpi(f"{wg:.0f}km/h","Max Gust"), True)
        c5.markdown(kpi(f"{len(dfs):,}","Days"), True)
 
        st.markdown('<div class="sec">⚠️ Extreme Events Detected</div>', True)
        hw = (dfs["temperature_2m_max"]>=42).sum()
        hr = (dfs["precipitation_sum"]>=30).sum()
        gust_col = "windgusts_10m_max" if "windgusts_10m_max" in dfs.columns else "windspeed_10m_max"
        sm = (dfs[gust_col]>=60).sum() if gust_col in dfs.columns else 0
        c1,c2,c3 = st.columns(3)
        c1.markdown(f'<div class="alert-box alert-hw">🔥 <b>{hw}</b> Heatwave Days (≥42°C)</div>', True)
        c2.markdown(f'<div class="alert-box alert-rn">🌧️ <b>{hr}</b> Heavy Rain Days (≥30mm)</div>', True)
        c3.markdown(f'<div class="alert-box alert-st">🌀 <b>{sm}</b> Storm Days (gusts ≥60km/h)</div>', True)
 
        # Annual trends
        yearly = dfs.groupby(dfs["date"].dt.year).agg(
            heatwave=("temperature_2m_max", lambda x: (x>=42).sum()),
            heavy_rain=("precipitation_sum", lambda x: (x>=30).sum()),
        ).reset_index()
        yearly.columns = ["Year","Heatwave Days","Heavy Rain Days"]
        fig = px.bar(yearly.melt(id_vars="Year", var_name="Event", value_name="Days"),
                     x="Year", y="Days", color="Event", barmode="group",
                     title="Annual Extreme Weather Events",
                     color_discrete_map={"Heatwave Days":"#e74c3c","Heavy Rain Days":"#3498db"})
        fig.update_layout(height=380, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
 
        # Heatmap
        mth = dfs.groupby([dfs["date"].dt.year, dfs["date"].dt.month])["temperature_2m_mean"].mean().reset_index()
        mth.columns = ["Year","Month","Temp"]
        piv = mth.pivot(index="Year", columns="Month", values="Temp")
        piv.columns = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        fig = px.imshow(piv, aspect="auto", color_continuous_scale="RdYlBu_r",
                        labels=dict(color="°C"), title="Monthly Temperature Heatmap")
        fig.update_layout(height=400, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
 
    # ══════════════════════════════════════════════════════════════════
    #  PAGE: TEMPERATURE PIPELINE
    # ══════════════════════════════════════════════════════════════════
    elif page == "🌡️ Pipeline 1: Temperature":
        st.markdown('<div class="sec">Pipeline 1 — Temperature Forecasting (XGBoost Regression)</div>', True)
        r = results["temperature"]
        m = r["metrics"]
 
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.markdown(kpi(f"{m['MAE']}°C","MAE"), True)
        c2.markdown(kpi(f"{m['RMSE']}°C","RMSE"), True)
        c3.markdown(kpi(m["R2"],"R² Score"), True)
        c4.markdown(kpi(f"{m['Baseline_MAE']}°C","Baseline MAE"), True)
        c5.markdown(kpi(f"{m['Improvement_%']}%","Improvement"), True)
 
        # Uncertainty section
        st.markdown('<div class="sec">📐 Uncertainty Analysis (Quantile Regression)</div>', True)
        st.markdown("""
        We train **5 separate XGBoost models** at quantiles 0.05, 0.25, 0.50, 0.75, 0.95 to produce
        genuine confidence intervals. This tells us not just *what* the predicted temperature is, 
        but *how confident* the model is about that prediction.
        """)
        c1,c2,c3 = st.columns(3)
        c1.markdown(kpi(f"{m['CI90_coverage_%']}%","90% CI Coverage (target: 90%)"), True)
        c2.markdown(kpi(f"{m['CI50_coverage_%']}%","50% CI Coverage (target: 50%)"), True)
        c3.markdown(kpi(f"±{m['Avg_CI90_width']/2:.1f}°C","Avg Uncertainty"), True)
 
        td = r["test_data"]
        n = min(365, len(td["y"]))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(n)), y=td["q05"][:n],
            line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=list(range(n)), y=td["q95"][:n],
            fill="tonexty", fillcolor="rgba(231,76,60,0.12)",
            line=dict(width=0), name="90% CI"))
        fig.add_trace(go.Scatter(x=list(range(n)), y=td["q25"][:n],
            line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=list(range(n)), y=td["q75"][:n],
            fill="tonexty", fillcolor="rgba(231,76,60,0.25)",
            line=dict(width=0), name="50% CI"))
        fig.add_trace(go.Scatter(x=list(range(n)), y=td["y"].values[:n],
            line=dict(color="black", width=1), name="Actual"))
        fig.add_trace(go.Scatter(x=list(range(n)), y=td["preds"][:n],
            line=dict(color="red", width=1, dash="dot"), name="Predicted"))
        fig.update_layout(title="Temperature Forecast with Uncertainty Bands (Test Set — 2022-2024)",
                          yaxis_title="°C", height=450, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
 
        # SHAP
        st.markdown('<div class="sec">🔍 Explainability (SHAP)</div>', True)
        st.markdown("SHAP values show **which features drive each prediction** and in which direction.")
        c1, c2 = st.columns(2)
        c1.image(shap_bar_fig(r["shap"]["values"], r["shap"]["X"], "Feature Importance"),
                 caption="Top 15 Features by SHAP Importance", use_container_width=True)
        c2.image(shap_beeswarm_fig(r["shap"]["values"], r["shap"]["X"], "SHAP Beeswarm"),
                 caption="Feature Impact Direction", use_container_width=True)
 
    # ══════════════════════════════════════════════════════════════════
    #  PAGES: CLASSIFICATION PIPELINES (shared layout)
    # ══════════════════════════════════════════════════════════════════
    elif page in ("🔥 Pipeline 2: Heatwave", "🌧️ Pipeline 3: Rainfall", "🌀 Pipeline 4: Storm"):
        pmap = {"🔥 Pipeline 2: Heatwave": "heatwave",
                "🌧️ Pipeline 3: Rainfall": "rainfall",
                "🌀 Pipeline 4: Storm": "storm"}
        pname = pmap[page]
        labels = {"heatwave": ("Heatwave","≥42°C","🔥"),
                  "rainfall": ("Heavy Rainfall","≥30mm/day","🌧️"),
                  "storm": ("Storm Risk","gusts ≥60km/h","🌀")}
        lbl, thresh, emoji = labels[pname]
 
        st.markdown(f'<div class="sec">Pipeline — {lbl} Detection (XGBoost Classification, threshold: {thresh})</div>', True)
        r = results[pname]
        m = r["metrics"]
 
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.markdown(kpi(m["Accuracy"],"Accuracy"), True)
        c2.markdown(kpi(m["Precision"],"Precision"), True)
        c3.markdown(kpi(m["Recall"],"Recall"), True)
        c4.markdown(kpi(m["F1"],"F1 Score"), True)
        c5.markdown(kpi(m["AUC_ROC"],"AUC-ROC"), True)
 
        # Confusion matrix
        st.markdown('<div class="sec">Confusion Matrix & Classification Report</div>', True)
        c1, c2 = st.columns(2)
        with c1:
            cm = r["confusion"]
            fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                            x=["Normal", lbl], y=["Normal", lbl],
                            labels=dict(x="Predicted", y="Actual"))
            fig.update_layout(height=350, title="Confusion Matrix")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.code(r["report"])
 
        # Uncertainty
        st.markdown('<div class="sec">📐 Uncertainty Analysis (Calibrated Probabilities + Entropy)</div>', True)
        st.markdown("""
        We use **Platt scaling** to calibrate raw model probabilities, then compute **prediction entropy** 
        (information-theoretic uncertainty). Low entropy = model is confident. High entropy = model is uncertain.
        """)
        c1, c2 = st.columns(2)
        c1.markdown(kpi(m["Avg_entropy"],"Avg Entropy (lower = more certain)"), True)
        c2.markdown(kpi(f"{m['High_confidence_%']}%","High Confidence Predictions"), True)
 
        td = r["test_data"]
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure(go.Histogram(x=td["entropy"], nbinsx=50,
                marker_color={"heatwave":"#e74c3c","rainfall":"#3498db","storm":"#8e44ad"}[pname]))
            fig.add_vline(x=m["Avg_entropy"], line_dash="dash",
                          annotation_text=f"Mean={m['Avg_entropy']:.3f}")
            fig.update_layout(title="Prediction Entropy Distribution",
                              xaxis_title="Entropy (bits)", yaxis_title="Count",
                              height=380, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        
        with c2:
            # Reliability diagram
            if td["y"].nunique() > 1:
                try:
                    frac_pos, mean_pred = calibration_curve(td["y"], td["cal_probs"], n_bins=10)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=mean_pred, y=frac_pos, mode="lines+markers", name="Model"))
                    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                            line=dict(dash="dash", color="gray"), name="Perfect"))
                    fig.update_layout(title="Reliability Diagram (Calibration)",
                                      xaxis_title="Mean Predicted Probability",
                                      yaxis_title="Fraction of Positives",
                                      height=380, template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    st.info("Not enough positive samples for reliability diagram.")
 
        # SHAP
        st.markdown('<div class="sec">🔍 Explainability (SHAP)</div>', True)
        c1, c2 = st.columns(2)
        c1.image(shap_bar_fig(r["shap"]["values"], r["shap"]["X"],
                 f"Feature Importance — {lbl}"), use_container_width=True)
        c2.image(shap_beeswarm_fig(r["shap"]["values"], r["shap"]["X"],
                 f"SHAP Beeswarm — {lbl}"), use_container_width=True)
 
    # ══════════════════════════════════════════════════════════════════
    #  PAGE: ALL MODELS COMPARISON
    # ══════════════════════════════════════════════════════════════════
    elif page == "📊 All Models Comparison":
        st.markdown('<div class="sec">All 4 Pipelines — Performance Comparison</div>', True)
 
        # Metrics table
        rows = []
        for p in ["temperature","heatwave","rainfall","storm"]:
            row = {"Pipeline": p.title(), "Type": results[p]["type"].title()}
            row.update(results[p]["metrics"])
            rows.append(row)
        comp = pd.DataFrame(rows)
        st.dataframe(comp, use_container_width=True, hide_index=True)
 
        # Classification bar chart
        cls = comp[comp["Type"]=="Classification"]
        if not cls.empty:
            fig = go.Figure()
            for met, col in [("Accuracy","#2ecc71"),("Precision","#3498db"),
                             ("Recall","#e74c3c"),("F1","#f39c12"),("AUC_ROC","#8e44ad")]:
                if met in cls:
                    fig.add_trace(go.Bar(x=cls["Pipeline"], y=cls[met], name=met, marker_color=col))
            fig.update_layout(title="Classification Pipelines — Metrics Comparison",
                              barmode="group", height=420, template="plotly_white",
                              yaxis=dict(range=[0,1.05]))
            st.plotly_chart(fig, use_container_width=True)
 
        # SHAP grid
        st.markdown('<div class="sec">SHAP Feature Importance — All Pipelines</div>', True)
        cols = st.columns(4)
        for i, p in enumerate(["temperature","heatwave","rainfall","storm"]):
            r = results[p]
            cols[i].image(shap_bar_fig(r["shap"]["values"], r["shap"]["X"], p.title()),
                          caption=p.title(), use_container_width=True)
 
        # Uncertainty comparison
        st.markdown('<div class="sec">Uncertainty Metrics</div>', True)
        unc_rows = []
        for p in ["temperature","heatwave","rainfall","storm"]:
            m = results[p]["metrics"]
            row = {"Pipeline": p.title()}
            for k in ["CI90_coverage_%","CI50_coverage_%","Avg_CI90_width",
                       "Avg_entropy","High_confidence_%"]:
                if k in m: row[k] = m[k]
            unc_rows.append(row)
        st.dataframe(pd.DataFrame(unc_rows), use_container_width=True, hide_index=True)
 
 
if __name__ == "__main__":
    main()
