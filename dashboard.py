"""
=============================================================================
 STREAMLIT DASHBOARD — Pakistan Coastal Climate Early Warning System
=============================================================================
 Showcases all 4 separate ML pipelines with:
   ✓ Historical analysis
   ✓ Predictions with uncertainty bands / confidence levels
   ✓ SHAP explainability
   ✓ Model performance comparison

 Run: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle, json, os
from datetime import timedelta
from PIL import Image

# ── Page Config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pakistan Coastal Climate Early Warning",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────
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
.hero p { margin:0.4rem 0 0; opacity:0.85; font-weight:300; }

.kpi { background: linear-gradient(145deg,#fff,#f0f4f8); border:1px solid #e2e8f0;
       border-radius:12px; padding:1.1rem; text-align:center;
       box-shadow:0 2px 12px rgba(0,0,0,0.05); }
.kpi .val { font-size:1.7rem; font-weight:700; color:#0c2d48; }
.kpi .lbl { font-size:0.82rem; color:#64748b; margin-top:0.2rem; }

.sec { font-size:1.25rem; font-weight:600; color:#0c2d48;
       border-bottom:3px solid #2e8bc0; padding-bottom:0.4rem;
       margin:1.5rem 0 1rem; }

.alert-box { padding: 1rem 1.5rem; border-radius: 10px; margin: 0.5rem 0;
             font-weight: 500; }
.alert-heatwave { background: #fdedec; border-left: 5px solid #e74c3c; color: #922b21; }
.alert-rain     { background: #eaf2f8; border-left: 5px solid #2980b9; color: #1a5276; }
.alert-storm    { background: #f4ecf7; border-left: 5px solid #8e44ad; color: #6c3483; }
.alert-safe     { background: #eafaf1; border-left: 5px solid #27ae60; color: #1e8449; }
</style>
""", unsafe_allow_html=True)


# ── Loaders ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("data/raw/coastal_weather_raw.csv", parse_dates=["date"])

@st.cache_data
def load_metrics(pipeline):
    path = f"models/{pipeline}/metrics.json"
    if os.path.exists(path):
        with open(path) as f: return json.load(f)
    return {}

@st.cache_resource
def load_model(pipeline, filename="model.pkl"):
    path = f"models/{pipeline}/{filename}"
    with open(path, "rb") as f: return pickle.load(f)

@st.cache_data
def load_features(pipeline):
    with open(f"models/{pipeline}/features.json") as f: return json.load(f)

def load_img(pipeline, filename):
    path = f"models/{pipeline}/{filename}"
    if os.path.exists(path):
        return Image.open(path)
    return None

def kpi(val, lbl):
    return f'<div class="kpi"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>'


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    try:
        df = load_data()
    except FileNotFoundError:
        st.error("Data not found. Run `python 01_download_data.py` first.")
        return

    stations = sorted(df["station"].unique())

    # ── Header ───────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero">
        <h1>🌊 Pakistan Coastal Climate Early Warning System</h1>
        <p>4 Separate AI Pipelines — Temperature • Heatwave • Rainfall • Storm | With Uncertainty & Explainability</p>
    </div>""", unsafe_allow_html=True)

    # ── Sidebar ──────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Controls")
        station = st.selectbox("📍 Station", stations)
        yr = st.slider("📅 Years", int(df["date"].dt.year.min()),
                        int(df["date"].dt.year.max()), (2000, 2024))
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
        st.caption("Syed Bilal • Raiyan Sheikh • Numra Amjad\nSMIU Karachi — FYP 2025")

    # Filter
    mask = (df["station"] == station) & (df["date"].dt.year.between(*yr))
    dfs = df[mask].copy()
    if dfs.empty:
        st.warning("No data for selection."); return

    # ══════════════════════════════════════════════════════════════════
    #  OVERVIEW & ALERTS
    # ══════════════════════════════════════════════════════════════════
    if page == "🏠 Overview & Alerts":
        st.markdown('<div class="sec">Station Dashboard</div>', unsafe_allow_html=True)

        c1,c2,c3,c4,c5 = st.columns(5)
        c1.markdown(kpi(f"{dfs['temperature_2m_mean'].mean():.1f}°C","Avg Temp"), unsafe_allow_html=True)
        c2.markdown(kpi(f"{dfs['temperature_2m_max'].max():.1f}°C","Record High"), unsafe_allow_html=True)
        c3.markdown(kpi(f"{dfs['precipitation_sum'].sum():.0f}mm","Total Rain"), unsafe_allow_html=True)
        c4.markdown(kpi(f"{dfs['windgusts_10m_max'].max():.0f}km/h","Max Gust"), unsafe_allow_html=True)
        c5.markdown(kpi(f"{len(dfs):,}","Data Points"), unsafe_allow_html=True)

        # Extreme weather counts
        st.markdown('<div class="sec">⚠️ Extreme Weather Events Detected</div>', unsafe_allow_html=True)
        hw_days = (dfs["temperature_2m_max"] >= 42).sum()
        hr_days = (dfs["precipitation_sum"] >= 30).sum()
        st_days = (dfs["windgusts_10m_max"] >= 60).sum() if "windgusts_10m_max" in dfs else 0

        c1,c2,c3 = st.columns(3)
        c1.markdown(f'<div class="alert-box alert-heatwave">🔥 <b>{hw_days}</b> Heatwave Days (≥42°C)</div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="alert-box alert-rain">🌧️ <b>{hr_days}</b> Heavy Rain Days (≥30mm)</div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="alert-box alert-storm">🌀 <b>{st_days}</b> Storm Risk Days (gusts ≥60km/h)</div>', unsafe_allow_html=True)

        # Annual extreme events trend
        yearly = dfs.groupby(dfs["date"].dt.year).agg(
            heatwaves=("temperature_2m_max", lambda x: (x >= 42).sum()),
            heavy_rain=("precipitation_sum", lambda x: (x >= 30).sum()),
        ).reset_index()
        yearly.columns = ["Year", "Heatwave Days", "Heavy Rain Days"]

        fig = px.bar(yearly.melt(id_vars="Year", var_name="Event", value_name="Days"),
                     x="Year", y="Days", color="Event", barmode="group",
                     title="Annual Extreme Weather Events", color_discrete_map={
                         "Heatwave Days": "#e74c3c", "Heavy Rain Days": "#3498db"})
        fig.update_layout(height=380, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        # Temperature heatmap
        st.markdown('<div class="sec">Monthly Temperature Heatmap</div>', unsafe_allow_html=True)
        mth = dfs.groupby([dfs["date"].dt.year, dfs["date"].dt.month])["temperature_2m_mean"].mean().reset_index()
        mth.columns = ["Year","Month","Temp"]
        piv = mth.pivot(index="Year", columns="Month", values="Temp")
        piv.columns = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        fig = px.imshow(piv, aspect="auto", color_continuous_scale="RdYlBu_r", labels=dict(color="°C"))
        fig.update_layout(height=380, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════
    #  PIPELINE 1: TEMPERATURE
    # ══════════════════════════════════════════════════════════════════
    elif page == "🌡️ Pipeline 1: Temperature":
        st.markdown('<div class="sec">Pipeline 1 — Temperature Forecasting (Regression)</div>', unsafe_allow_html=True)

        met = load_metrics("temperature")
        if met:
            c1,c2,c3,c4 = st.columns(4)
            c1.markdown(kpi(f"{met.get('MAE','?')}°C", "MAE"), unsafe_allow_html=True)
            c2.markdown(kpi(f"{met.get('RMSE','?')}°C", "RMSE"), unsafe_allow_html=True)
            c3.markdown(kpi(met.get("R2","?"), "R² Score"), unsafe_allow_html=True)
            c4.markdown(kpi(f"{met.get('Improvement_over_baseline_%','?')}%", "vs Baseline"), unsafe_allow_html=True)

            st.markdown('<div class="sec">Uncertainty Analysis</div>', unsafe_allow_html=True)
            c1,c2,c3 = st.columns(3)
            c1.markdown(kpi(f"{met.get('CI_90_coverage_%','?')}%", "90% CI Coverage"), unsafe_allow_html=True)
            c2.markdown(kpi(f"{met.get('CI_50_coverage_%','?')}%", "50% CI Coverage"), unsafe_allow_html=True)
            c3.markdown(kpi(f"±{met.get('Avg_CI_90_width_C',0)/2:.1f}°C", "Avg Uncertainty"), unsafe_allow_html=True)

            img = load_img("temperature", "uncertainty_plot.png")
            if img: st.image(img, caption="Temperature Forecast with Confidence Intervals", use_container_width=True)

            st.markdown('<div class="sec">Explainability (SHAP)</div>', unsafe_allow_html=True)
            c1,c2 = st.columns(2)
            img1 = load_img("temperature", "shap_importance.png")
            img2 = load_img("temperature", "shap_beeswarm.png")
            if img1: c1.image(img1, caption="Feature Importance", use_container_width=True)
            if img2: c2.image(img2, caption="SHAP Beeswarm", use_container_width=True)
        else:
            st.warning("Run `python pipeline_1_temperature.py` first.")

    # ══════════════════════════════════════════════════════════════════
    #  PIPELINE 2: HEATWAVE
    # ══════════════════════════════════════════════════════════════════
    elif page == "🔥 Pipeline 2: Heatwave":
        st.markdown('<div class="sec">Pipeline 2 — Heatwave Detection (Classification)</div>', unsafe_allow_html=True)
        _show_classification_pipeline("heatwave", "🔥", "#e74c3c")

    # ══════════════════════════════════════════════════════════════════
    #  PIPELINE 3: RAINFALL
    # ══════════════════════════════════════════════════════════════════
    elif page == "🌧️ Pipeline 3: Rainfall":
        st.markdown('<div class="sec">Pipeline 3 — Heavy Rainfall Prediction (Classification)</div>', unsafe_allow_html=True)
        _show_classification_pipeline("rainfall", "🌧️", "#3498db")

    # ══════════════════════════════════════════════════════════════════
    #  PIPELINE 4: STORM
    # ══════════════════════════════════════════════════════════════════
    elif page == "🌀 Pipeline 4: Storm":
        st.markdown('<div class="sec">Pipeline 4 — Storm Risk Detection (Classification)</div>', unsafe_allow_html=True)
        _show_classification_pipeline("storm", "🌀", "#8e44ad")

    # ══════════════════════════════════════════════════════════════════
    #  ALL MODELS COMPARISON
    # ══════════════════════════════════════════════════════════════════
    elif page == "📊 All Models Comparison":
        st.markdown('<div class="sec">All 4 Pipelines — Performance Comparison</div>', unsafe_allow_html=True)

        rows = []
        for p in ["temperature", "heatwave", "rainfall", "storm"]:
            m = load_metrics(p)
            if m:
                row = {"Pipeline": p.title()}
                row.update(m)
                rows.append(row)
        
        if rows:
            comp = pd.DataFrame(rows)
            st.dataframe(comp, use_container_width=True)

            # Classification comparison
            cls_pipes = [r for r in rows if "F1" in r]
            if cls_pipes:
                cls_df = pd.DataFrame(cls_pipes)
                fig = go.Figure()
                for metric, color in [("Accuracy","#2ecc71"), ("Precision","#3498db"),
                                       ("Recall","#e74c3c"), ("F1","#f39c12")]:
                    if metric in cls_df:
                        fig.add_trace(go.Bar(
                            x=cls_df["Pipeline"], y=cls_df[metric],
                            name=metric, marker_color=color,
                        ))
                fig.update_layout(
                    title="Classification Metrics Comparison",
                    barmode="group", height=420, template="plotly_white",
                    yaxis=dict(range=[0, 1.05]),
                )
                st.plotly_chart(fig, use_container_width=True)

            # Uncertainty comparison
            unc_pipes = [r for r in rows if "Avg_prediction_entropy" in r or "CI_90_coverage_%" in r]
            if unc_pipes:
                st.markdown('<div class="sec">Uncertainty Comparison</div>', unsafe_allow_html=True)
                unc_df = pd.DataFrame(unc_pipes)
                st.dataframe(unc_df[["Pipeline"] + [c for c in unc_df.columns 
                    if "entropy" in c.lower() or "ci" in c.lower() or "confidence" in c.lower()]],
                    use_container_width=True)

            # SHAP plots grid
            st.markdown('<div class="sec">Explainability (SHAP) — All Pipelines</div>', unsafe_allow_html=True)
            cols = st.columns(4)
            for i, p in enumerate(["temperature", "heatwave", "rainfall", "storm"]):
                img = load_img(p, "shap_importance.png")
                if img:
                    cols[i].image(img, caption=f"{p.title()}", use_container_width=True)
        else:
            st.warning("No model metrics found. Run all pipelines first.")


def _show_classification_pipeline(pipeline, emoji, color):
    """Reusable display for classification pipelines (heatwave/rain/storm)."""
    met = load_metrics(pipeline)
    if not met:
        st.warning(f"Run `python pipeline_{['','','2','3','4'][['','temperature','heatwave','rainfall','storm'].index(pipeline)]}_{pipeline}.py` first.")
        return

    # Metrics
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.markdown(kpi(met.get("Accuracy","?"), "Accuracy"), unsafe_allow_html=True)
    c2.markdown(kpi(met.get("Precision","?"), "Precision"), unsafe_allow_html=True)
    c3.markdown(kpi(met.get("Recall","?"), "Recall"), unsafe_allow_html=True)
    c4.markdown(kpi(met.get("F1","?"), "F1 Score"), unsafe_allow_html=True)
    c5.markdown(kpi(met.get("AUC_ROC","?"), "AUC-ROC"), unsafe_allow_html=True)

    # Uncertainty
    st.markdown('<div class="sec">Uncertainty Analysis</div>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    c1.markdown(kpi(met.get("Avg_prediction_entropy","?"), "Avg Entropy (lower=better)"), unsafe_allow_html=True)
    c2.markdown(kpi(f"{met.get('High_confidence_%','?')}%", "High Confidence Predictions"), unsafe_allow_html=True)

    cl, cr = st.columns(2)
    img_rel = load_img(pipeline, "reliability_diagram.png")
    img_ent = load_img(pipeline, "uncertainty_entropy.png")
    if img_rel: cl.image(img_rel, caption="Reliability Diagram (Calibration)", use_container_width=True)
    if img_ent: cr.image(img_ent, caption="Uncertainty Distribution", use_container_width=True)

    # SHAP
    st.markdown('<div class="sec">Explainability (SHAP)</div>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    img1 = load_img(pipeline, "shap_importance.png")
    img2 = load_img(pipeline, "shap_beeswarm.png")
    if img1: c1.image(img1, caption="Feature Importance", use_container_width=True)
    if img2: c2.image(img2, caption="SHAP Beeswarm", use_container_width=True)

    # Classification report
    rpt_path = f"models/{pipeline}/classification_report.txt"
    if os.path.exists(rpt_path):
        with open(rpt_path) as f:
            st.markdown('<div class="sec">Classification Report</div>', unsafe_allow_html=True)
            st.code(f.read())


if __name__ == "__main__":
    main()
