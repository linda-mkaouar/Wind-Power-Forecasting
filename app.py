# ============================================================
# app.py  —  Wind Turbine Power Forecasting Dashboard (v3)
# Run with:  streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from datetime import datetime

from pipeline import run_pipeline, load_all, load_ensemble_models

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Wind Power Forecasting",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# THEME SYSTEM
# ─────────────────────────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

DARK = {
    "bg":        "#0a0e17",
    "bg2":       "#111827",
    "card":      "#1a2236",
    "border":    "#2d3a52",
    "text":      "#e2eaf5",
    "text2":     "#7a8fa8",
    "accent":    "#4da6ff",
    "accent2":   "#34d058",
    "accent3":   "#ff6b6b",
    "plotly_tpl": "plotly_dark",
    "colorscale": "Blues",
}
LIGHT = {
    "bg":        "#f6f8fa",
    "bg2":       "#ffffff",
    "card":      "#ffffff",
    "border":    "#d0d7de",
    "text":      "#1f2328",
    "text2":     "#656d76",
    "accent":    "#0969da",
    "accent2":   "#1a7f37",
    "accent3":   "#cf222e",
    "plotly_tpl": "plotly",
    "colorscale": "Blues",
}

def T():
    return DARK if st.session_state.theme == "dark" else LIGHT

def inject_css():
    t = T()
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}
    .stApp {{
        background-color: {t['bg']};
        color: {t['text']};
        background-image: {'radial-gradient(ellipse at 20% 0%, #0f1e3a 0%, transparent 50%), radial-gradient(ellipse at 80% 100%, #0a1a2e 0%, transparent 50%)' if st.session_state.theme == 'dark' else 'none'};
    }}
    section[data-testid="stSidebar"] {{
        background: {'linear-gradient(180deg, #111827 0%, #0d1520 100%)' if st.session_state.theme == 'dark' else t['bg2']} !important;
        border-right: 1px solid {t['border']};
    }}
    section[data-testid="stSidebar"] * {{
        color: {t['text']} !important;
    }}
    .stSelectbox > div > div, .stRadio > div {{
        background-color: {t['card']};
        color: {t['text']};
    }}
    .stSlider > div > div > div > div {{
        background-color: {t['accent']};
    }}
    .metric-card {{
        background: {'linear-gradient(135deg, #1a2236 0%, #131c30 100%)' if st.session_state.theme == 'dark' else t['card']};
        border: 1px solid {t['border']};
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        {'box-shadow: 0 2px 12px rgba(0,0,0,0.4);' if st.session_state.theme == 'dark' else ''}
    }}
    .metric-card:hover {{ transform: translateY(-2px); {'box-shadow: 0 6px 20px rgba(77,166,255,0.12);' if st.session_state.theme == 'dark' else ''} }}
    .metric-value {{
        font-size: 2rem;
        font-weight: 700;
        color: {t['accent']};
        line-height: 1.2;
    }}
    .metric-label {{
        font-size: 0.8rem;
        font-weight: 500;
        color: {t['text2']};
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 4px;
    }}
    .metric-delta {{
        font-size: 0.85rem;
        color: {t['accent2']};
        margin-top: 4px;
    }}
    .section-header {{
        font-size: 1.4rem;
        font-weight: 600;
        color: {t['text']};
        border-left: 4px solid {t['accent']};
        padding-left: 12px;
        margin: 24px 0 16px 0;
    }}
    .info-box {{
        background: {t['card']};
        border: 1px solid {t['border']};
        border-radius: 10px;
        padding: 16px 20px;
        margin: 8px 0;
        color: {t['text']};
    }}
    .badge {{
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        background: {t['accent']};
        color: white;
    }}
    .hero-gradient {{
        background: linear-gradient(135deg, {t['bg2']} 0%, {t['card']} 100%);
        border: 1px solid {t['border']};
        border-radius: 16px;
        padding: 32px;
        margin-bottom: 24px;
    }}
    div[data-testid="metric-container"] {{
        background: {t['card']};
        border: 1px solid {t['border']};
        border-radius: 10px;
        padding: 16px;
    }}
    div[data-testid="metric-container"] label {{
        color: {t['text2']} !important;
    }}
    .stDataFrame {{ background: {t['card']}; border-radius: 10px; }}
    .sidebar-title {{
        font-size: 1.1rem;
        font-weight: 700;
        color: {t['accent']};
        margin-bottom: 4px;
        letter-spacing: 0.02em;
    }}
    .nav-item {{
        padding: 8px 12px;
        border-radius: 8px;
        margin: 2px 0;
        cursor: pointer;
        font-size: 0.9rem;
        color: {t['text']};
    }}
    .nav-item:hover {{ background: {t['card']}; }}
    div[data-testid="stForm"] {{ background: {t['card']}; border-radius: 12px; padding: 16px; }}
    .stButton > button {{
        background: {t['accent']};
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s;
    }}
    .stButton > button:hover {{ opacity: 0.85; transform: translateY(-1px); }}
    h1, h2, h3, h4 {{ color: {t['text']} !important; }}
    p {{ color: {t['text']}; }}
    /* Top navigation radio styling */
    div[role="radiogroup"] {{
        display: flex !important;
        flex-wrap: nowrap !important;
        gap: 4px !important;
        overflow-x: auto !important;
    }}
    div[role="radiogroup"] label {{
        padding: 4px 10px !important;
        border-radius: 6px !important;
        margin: 0 !important;
        background: {t['card']} !important;
        border: 1px solid {t['border']} !important;
        font-weight: 500 !important;
        font-size: 0.78rem !important;
        white-space: nowrap !important;
        flex-shrink: 0 !important;
    }}
    div[role="radiogroup"] label[data-selected="true"] {{
        background: {t['accent']} !important;
        color: white !important;
        border-color: {t['accent']} !important;
    }}
    /* Upload area */
    .upload-hint {{
        font-size: 0.75rem;
        color: {t['text2']};
        margin-top: 4px;
        line-height: 1.4;
    }}
    .upload-status {{
        background: {t['card']};
        border: 1px solid {t['border']};
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 0.82rem;
        margin-top: 8px;
    }}
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# CACHED WRAPPERS (Streamlit cache must live in app.py)
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def _cached_load_all():
    return load_all()

@st.cache_resource
def _cached_load_ensemble_models():
    return load_ensemble_models()


PAGES = [
    "🏠  Overview",
    "🔎  Data Quality",
    "📈  Energy Production",
    "🔍  Wind & Climate Analysis",
    "⚡  Turbine Performance",
    "🤖  Forecast Accuracy",
    "🎯  Power Forecasting",
]
if "page" not in st.session_state:
    st.session_state.page = PAGES[0]
if "pipeline_log" not in st.session_state:
    st.session_state.pipeline_log = []
if "upload_done" not in st.session_state:
    st.session_state.upload_done = False

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<div class="sidebar-title">🌬️ WindCast</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:0.72rem; color:#8b949e;">Power Forecasting Dashboard</div>', unsafe_allow_html=True)
    with col2:
        icon = "☀️" if st.session_state.theme == "dark" else "🌙"
        if st.button(icon, help="Toggle theme"):
            st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
            st.rerun()

    st.markdown("---")

    # ── Dataset upload (first) ─────────────────────────────────
    st.markdown("**📂 Upload Turbine Dataset**")
    st.markdown('<div class="upload-hint">CSV with columns: Time, Power, windspeed_10m, windspeed_100m, windgusts_10m, temperature_2m, relativehumidity_2m, winddirection_100m, dewpoint_2m.</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drop CSV here",
        type=["csv"],
        label_visibility="collapsed",
        help="Upload a CSV matching the required column schema",
    )

    if uploaded_file is not None and not st.session_state.upload_done:
        with st.spinner("⚙️ Running preprocessing & training all models…"):
            try:
                df_uploaded = pd.read_csv(uploaded_file)
                if "Power" in df_uploaded.columns and df_uploaded["Power"].max() > 1.0:
                    st.warning(
                        "⚠️ **Power column doesn't appear to be normalized.** "
                        f"Max value detected: `{df_uploaded['Power'].max():.2f}`. "
                        "The pipeline expects Power values in the range [0, 1]. "
                        "Predictions will be scaled against your rated power input, "
                        "so raw kW values will produce incorrect forecasts. "
                        "Divide your Power column by the turbine's rated capacity before uploading."
                    )
                log = run_pipeline(df_uploaded)
                st.session_state.pipeline_log = log
                st.session_state.upload_done = True
                st.cache_resource.clear()
                st.success("✅ Models trained successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Pipeline failed: {e}")

    if uploaded_file is None:
        st.session_state.upload_done = False

    if st.session_state.pipeline_log:
        with st.expander("📋 Training Log", expanded=False):
            for line in st.session_state.pipeline_log:
                st.markdown(f"<div style='font-size:0.78rem;'>{line}</div>", unsafe_allow_html=True)

    # ── Dataset info (if loaded) ──
    _data_sidebar = _cached_load_all()
    if _data_sidebar:
        _meta = _data_sidebar["meta"]
        st.markdown("---")
        st.markdown(f"""
        <div class="info-box">
            <div style="font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;color:#8b949e;">Dataset</div>
            <div style="font-size:0.85rem;margin-top:6px;">{_meta['n_train']+_meta['n_test']:,} rows</div>
            <div style="font-size:0.72rem;color:#8b949e;">{_meta.get('train_start', _meta['train_end'])[:4]} – {_meta['test_end'][:4]}</div>
        </div>
        <div class="info-box" style="margin-top:8px;">
            <div style="font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;color:#8b949e;">Best Model</div>
            <div style="font-size:0.85rem;margin-top:6px;">{_meta['best_model_name']}</div>
            <div style="font-size:0.72rem;color:#8b949e;">{_meta['n_features']} features</div>
        </div>
        """, unsafe_allow_html=True)

inject_css()

# ─────────────────────────────────────────────────────────────
# STICKY TOP NAVIGATION BAR — native st.radio + CSS styling
# ─────────────────────────────────────────────────────────────
t = T()

st.markdown(f"""
<style>
/* ── Make the radio row sticky ── */
div[data-testid="stHorizontalBlock"]:has(div[data-testid="stRadio"]) {{
    position: sticky;
    top: 0;
    z-index: 9999;
    background: {t['bg2']};
    border-bottom: 1.5px solid {t['border']};
    padding: 8px 12px 8px 12px;
    margin-bottom: 12px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.25);
}}

/* ── Radio group: horizontal flex row ── */
div[data-testid="stRadio"] > div[role="radiogroup"] {{
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    gap: 4px !important;
    overflow-x: auto;
    scrollbar-width: none;
    padding: 2px 0;
}}
div[data-testid="stRadio"] > div[role="radiogroup"]::-webkit-scrollbar {{
    display: none;
}}

/* ── Hide the native radio circle ── */
div[data-testid="stRadio"] > div[role="radiogroup"] label > div:first-child {{
    display: none !important;
}}

/* ── Each label → pill ── */
div[data-testid="stRadio"] > div[role="radiogroup"] label {{
    display: inline-flex !important;
    align-items: center !important;
    padding: 5px 14px !important;
    border-radius: 20px !important;
    border: 1px solid {t['border']} !important;
    background: transparent !important;
    color: {t['text2']} !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    white-space: nowrap !important;
    flex-shrink: 0 !important;
    cursor: pointer !important;
    transition: background 0.15s, color 0.15s, border-color 0.15s;
    margin: 0 !important;
}}

/* ── Hover state ── */
div[data-testid="stRadio"] > div[role="radiogroup"] label:hover {{
    background: {t['card']} !important;
    color: {t['text']} !important;
    border-color: {t['accent']} !important;
}}

/* ── Active / selected pill ── */
div[data-testid="stRadio"] > div[role="radiogroup"] label[data-selected="true"],
div[data-testid="stRadio"] > div[role="radiogroup"] label[aria-checked="true"] {{
    background: {t['accent']} !important;
    color: #ffffff !important;
    border-color: {t['accent']} !important;
    font-weight: 600 !important;
}}

/* ── Hide the radio widget label ── */
div[data-testid="stRadio"] > label {{
    display: none !important;
}}
</style>
""", unsafe_allow_html=True)

top_page = st.radio(
    "Navigate",
    PAGES,
    index=PAGES.index(st.session_state.page),
    horizontal=True,
    label_visibility="collapsed",
    key="top_nav_radio",
)
if top_page != st.session_state.page:
    st.session_state.page = top_page
    st.rerun()

# ─────────────────────────────────────────────────────────────
# GUARD: models not trained yet
# ─────────────────────────────────────────────────────────────
data = _cached_load_all()

if data is None:
    st.markdown(f"""
    <div class="hero-gradient">
        <h1 style="font-size:2rem;margin:0;font-weight:700;">🌬️ Wind Turbine Power Forecasting Dashboard</h1>
        <p style="color:{t['text2']};margin:8px 0 16px 0;font-size:1rem;">
            Upload your wind turbine SCADA / weather dataset to train forecasting models and explore turbine performance.
        </p>
    </div>
    """, unsafe_allow_html=True)

    guide_col1, guide_col2 = st.columns([3, 2])

    with guide_col1:
        st.markdown("### 📋 Required Dataset Format")
        st.markdown(
            "Your CSV must have a **datetime column** and the meteorological / power columns listed below. "
            "The **Power** column must be **normalized** (0–1 scale, where 1 = rated power). "
            "Row order doesn't matter — the pipeline will sort by time automatically."
        )
        req_cols = pd.DataFrame({
            "Column": [
                "Time", "Power",
                "windspeed_10m", "windspeed_100m",
                "windgusts_10m", "winddirection_100m",
                "temperature_2m", "relativehumidity_2m",
                "dewpoint_2m",
            ],
            "Type": [
                "datetime", "float (0–1)",
                "float", "float",
                "float", "float (0–360)",
                "float", "int/float",
                "float",
            ],
            "Description": [
                "Timestamp (e.g. 2022-01-01 00:00)",
                "Normalized power output (actual kW ÷ rated kW)",
                "Wind speed at hub height 10 m (m/s)",
                "Wind speed at hub height 100 m (m/s)",
                "Wind gust speed at 10 m (m/s)",
                "Wind direction at 100 m (degrees from North)",
                "Air temperature at 2 m (°F)",
                "Relative humidity at 2 m (%)",
                "Dew point temperature at 2 m (°F)",
            ],
        })
        st.table(req_cols)

        st.markdown("""
**Tips for a good dataset:**
- Minimum recommended size: **8 000+ rows** (hourly data ≈ 1 year)
- Timestamps should be **equally spaced** (e.g. every hour)
- Remove complete outage windows where the turbine was shut down for maintenance if known
- Missing values (NaN) are handled automatically during preprocessing
        """)

    with guide_col2:
        st.markdown("### 🗂️ Dashboard Sections")
        st.markdown(f"""
<div class="info-box">
<b>🏠 Overview</b><br>
KPI summary cards — average power, capacity factor, best model RMSE — plus monthly production bar chart and seasonal distribution.
</div>
<div class="info-box" style="margin-top:8px;">
<b>📈 Energy Production</b><br>
Interactive time-series of power output and wind speed. Zoom into any period, compare year-over-year trends, and view the power heatmap by hour × weekday.
</div>
<div class="info-box" style="margin-top:8px;">
<b>🔍 Wind & Climate Analysis</b><br>
Feature distributions, correlation matrix, wind rose showing dominant directions and speeds, and scatter plots of power vs. weather variables.
</div>
<div class="info-box" style="margin-top:8px;">
<b>⚡ Turbine Performance</b><br>
Empirical power curve vs. theoretical IEC curve, per-bin mean power with uncertainty, and curtailment detection chart.
</div>
<div class="info-box" style="margin-top:8px;">
<b>🤖 Forecast Model Accuracy</b><br>
Compare all trained models (Linear, Lasso, Ridge, RF, XGBoost, LightGBM, CatBoost). Residual plots, predicted vs. actual, and feature importances.
</div>
<div class="info-box" style="margin-top:8px;">
<b>🎯 Power Forecasting</b><br>
Real-time prediction: set weather conditions, choose Best Model or Ensemble Average, enter your turbine's rated power (kW) and get the forecasted output in kW.
</div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.info("⬆️ Upload your CSV in the **sidebar on the left** to train all models and unlock the dashboard. Training typically takes 2–5 minutes depending on dataset size.")
    st.stop()

df_raw  = data["df_raw"]
comp_df = data["comparison_df"]
preds   = data["predictions"]
fi      = data["feature_importances"]
feat    = data["feature_cols"]
best    = data["best_model"]
scaler  = data["scaler"]
last_r  = data["last_rows"]
frange  = data["feature_ranges"]
meta    = data["meta"]
qreport = data.get("quality_report") or {}
tpl     = T()["plotly_tpl"]
page    = st.session_state.page


# ═════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═════════════════════════════════════════════════════════════
if page == "🏠  Overview":
    t = T()

    st.markdown(f"""
    <div class="hero-gradient">
        <h1 style="font-size:2rem;margin:0;font-weight:700;">🌬️ Wind Turbine Power Forecasting</h1>
        <p style="color:{t['text2']};margin:8px 0 0 0;font-size:1rem;">
            Machine Learning dashboard · Location 1 · {meta.get('train_start', meta['train_end'])[:4]} – {meta['test_end'][:4]}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # KPI Cards
    avg_power  = df_raw["Power"].mean()
    max_power  = 1
    avg_wind   = df_raw["windspeed_100m"].mean()
    cap_factor = avg_power / max_power * 100
    best_rmse  = float(comp_df.iloc[0]["RMSE"])
    best_r2    = float(comp_df.iloc[0]["R2"])

    cols = st.columns(6)
    kpis = [
        ("Avg Power", f"{avg_power:.3f}", "normalized"),
        ("Rated Power", f"{max_power:.3f}", "normalized"),
        ("Capacity Factor", f"{cap_factor:.1f}", "%"),
        ("Avg Wind Speed", f"{avg_wind:.1f}", "m/s"),
        (f"Best RMSE ({meta['best_model_name']})", f"{best_rmse:.4f}", "normalized"),
        ("Best R²", f"{best_r2:.4f}", ""),
    ]
    for col, (label, val, unit) in zip(cols, kpis):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{unit}</div>
                <div class="metric-delta">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Monthly Power Production</div>', unsafe_allow_html=True)
    monthly = df_raw["Power"].resample("M").mean().reset_index()
    monthly.columns = ["Date", "Avg Power"]
    fig = px.bar(
        monthly, x="Date", y="Avg Power",
        color="Avg Power",
        color_continuous_scale=["#1f4e79","#2980b9","#58a6ff","#7ec8e3"],
        template=tpl,
        labels={"Avg Power": "Avg Power (normalized)"},
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False, coloraxis_showscale=False,
        margin=dict(l=0,r=0,t=10,b=0), height=320,
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Hourly Average Power</div>', unsafe_allow_html=True)
        hourly = df_raw.copy()
        hourly["hour"] = hourly.index.hour
        h_avg = hourly.groupby("hour")["Power"].mean().reset_index()
        fig = px.line(h_avg, x="hour", y="Power", markers=True, template=tpl,
                      color_discrete_sequence=["#58a6ff"])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          margin=dict(l=0,r=0,t=10,b=0), height=260)
        fig.update_traces(line_width=2.5)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Seasonal Power Distribution</div>', unsafe_allow_html=True)
        def get_season(m):
            return "Winter" if m in [12,1,2] else "Spring" if m in [3,4,5] else "Summer" if m in [6,7,8] else "Autumn"
        df_s = df_raw.copy(); df_s["season"] = df_s.index.month.map(get_season)
        fig = px.box(df_s, x="season", y="Power", template=tpl,
                     category_orders={"season":["Spring","Summer","Autumn","Winter"]},
                     color="season",
                     color_discrete_sequence=["#3fb950","#f78166","#d29922","#58a6ff"])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          margin=dict(l=0,r=0,t=10,b=0), height=260, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Model comparison summary
    st.markdown('<div class="section-header">Model Comparison Summary</div>', unsafe_allow_html=True)
    fig = go.Figure()
    colors = ["#f78166","#d29922","#58a6ff","#3fb950","#bc8cff","#f0883e","#e3b341","#56d364"]
    for i, row in comp_df.iterrows():
        fig.add_trace(go.Bar(
            name=row["Model"], x=["MAE","RMSE","R²"],
            y=[row["MAE"], row["RMSE"], row["R2"]],
            marker_color=colors[i % len(colors)],
        ))
    fig.update_layout(
        barmode="group", template=tpl,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0,r=0,t=10,b=0), height=340,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════
# PAGE — DATA QUALITY
# ═════════════════════════════════════════════════════════════
elif page == "🔎  Data Quality":
    t = T()
    st.markdown('<h2>🔎 Data Quality Report</h2>', unsafe_allow_html=True)

    orig_rows  = qreport.get("original_rows", len(df_raw))
    final_rows = qreport.get("final_rows",    len(df_raw))
    n_dupes    = qreport.get("n_dupes", 0)
    missing_df = qreport.get("missing_report", pd.DataFrame())
    out_report = qreport.get("outlier_report", {})

    # ── Summary KPI cards ──────────────────────────────────────
    total_missing = int(missing_df["Missing Count"].sum()) if not missing_df.empty else 0
    total_outliers = sum(v["count"] for v in out_report.values()) if out_report else 0

    kpi_cols = st.columns(4)
    kpis_dq = [
        ("Original Rows",   f"{orig_rows:,}",      "before cleaning"),
        ("Cleaned Rows",    f"{final_rows:,}",      "after deduplication"),
        ("Missing Values",  f"{total_missing:,}",   "forward/back-filled"),
        ("Outliers Capped", f"{total_outliers:,}",  "IQR × 3 threshold"),
    ]
    for col, (label, val, sub) in zip(kpi_cols, kpis_dq):
        with col:
            color = t["accent3"] if (val != "0" and label != "Original Rows" and label != "Cleaned Rows") else t["accent"]
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:{color};">{val}</div>
                <div class="metric-label">{sub}</div>
                <div class="metric-delta">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Missing Values ─────────────────────────────────────────
    st.markdown('<div class="section-header">Missing Values</div>', unsafe_allow_html=True)
    if missing_df.empty:
        st.success("✅ No missing values found in the dataset.")
    else:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.dataframe(missing_df.style.background_gradient(subset=["Missing %"], cmap="Reds"), use_container_width=True)
        with col2:
            fig_mv = px.bar(
                missing_df.reset_index().rename(columns={"index": "Column"}),
                x="Column", y="Missing %",
                color="Missing %", color_continuous_scale="Reds",
                template=tpl,
                title="Missing % per Column",
            )
            fig_mv.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=40, b=40), height=320,
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_mv, use_container_width=True)

    # ── Duplicates ─────────────────────────────────────────────
    st.markdown('<div class="section-header">Duplicate Timestamps</div>', unsafe_allow_html=True)
    if n_dupes == 0:
        st.success("✅ No duplicate timestamps found.")
    else:
        st.warning(f"⚠️ **{n_dupes:,}** duplicate timestamp(s) detected and removed (kept first occurrence).")

    # ── Outliers ───────────────────────────────────────────────
    st.markdown('<div class="section-header">Outlier Detection (IQR × 3)</div>', unsafe_allow_html=True)
    if not out_report:
        st.success("✅ No outliers detected using the IQR × 3 threshold.")
    else:
        out_df = pd.DataFrame(out_report).T.reset_index().rename(columns={"index": "Column"})
        out_df["count"] = out_df["count"].astype(int)
        out_df["pct"]   = out_df["pct"].astype(float)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.dataframe(
                out_df[["Column", "count", "pct", "lower", "upper"]]
                    .rename(columns={"count": "# Outliers", "pct": "% Outliers",
                                     "lower": "Cap Lower", "upper": "Cap Upper"})
                    .style.background_gradient(subset=["# Outliers"], cmap="Oranges"),
                use_container_width=True,
            )
        with col2:
            fig_out = px.bar(
                out_df, x="Column", y="count",
                color="pct", color_continuous_scale="Oranges",
                labels={"count": "# Outliers", "pct": "% Outliers"},
                template=tpl,
                title="Outlier Count per Column",
            )
            fig_out.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=40, b=40), height=320,
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_out, use_container_width=True)

        # Box plots for top outlier columns
        st.markdown('<div class="section-header">Distribution of Top Outlier Columns (after capping)</div>', unsafe_allow_html=True)
        top_cols = out_df.sort_values("count", ascending=False)["Column"].head(6).tolist()
        top_cols = [c for c in top_cols if c in df_raw.columns]
        if top_cols:
            n_cols_grid = min(3, len(top_cols))
            rows_grid   = (len(top_cols) + n_cols_grid - 1) // n_cols_grid
            fig_box = make_subplots(rows=rows_grid, cols=n_cols_grid,
                                    subplot_titles=top_cols)
            palette = [t["accent"], t["accent2"], t["accent3"], "#d29922", "#bc8cff", "#f0883e"]
            for idx, col_name in enumerate(top_cols):
                r, ci = divmod(idx, n_cols_grid)
                fig_box.add_trace(
                    go.Box(y=df_raw[col_name].dropna(), name=col_name,
                           marker_color=palette[idx % len(palette)], showlegend=False),
                    row=r + 1, col=ci + 1,
                )
            fig_box.update_layout(
                template=tpl, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=40, b=0), height=360,
            )
            st.plotly_chart(fig_box, use_container_width=True)


# ═════════════════════════════════════════════════════════════
# PAGE — ENERGY PRODUCTION
# ═════════════════════════════════════════════════════════════
elif page == "📈  Energy Production":
    st.markdown('<h2>📈 Energy Production Over Time</h2>', unsafe_allow_html=True)
    t = T()

    col1, col2, col3 = st.columns(3)
    with col1:
        resample_opt = st.selectbox("Resample", ["Hourly (raw)", "Daily", "Weekly", "Monthly"], index=2)
    with col2:
        feature_sel = st.selectbox("Feature", ["Power", "windspeed_100m", "windspeed_10m",
                                                "temperature_2m", "relativehumidity_2m"], index=0)
    with col3:
        years = sorted(df_raw.index.year.unique().tolist())
        default_years = years[-2:] if len(years) >= 2 else years
        year_filter = st.multiselect("Year(s)", years, default=default_years)

    df_ts = df_raw.copy()
    if year_filter:
        df_ts = df_ts[df_ts.index.year.isin(year_filter)]

    rule_map = {"Hourly (raw)": None, "Daily": "D", "Weekly": "W", "Monthly": "M"}
    rule = rule_map[resample_opt]
    if rule:
        df_plot = df_ts[feature_sel].resample(rule).mean().reset_index()
    else:
        df_plot = df_ts[feature_sel].reset_index()
    df_plot.columns = ["Time", feature_sel]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_plot["Time"], y=df_plot[feature_sel],
        mode="lines", name=feature_sel,
        line=dict(color=t["accent"], width=1.5),
        fill="tozeroy",
        fillcolor="rgba(88,166,255,0.07)",
    ))
    fig.update_layout(
        template=tpl, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0,r=0,t=10,b=0), height=380,
        xaxis=dict(rangeslider=dict(visible=True)),
        yaxis_title=feature_sel,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Power Heatmap — Hour × Weekday</div>', unsafe_allow_html=True)
    df_h = df_raw.copy()
    df_h["hour"] = df_h.index.hour
    df_h["weekday"] = df_h.index.day_name()
    hmap = df_h.groupby(["weekday","hour"])["Power"].mean().unstack()
    day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    hmap = hmap.reindex([d for d in day_order if d in hmap.index])
    fig = px.imshow(
        hmap, color_continuous_scale="Blues", aspect="auto",
        labels=dict(x="Hour of Day", y="Day of Week", color="Avg Power"),
        template=tpl,
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=10,b=0), height=300)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Year-over-Year Power</div>', unsafe_allow_html=True)
        df_y = df_raw.copy()
        df_y["month"] = df_y.index.month; df_y["year"] = df_y.index.year
        yoy = df_y.groupby(["year","month"])["Power"].mean().reset_index()
        fig = px.line(yoy, x="month", y="Power", color="year", template=tpl,
                      color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          margin=dict(l=0,r=0,t=10,b=0), height=280,
                          xaxis=dict(tickmode="array",tickvals=list(range(1,13)),
                                     ticktext=["Jan","Feb","Mar","Apr","May","Jun",
                                               "Jul","Aug","Sep","Oct","Nov","Dec"]))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Rolling 30-day Power Trend</div>', unsafe_allow_html=True)
        roll = df_raw["Power"].resample("D").mean().rolling(30).mean().dropna().reset_index()
        roll.columns = ["Date","Rolling Mean"]
        fig = px.area(roll, x="Date", y="Rolling Mean", template=tpl,
                      color_discrete_sequence=[t["accent2"]])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          margin=dict(l=0,r=0,t=10,b=0), height=280)
        st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════
# PAGE 3 — EDA
# ═════════════════════════════════════════════════════════════
elif page == "🔍  Wind & Climate Analysis":
    st.markdown('<h2>🔍 Wind & Climate Analysis</h2>', unsafe_allow_html=True)
    t = T()

    st.markdown('<div class="section-header">Feature Distributions</div>', unsafe_allow_html=True)
    num_cols = ["Power","windspeed_100m","windspeed_10m","temperature_2m",
                "relativehumidity_2m","windgusts_10m"]
    fig = make_subplots(rows=2, cols=3, subplot_titles=num_cols)
    colors = [t["accent"], t["accent2"], t["accent3"], "#d29922", "#bc8cff", "#f0883e"]
    for idx, (col, c) in enumerate(zip(num_cols, colors)):
        r, ci = divmod(idx, 3)
        vals = df_raw[col].dropna()
        fig.add_trace(
            go.Histogram(x=vals, nbinsx=60, marker_color=c, opacity=0.8, showlegend=False),
            row=r+1, col=ci+1,
        )
    fig.update_layout(
        template=tpl, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0,r=0,t=40,b=0), height=440,
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Correlation Matrix</div>', unsafe_allow_html=True)
        corr = df_raw[num_cols].corr()
        fig = px.imshow(
            corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            text_auto=".2f", template=tpl, aspect="auto",
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=10,b=0), height=380)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Wind Rose</div>', unsafe_allow_html=True)
        df_wr = df_raw.copy()
        df_wr["dir_bin"] = (df_wr["winddirection_100m"] // 22.5 * 22.5).astype(int)
        wr = df_wr.groupby("dir_bin").agg(
            avg_speed=("windspeed_100m","mean"),
            count=("windspeed_100m","count"),
        ).reset_index()
        fig = go.Figure(go.Barpolar(
            r=wr["avg_speed"], theta=wr["dir_bin"], width=22,
            marker_color=wr["avg_speed"],
            marker_colorscale="Blues", showlegend=False,
        ))
        fig.update_layout(
            template=tpl, paper_bgcolor="rgba(0,0,0,0)",
            polar=dict(radialaxis=dict(showticklabels=True, ticks="")),
            margin=dict(l=20,r=20,t=20,b=20), height=380,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Power vs Key Features</div>', unsafe_allow_html=True)
    x_feat = st.selectbox("X-axis feature", ["windspeed_100m","windspeed_10m","temperature_2m",
                                              "relativehumidity_2m","windgusts_10m"], key="eda_scatter")
    sample = df_raw.sample(min(5000, len(df_raw)), random_state=42)
    fig = px.scatter(
        sample, x=x_feat, y="Power", opacity=0.35,
        color="Power", color_continuous_scale="Blues",
        template=tpl,
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=0,r=0,t=10,b=0), height=360, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════
# PAGE 4 — POWER CURVE
# ═════════════════════════════════════════════════════════════
elif page == "⚡  Turbine Performance":
    st.markdown('<h2>⚡ Turbine Performance & Power Curve</h2>', unsafe_allow_html=True)
    t = T()

    cut_in  = st.sidebar.slider("Cut-in speed (m/s)", 1.0, 5.0, 3.0, 0.5)
    rated_v = st.sidebar.slider("Rated speed (m/s)", 8.0, 16.0, 12.0, 0.5)
    cut_out = st.sidebar.slider("Cut-out speed (m/s)", 20.0, 30.0, 25.0, 1.0)
    rated_p = 1.0  # Power is normalized to rated power in this dataset

    v_theory = np.linspace(0, 30, 600)
    p_theory = np.zeros_like(v_theory)
    for i, v in enumerate(v_theory):
        if cut_in <= v < rated_v:
            p_theory[i] = rated_p * (v**3 - cut_in**3) / (rated_v**3 - cut_in**3)
        elif rated_v <= v < cut_out:
            p_theory[i] = rated_p
        else:
            p_theory[i] = 0

    sample = df_raw.sample(min(8000, len(df_raw)), random_state=42)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sample["windspeed_100m"], y=sample["Power"],
        mode="markers", name="Actual Data",
        marker=dict(color=t["accent"], size=4, opacity=0.35),
    ))
    fig.add_trace(go.Scatter(
        x=v_theory, y=p_theory,
        mode="lines", name="Theoretical Curve",
        line=dict(color=t["accent3"], width=3),
    ))
    fig.update_layout(
        template=tpl, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0,r=0,t=10,b=0), height=460,
        xaxis_title="Wind Speed at 100m (m/s)",
        yaxis_title="Power (normalized)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Power Binned by Wind Speed</div>', unsafe_allow_html=True)
        df_bin = df_raw.copy()
        df_bin["ws_bin"] = pd.cut(df_bin["windspeed_100m"],
                                   bins=np.arange(0, 26, 1), labels=np.arange(0.5, 25.5, 1))
        bin_stats = df_bin.groupby("ws_bin", observed=False)["Power"].agg(["mean","std","count"]).reset_index()
        bin_stats.columns = ["ws_bin","mean","std","count"]
        bin_stats["ws_bin"] = bin_stats["ws_bin"].astype(float)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=bin_stats["ws_bin"], y=bin_stats["mean"],
            error_y=dict(type="data", array=bin_stats["std"], visible=True),
            name="Mean Power", marker_color=t["accent"],
        ))
        fig.update_layout(
            template=tpl, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0,r=0,t=10,b=0), height=300,
            xaxis_title="Wind Speed Bin (m/s)", yaxis_title="Power (normalized)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Curtailment Detection</div>', unsafe_allow_html=True)
        df_curt = df_raw.copy()
        df_curt["expected"] = 0.0
        for i, row in df_curt.iterrows():
            v = row["windspeed_100m"]
            if cut_in <= v < rated_v:
                df_curt.at[i, "expected"] = rated_p * (v**3 - cut_in**3) / (rated_v**3 - cut_in**3)
            elif rated_v <= v < cut_out:
                df_curt.at[i, "expected"] = rated_p
        df_curt["delta"] = df_curt["Power"] - df_curt["expected"]
        df_curt["ws_bin"] = pd.cut(df_curt["windspeed_100m"], bins=np.arange(0, 26, 1),
                                    labels=np.arange(0.5, 25.5, 1))
        delta_bin = df_curt.groupby("ws_bin", observed=False)["delta"].mean().reset_index()
        delta_bin["ws_bin"] = delta_bin["ws_bin"].astype(float)
        fig = px.bar(delta_bin, x="ws_bin", y="delta",
                     color="delta", color_continuous_scale="RdBu",
                     color_continuous_midpoint=0, template=tpl)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          margin=dict(l=0,r=0,t=10,b=0), height=300,
                          xaxis_title="Wind Speed Bin (m/s)", yaxis_title="Power Delta (normalized)",
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════
# PAGE 5 — MODEL PERFORMANCE
# ═════════════════════════════════════════════════════════════
elif page == "🤖  Forecast Accuracy":
    st.markdown('<h2>🤖 Forecast Model Accuracy</h2>', unsafe_allow_html=True)
    t = T()

    # Metrics table
    st.markdown('<div class="section-header">Metrics Comparison</div>', unsafe_allow_html=True)
    display_df = comp_df.copy()
    display_df["MAE"]  = display_df["MAE"].map("{:.4f}".format)
    display_df["RMSE"] = display_df["RMSE"].map("{:.4f}".format)
    display_df["R2"]   = display_df["R2"].map("{:.4f}".format)
    cols_to_show = ["MAE", "RMSE", "R2"]
    st.dataframe(display_df.set_index("Model")[cols_to_show], use_container_width=True)

    col1, col2 = st.columns([1,2])
    with col1:
        model_sel = st.selectbox("Select model to inspect", list(preds.keys()))

    pred_data = preds[model_sel]
    y_pred    = np.array(pred_data["y_pred"])
    y_actual  = np.array(pred_data["y_actual"])
    ts        = pred_data["ts"]

    with col2:
        fig = px.bar(
            comp_df, x="Model", y="RMSE",
            color="RMSE", color_continuous_scale="Blues_r",
            template=tpl, text="RMSE",
        )
        fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          margin=dict(l=0,r=0,t=10,b=0), height=260,
                          coloraxis_showscale=False, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Predicted vs Actual</div>', unsafe_allow_html=True)
        max_v = max(y_actual.max(), y_pred.max())
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y_actual, y=y_pred, mode="markers",
            marker=dict(color=t["accent"], size=3, opacity=0.3), name="Predictions",
        ))
        fig.add_trace(go.Scatter(
            x=[0, max_v], y=[0, max_v], mode="lines",
            line=dict(color=t["accent3"], dash="dash", width=2), name="Perfect Fit",
        ))
        fig.update_layout(
            template=tpl, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0,r=0,t=10,b=0), height=340,
            xaxis_title="Actual Power (normalized)", yaxis_title="Predicted Power (normalized)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Residuals</div>', unsafe_allow_html=True)
        residuals = y_actual - y_pred
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y_pred, y=residuals, mode="markers",
            marker=dict(color=t["accent2"], size=3, opacity=0.3), name="Residuals",
        ))
        fig.add_hline(y=0, line_dash="dash", line_color=t["accent3"], line_width=2)
        fig.update_layout(
            template=tpl, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0,r=0,t=10,b=0), height=340,
            xaxis_title="Predicted Power (normalized)", yaxis_title="Residual",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Forecast vs Actual — Time Series (last 500 points)</div>', unsafe_allow_html=True)
    n_show = 500
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts[-n_show:], y=y_actual[-n_show:], mode="lines",
                             name="Actual", line=dict(color=t["accent2"], width=1.5)))
    fig.add_trace(go.Scatter(x=ts[-n_show:], y=y_pred[-n_show:], mode="lines",
                             name="Predicted", line=dict(color=t["accent"], width=1.5, dash="dot")))
    fig.update_layout(
        template=tpl, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0,r=0,t=10,b=0), height=320,
        yaxis_title="Power (normalized)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Feature importances
    st.markdown('<div class="section-header">Feature Importances</div>', unsafe_allow_html=True)
    fi_options = [k for k in fi.keys()]
    fi_sel = st.selectbox("Model", fi_options, key="fi_sel")
    fi_data = fi[fi_sel].head(16).reset_index()
    fi_data.columns = ["Feature","Importance"]
    fig = px.bar(fi_data, x="Importance", y="Feature", orientation="h",
                 color="Importance", color_continuous_scale="Blues",
                 template=tpl)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0,r=0,t=10,b=0), height=420,
        yaxis=dict(autorange="reversed"), coloraxis_showscale=False,
    )
    st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════
# PAGE 6 — INTERACTIVE FORECASTING
# ═════════════════════════════════════════════════════════════
elif page == "🎯  Power Forecasting":
    st.markdown('<h2>🎯 Wind Turbine Power Forecasting</h2>', unsafe_allow_html=True)
    t = T()

    # ── Turbine rated power & model choice ─────────────────────
    cfg_col1, cfg_col2, cfg_col3 = st.columns([1, 1, 2])
    with cfg_col1:
        rated_power_input = st.number_input(
            "Turbine Rated Power (kW)",
            min_value=1.0,
            max_value=20000.0,
            value=2000.0,
            step=100.0,
            help="Enter your turbine's rated (nameplate) power in kW. "
                 "The model predicts normalized power (0–1); this scales it to real kW.",
        )
    with cfg_col2:
        forecast_mode = st.radio(
            "Forecast Mode",
            ["Best Model", "Ensemble Average"],
            index=0,
            help="Best Model uses the top-ranked model by RMSE. "
                 "Ensemble Average averages predictions from all trained models.",
        )
    with cfg_col3:
        if forecast_mode == "Best Model":
            st.info(f"🏆 Using **{meta['best_model_name']}** (lowest RMSE on test set).")
        else:
            st.info(f"🔀 Ensemble of **{len(preds)} models** — their predictions will be averaged.")

    with st.expander("⚙️ Weather Conditions & Context", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Wind**")
            windspeed_100m = st.slider("Wind Speed 100m (m/s)",
                                       float(frange["windspeed_100m"]["min"]),
                                       float(frange["windspeed_100m"]["max"]),
                                       float(frange["windspeed_100m"]["mean"]),
                                       0.1)
            windspeed_10m  = st.slider("Wind Speed 10m (m/s)",
                                       float(frange["windspeed_10m"]["min"]),
                                       float(frange["windspeed_10m"]["max"]),
                                       float(frange["windspeed_10m"]["mean"]),
                                       0.1)
            windgusts_10m  = st.slider("Wind Gusts 10m (m/s)",
                                       float(frange["windgusts_10m"]["min"]),
                                       float(frange["windgusts_10m"]["max"]),
                                       float(frange["windgusts_10m"]["mean"]),
                                       0.1)
            wind_dir       = st.slider("Wind Direction 100m (°)", 0, 360, 180)
        with col2:
            st.markdown("**Atmosphere**")
            temperature = st.slider("Temperature (°F)",
                                    float(frange["temperature_2m"]["min"]),
                                    float(frange["temperature_2m"]["max"]),
                                    float(frange["temperature_2m"]["mean"]))
            dewpoint    = st.slider("Dew Point (°F)",
                                    float(frange["dewpoint_2m"]["min"]),
                                    float(frange["dewpoint_2m"]["max"]),
                                    float(frange["dewpoint_2m"]["mean"]))
            humidity    = st.slider("Rel. Humidity (%)",
                                    int(frange["relativehumidity_2m"]["min"]),
                                    int(frange["relativehumidity_2m"]["max"]),
                                    int(frange["relativehumidity_2m"]["mean"]))
        with col3:
            st.markdown("**Context**")
            season_sel = st.selectbox("Season", ["Winter", "Spring", "Summer", "Autumn"])
            turb_sel   = st.selectbox("Turbulence", ["Low", "Moderate", "High"])
            st.markdown(
                "<div style='font-size:0.75rem;color:#8b949e;margin-top:8px;'>"
                "Lag and rolling features are set automatically assuming steady wind conditions.</div>",
                unsafe_allow_html=True,
            )

    def build_input_row():
        # ── Original dataset features ────────────────────────────
        row = {}
        row["windspeed_100m"]      = windspeed_100m
        row["windspeed_10m"]       = windspeed_10m
        row["windgusts_10m"]       = windgusts_10m
        row["winddirection_100m"]  = wind_dir
        row["temperature_2m"]      = temperature
        row["dewpoint_2m"]         = dewpoint
        row["relativehumidity_2m"] = humidity

        # ── Derived features (same formulas as _engineer_features) ──
        temp_K_val = (temperature - 32) * 5.0 / 9.0 + 273.15
        row["temp_K"]               = temp_K_val
        row["air_density"]          = 101325 / (287.05 * temp_K_val)
        row["windspeed_100m_cubed"] = windspeed_100m ** 3
        row["wind_dir_sin"]         = np.sin(np.radians(wind_dir))
        row["wind_dir_cos"]         = np.cos(np.radians(wind_dir))

        # ── Season / turbulence dummies ──────────────────────────
        row["season_Spring"] = 1.0 if season_sel == "Spring" else 0.0
        row["season_Summer"] = 1.0 if season_sel == "Summer" else 0.0
        row["season_Winter"] = 1.0 if season_sel == "Winter" else 0.0
        row["turbulence_category_Moderate"] = 1.0 if turb_sel == "Moderate" else 0.0
        row["turbulence_category_High"]     = 1.0 if turb_sel == "High"     else 0.0

        # ── Wind speed lags & rolling — assume steady wind ───────
        for lag in [1, 2, 3, 6, 12, 24]:
            row[f"windspeed100_lag{lag}"] = windspeed_100m
        row["windspeed100_roll3"] = windspeed_100m

        # ── Power lags — theoretical power curve at current wind speed ──
        cut_in  = float(meta["cut_in"])
        rated_v = float(meta["rated_speed"])
        cut_out = float(meta["cut_out"])
        v = windspeed_100m
        if v < cut_in or v >= cut_out:
            p_theory = 0.0
        elif v < rated_v:
            p_theory = (v**3 - cut_in**3) / (rated_v**3 - cut_in**3)
        else:
            p_theory = 1.0
        for lag in [1, 2, 3, 12, 24]:
            row[f"Power_lag{lag}"] = p_theory
        row["Power_roll3"] = p_theory

        return row

    input_row = build_input_row()
    lr_median = last_r[feat].median()
    for f in feat:
        if f not in input_row:
            input_row[f] = lr_median[f]

    X_input = pd.DataFrame([input_row])[feat]

    model_obj  = best["model"]
    model_name = best["name"]
    linear_models = ["Linear Regression", "Lasso", "Ridge"]

    def _predict_single(m_obj, m_name, X_raw):
        if m_name in linear_models:
            return float(np.expm1(m_obj.predict(scaler.transform(X_raw))[0]))
        else:
            return float(m_obj.predict(X_raw)[0])

    if forecast_mode == "Best Model":
        prediction_norm = _predict_single(model_obj, model_name, X_input)
    else:
        # Ensemble: average across all loaded models
        all_model_objects = _cached_load_ensemble_models()
        preds_ensemble = [
            _predict_single(m, n, X_input)
            for n, m in all_model_objects.items() if m is not None
        ]
        prediction_norm = float(np.mean(preds_ensemble)) if preds_ensemble else 0.0

    # Clamp to [0, 1] since power is normalized
    prediction_norm = max(0.0, min(prediction_norm, 1.0))
    # Scale to actual kW using user's rated power
    prediction = prediction_norm * rated_power_input
    pct_rated   = prediction_norm * 100

    st.markdown("---")
    col1, col2, col3 = st.columns([1,1,2])

    avg_power_actual = df_raw["Power"].mean() * rated_power_input

    with col1:
        st.markdown(f"""
        <div class="metric-card" style="margin-bottom:12px;">
            <div class="metric-value" style="font-size:2.4rem;">{prediction:.1f}</div>
            <div class="metric-label">kW  (normalized: {prediction_norm:.3f})</div>
            <div class="metric-delta">Predicted Power Output</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        color = t["accent2"] if pct_rated > 50 else t["accent3"] if pct_rated < 20 else t["accent"]
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="font-size:2.4rem;color:{color};">{pct_rated:.1f}%</div>
            <div class="metric-label">of Rated Capacity</div>
            <div class="metric-delta">Rated: {rated_power_input:,.0f} kW</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prediction,
            delta={"reference": avg_power_actual, "valueformat": ".1f"},
            number={"suffix": " kW", "valueformat": ".1f"},
            title={"text": "Power Output<br><span style='font-size:0.75rem;color:gray'>vs dataset avg</span>"},
            gauge={
                "axis": {"range": [0, rated_power_input], "tickformat": ".0f"},
                "bar": {"color": t["accent"]},
                "steps": [
                    {"range": [0, rated_power_input * 0.3], "color": t["bg2"]},
                    {"range": [rated_power_input * 0.3, rated_power_input * 0.7], "color": t["card"]},
                    {"range": [rated_power_input * 0.7, rated_power_input], "color": t["bg2"]},
                ],
                "threshold": {
                    "line": {"color": t["accent3"], "width": 3},
                    "thickness": 0.75,
                    "value": avg_power_actual,
                },
            },
        ))
        fig.update_layout(
            template=tpl, paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20,r=20,t=30,b=10), height=220,
            font_color=t["text"],
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Wind Speed Scenario Comparison</div>', unsafe_allow_html=True)
    scenarios = {"Low Wind (5 m/s)": 5.0, "Med Wind (10 m/s)": 10.0,
                 "High Wind (15 m/s)": 15.0, "Your Setting": windspeed_100m}
    scenario_preds = {}
    cut_in_s  = float(meta["cut_in"])
    rated_v_s = float(meta["rated_speed"])
    cut_out_s = float(meta["cut_out"])
    for label, ws_val in scenarios.items():
        r2 = dict(input_row)
        # Wind lags — steady wind assumption
        for lag in [1, 2, 3, 6, 12, 24]:
            r2[f"windspeed100_lag{lag}"] = ws_val
        r2["windspeed100_roll3"]    = ws_val
        r2["windspeed_100m"]        = ws_val
        r2["windspeed_100m_cubed"]  = ws_val ** 3
        # Power lags — theoretical power curve at scenario wind speed
        v_s = ws_val
        if v_s < cut_in_s or v_s >= cut_out_s:
            p_s = 0.0
        elif v_s < rated_v_s:
            p_s = (v_s**3 - cut_in_s**3) / (rated_v_s**3 - cut_in_s**3)
        else:
            p_s = 1.0
        for lag in [1, 2, 3, 12, 24]:
            r2[f"Power_lag{lag}"] = p_s
        r2["Power_roll3"] = p_s
        for f in feat:
            if f not in r2:
                r2[f] = lr_median[f]
        X2 = pd.DataFrame([r2])[feat]
        p2_norm = _predict_single(model_obj, model_name, X2)
        p2_norm = max(0.0, min(p2_norm, 1.0))
        scenario_preds[label] = p2_norm * rated_power_input

    sc_df = pd.DataFrame({"Scenario": list(scenario_preds.keys()),
                           "Predicted Power (kW)": list(scenario_preds.values())})
    sc_colors = [t["text2"], t["accent2"], t["accent3"], t["accent"]]
    fig = px.bar(sc_df, x="Scenario", y="Predicted Power (kW)",
                 color="Scenario",
                 color_discrete_sequence=sc_colors,
                 template=tpl, text="Predicted Power (kW)")
    fig.update_traces(texttemplate="%{text:.1f} kW", textposition="outside")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0,r=0,t=10,b=40), height=320, showlegend=False,
        yaxis=dict(range=[0, rated_power_input * 1.15]),
    )
    st.plotly_chart(fig, use_container_width=True)
