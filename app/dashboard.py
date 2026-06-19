"""
Streamlit app — three pages:
  1. Single Prediction  (manual form + 🤖 Auto-Fill from Location)
  2. Batch Prediction   (CSV upload)
  3. Monitoring Dashboard
"""

import io
import json
import sqlite3
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent))
from predict import predict, predict_batch, load_model
from location_enricher import enrich_location

DB_PATH = os.environ.get("LOG_DB", "model/predictions.db")

def _log_prediction(mode: str, input_json: str, score: float, label: str):
    """Write a prediction to the SQLite log so the Monitoring page can read it."""
    try:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        con = sqlite3.connect(DB_PATH)
        con.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                ts         TEXT,
                mode       TEXT,
                input_json TEXT,
                score      REAL,
                label      TEXT
            )
        """)
        con.execute(
            "INSERT INTO predictions (ts, mode, input_json, score, label) VALUES (?,?,?,?,?)",
            (pd.Timestamp.utcnow().isoformat(), mode, input_json, score, label)
        )
        con.commit()
        con.close()
    except Exception:
        pass  # never block the UI on a logging failure

st.set_page_config(
    page_title="Flood Risk Predictor",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ──────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1f3a 0%, #0f2d52 60%, #0b3d6b 100%);
    border-right: 1px solid rgba(255,255,255,0.07);
  }
  [data-testid="stSidebar"] * { color: #c8ddf0 !important; }
  [data-testid="stSidebar"] .stRadio label { color: #c8ddf0 !important; }
  [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.12) !important; }
  [data-testid="stSidebar"] .stSuccess {
    background: rgba(40,167,69,0.15) !important;
    border: 1px solid rgba(40,167,69,0.35) !important;
    border-radius: 8px;
  }
  [data-testid="stSidebar"] .stError {
    background: rgba(220,53,69,0.15) !important;
    border: 1px solid rgba(220,53,69,0.35) !important;
    border-radius: 8px;
  }

  /* ── Main background ── */
  .main .block-container {
    background: #f4f7fb;
    padding-top: 2rem;
    padding-bottom: 3rem;
  }

  /* ── Page titles ── */
  h1 {
    font-weight: 700 !important;
    color: #0b1f3a !important;
    letter-spacing: -0.5px;
  }
  h2, h3 { color: #0f2d52 !important; font-weight: 600 !important; }

  /* ── Risk result box ── */
  .risk-box {
    border-radius: 14px;
    padding: 28px 36px;
    text-align: center;
    font-size: 2rem;
    font-weight: 700;
    margin: 16px 0;
    letter-spacing: -0.3px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.10);
  }
  .risk-low       { background: linear-gradient(135deg,#d4f5e2,#b8ebd0); color:#0d5c2e; border: 1.5px solid #a3ddb9; }
  .risk-moderate  { background: linear-gradient(135deg,#fff4d6,#ffe9a0); color:#7a5700; border: 1.5px solid #f5d76e; }
  .risk-high      { background: linear-gradient(135deg,#ffe8d0,#ffcfa0); color:#b84100; border: 1.5px solid #ffb36b; }
  .risk-very-high { background: linear-gradient(135deg,#ffd6d9,#ffb0b6); color:#7a0c15; border: 1.5px solid #ff8590; }

  /* ── Auto-fill banner ── */
  .autofill-banner {
    background: linear-gradient(135deg, #eaf3ff 0%, #f0faee 100%);
    border-left: 4px solid #1a6cc4;
    border-radius: 10px;
    padding: 14px 20px;
    margin: 12px 0 20px 0;
    box-shadow: 0 2px 8px rgba(26,108,196,0.08);
  }
  .source-tag {
    display: inline-block;
    background: #ddeeff;
    border: 1px solid #b5d4f5;
    border-radius: 20px;
    padding: 3px 12px;
    margin: 3px 4px;
    font-size: 0.80rem;
    color: #1348a0;
    font-weight: 500;
  }

  /* ── Metric cards ── */
  [data-testid="stMetric"] {
    background: #ffffff;
    border-radius: 12px;
    padding: 16px 20px !important;
    box-shadow: 0 2px 10px rgba(11,31,58,0.08);
    border: 1px solid #e2e8f4;
  }
  [data-testid="stMetricLabel"] { color: #5a6e8c !important; font-size: 0.82rem !important; font-weight: 500 !important; text-transform: uppercase; letter-spacing: 0.05em; }
  [data-testid="stMetricValue"] { color: #0b1f3a !important; font-weight: 700 !important; }

  /* ── Buttons ── */
  .stButton > button[kind="primary"], .stButton > button {
    background: linear-gradient(135deg, #1a6cc4, #0f4a8a) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 10px 22px !important;
    transition: opacity 0.18s ease, box-shadow 0.18s ease !important;
    box-shadow: 0 3px 10px rgba(15,74,138,0.25) !important;
  }
  .stButton > button:hover {
    opacity: 0.90 !important;
    box-shadow: 0 5px 16px rgba(15,74,138,0.35) !important;
  }

  /* ── Form inputs ── */
  .stTextInput input, .stNumberInput input, .stSelectbox select {
    border-radius: 8px !important;
    border: 1.5px solid #d0daea !important;
    background: #ffffff !important;
    color: #0b1f3a !important;
    transition: border-color 0.2s;
  }
  .stTextInput input:focus, .stNumberInput input:focus {
    border-color: #1a6cc4 !important;
    box-shadow: 0 0 0 3px rgba(26,108,196,0.12) !important;
  }
  .stForm { border: none !important; background: transparent !important; }

  /* ── Section subheaders ── */
  .section-card {
    background: #ffffff;
    border-radius: 14px;
    padding: 20px 22px 10px 22px;
    margin-bottom: 16px;
    border: 1px solid #e2e8f4;
    box-shadow: 0 2px 8px rgba(11,31,58,0.05);
  }

  /* ── Divider ── */
  hr { border: none; border-top: 1.5px solid #e2e8f4 !important; margin: 20px 0 !important; }

  /* ── Dataframe ── */
  [data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

  /* ── Caption / small text ── */
  .stCaption { color: #6b7fa8 !important; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar nav ──────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/flood.png", width=60)
    st.title("Flood Risk\nPredictor")
    st.caption("ML Opsidian: Genesis")
    st.markdown("---")
    page = st.radio("Navigate", ["🔍 Single Prediction", "📂 Batch Prediction", "📊 Monitoring"])
    st.markdown("---")
    try:
        load_model()
        st.success("✅ Model loaded")
    except FileNotFoundError:
        st.error("⚠️ Model not found.\nRun `train.py` first.")


# ── Helpers ──────────────────────────────────────────────────────
RISK_COLORS = {"Low": "#28a745", "Moderate": "#ffc107", "High": "#fd7e14", "Very High": "#dc3545"}
RISK_CLASS  = {"Low": "risk-low", "Moderate": "risk-moderate", "High": "risk-high", "Very High": "risk-very-high"}

def gauge(score, y_min, y_max):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"font": {"size": 36}},
        gauge={
            "axis": {"range": [y_min, y_max], "tickwidth": 1},
            "bar":  {"color": "#1a4a8a"},
            "steps": [
                {"range": [y_min,  y_min + (y_max-y_min)*0.25], "color": "#d4edda"},
                {"range": [y_min + (y_max-y_min)*0.25, y_min + (y_max-y_min)*0.50], "color": "#fff3cd"},
                {"range": [y_min + (y_max-y_min)*0.50, y_min + (y_max-y_min)*0.75], "color": "#ffe0b2"},
                {"range": [y_min + (y_max-y_min)*0.75, y_max], "color": "#f8d7da"},
            ],
            "threshold": {"line": {"color": "#1a4a8a", "width": 4}, "value": score},
        },
    ))
    fig.update_layout(height=260, margin=dict(t=20, b=10, l=20, r=20))
    return fig

def _load_logs(limit=500):
    if not os.path.exists(DB_PATH):
        return pd.DataFrame(columns=["id","ts","mode","score","label"])
    con = sqlite3.connect(DB_PATH)
    df  = pd.read_sql("SELECT * FROM predictions ORDER BY id DESC LIMIT ?", con, params=(limit,))
    con.close()
    return df


# ════════════════════════════════════════════════════════════════
# PAGE 1 — SINGLE PREDICTION
# ════════════════════════════════════════════════════════════════
if page == "🔍 Single Prediction":
    st.title("🌊 Flood Risk — Single Location")

    # ── AUTO-FILL SECTION ────────────────────────────────────────
    st.markdown("### 🤖 Smart Location Auto-Fill")
    st.markdown(
        "Enter a location name **or** coordinates below and click **Fetch Data** — "
        "the form will be automatically filled with real weather, elevation, and "
        "terrain data fetched live from free APIs."
    )

    af_col1, af_col2, af_col3, af_col4 = st.columns([3, 1.5, 1.5, 1.2])
    with af_col1:
        autofill_place = st.text_input("📍 Location name", placeholder="e.g. Colombo, Galle, Kandy, Sri Lanka")
    with af_col2:
        autofill_lat = st.number_input("Latitude (optional)", value=None, format="%.4f", placeholder="6.9271")
    with af_col3:
        autofill_lon = st.number_input("Longitude (optional)", value=None, format="%.4f", placeholder="79.8612")
    with af_col4:
        st.write("")
        st.write("")
        fetch_clicked = st.button("🌐 Fetch Data", use_container_width=True, type="primary")

    # ── initialise session state ─────────────────────────────────
    if "af_data" not in st.session_state:
        st.session_state.af_data = {}

    if fetch_clicked:
        if not autofill_place and autofill_lat is None:
            st.warning("Please enter a location name or coordinates first.")
        else:
            with st.spinner("Fetching live data from APIs…"):
                fetched = enrich_location(
                    place_name=autofill_place or None,
                    lat=autofill_lat,
                    lon=autofill_lon,
                )
            st.session_state.af_data = fetched

    af = st.session_state.af_data

    # Show banner if data was fetched
    if af and af.get("meta", {}).get("sources"):
        meta = af["meta"]
        display_name = meta.get("display_name", "")
        sources_html = "".join(f'<span class="source-tag">{s}</span>' for s in meta["sources"])
        st.markdown(f"""
        <div class="autofill-banner">
          <b>✅ Auto-filled from:</b> {display_name}<br>
          <div style="margin-top:6px">{sources_html}</div>
        </div>
        """, unsafe_allow_html=True)
        for w in meta.get("warnings", []):
            st.warning(w)

    st.markdown("---")
    st.markdown("### ✏️ Prediction Form")
    st.caption("Fields pre-filled from live APIs are highlighted. You can edit any value before predicting.")

    # ── FORM ─────────────────────────────────────────────────────
    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("📍 Location")
            district   = st.text_input("District",   value=af.get("district",   ""), placeholder="e.g. Colombo")
            place_name = st.text_input("Place Name", value=af.get("place_name", ""), placeholder="e.g. Mirpur")
            latitude   = st.number_input("Latitude",  value=af.get("latitude"),  format="%.4f", placeholder="6.9271")
            longitude  = st.number_input("Longitude", value=af.get("longitude"), format="%.4f", placeholder="79.8612")
            elevation  = st.number_input(
                "Elevation (m)" + (" 🔄" if "elevation_m" in af else ""),
                value=af.get("elevation_m"),
                placeholder="5.0"
            )

        with col2:
            st.subheader("🌧️ Hydrology & Weather")
            rainfall_7d    = st.number_input(
                "Rainfall 7-day (mm)" + (" 🔄" if "rainfall_7d_mm" in af else ""),
                value=af.get("rainfall_7d_mm"), placeholder="120.0"
            )
            rainfall_month = st.number_input(
                "Monthly Rainfall (mm)" + (" 🔄" if "monthly_rainfall_mm" in af else ""),
                value=af.get("monthly_rainfall_mm"), placeholder="400.0"
            )
            river_dist     = st.number_input(
                "Distance to River (m)" + (" 🔄" if "distance_to_river_m" in af else ""),
                value=af.get("distance_to_river_m"), placeholder="250.0"
            )
            inundation     = st.number_input("Inundation Area (sqm)", value=None, placeholder="5000.0")
            drainage       = st.number_input(
                "Drainage Index" + (" 🔄" if "drainage_index" in af else ""),
                value=af.get("drainage_index"), placeholder="0.6"
            )
            ew_index       = st.number_input(
                "Extreme Weather Index" + (" 🔄" if "extreme_weather_index" in af else ""),
                value=af.get("extreme_weather_index"), placeholder="0.7"
            )
            seasonal       = st.number_input(
                "Seasonal Index" + (" 🔄" if "seasonal_index" in af else ""),
                value=af.get("seasonal_index"), placeholder="0.5"
            )
            flood_count    = st.number_input("Historical Flood Count", value=None, placeholder="3.0")

            _flood_occ_opts = ["", "Yes", "No"]
            _flood_occ_def  = af.get("flood_occurrence_current_event", "")
            _flood_occ_idx  = _flood_occ_opts.index(_flood_occ_def) if _flood_occ_def in _flood_occ_opts else 0
            flood_occ_lbl = "Flood Occurrence (current)?" + (" 🔄" if "flood_occurrence_current_event" in af else "")
            flood_occ      = st.selectbox(flood_occ_lbl, _flood_occ_opts, index=_flood_occ_idx)

            _wp_opts = ["", "Likely", "Unlikely"]
            _wp_def  = af.get("water_presence_flag", "")
            _wp_idx  = _wp_opts.index(_wp_def) if _wp_def in _wp_opts else 0
            water_pres = st.selectbox(
                "Water Presence Flag" + (" 🔄" if "water_presence_flag" in af else ""),
                _wp_opts, index=_wp_idx
            )

        with col3:
            st.subheader("🏘️ Land & Society")
            terrain        = st.number_input(
                "Terrain Roughness Index" + (" 🔄" if "terrain_roughness_index" in af else ""),
                value=af.get("terrain_roughness_index"), placeholder="0.4"
            )
            ndvi           = st.number_input(
                "NDVI" + (" 🔄" if "ndvi" in af else ""),
                value=af.get("ndvi"), placeholder="0.3"
            )
            ndwi           = st.number_input(
                "NDWI" + (" 🔄" if "ndwi" in af else ""),
                value=af.get("ndwi"), placeholder="0.1"
            )
            built_up       = st.number_input(
                "Built-up Percent (%)" + (" 🔄" if "built_up_percent" in af else ""),
                value=af.get("built_up_percent"), placeholder="45.0"
            )
            pop_density    = st.number_input("Population Density (/km²)", value=None, placeholder="8000.0")
            socio          = st.number_input("Socioeconomic Status Index", value=None, placeholder="0.5")
            infra          = st.number_input("Infrastructure Score",        value=None, placeholder="0.6")
            hospital_km    = st.number_input("Nearest Hospital (km)",       value=None, placeholder="2.5")
            evac_km        = st.number_input("Nearest Evac Site (km)",      value=None, placeholder="1.0")
            road_q         = st.selectbox("Road Quality",     ["", "Good (paved)", "Fair", "Poor (unpaved)", "No road access"])
            electricity_s  = st.selectbox("Electricity",      ["", "Grid", "Mixed", "Off-grid (solar)"])

            _ws_opts = ["", "Municipal", "Well", "Tube-well", "Rainwater harvesting", "Surface water"]
            _ws_def  = af.get("water_supply", "")
            _ws_idx  = _ws_opts.index(_ws_def) if _ws_def in _ws_opts else 0
            water_sup = st.selectbox(
                "Water Supply" + (" 🔄" if "water_supply" in af else ""),
                _ws_opts, index=_ws_idx
            )

            soil           = st.selectbox("Soil Type",        ["", "Clay", "Sandy", "Loamy", "Silty", "Peaty"])
            urban_r        = st.selectbox("Urban / Rural",    ["", "Urban", "Rural"])

        submitted = st.form_submit_button("🔮 Predict Flood Risk", use_container_width=True)

    st.caption("🔄 = auto-filled from live API data")

    if submitted:
        row = {}
        fields = {
            "district": district, "place_name": place_name,
            "latitude": latitude, "longitude": longitude, "elevation_m": elevation,
            "rainfall_7d_mm": rainfall_7d, "monthly_rainfall_mm": rainfall_month,
            "distance_to_river_m": river_dist, "inundation_area_sqm": inundation,
            "drainage_index": drainage, "extreme_weather_index": ew_index,
            "seasonal_index": seasonal, "historical_flood_count": flood_count,
            "flood_occurrence_current_event": flood_occ, "water_presence_flag": water_pres,
            "terrain_roughness_index": terrain, "ndvi": ndvi, "ndwi": ndwi,
            "built_up_percent": built_up, "population_density_per_km2": pop_density,
            "socioeconomic_status_index": socio, "infrastructure_score": infra,
            "nearest_hospital_km": hospital_km, "nearest_evac_km": evac_km,
            "road_quality": road_q, "electricity": electricity_s, "water_supply": water_sup,
            "soil_type": soil, "urban_rural": urban_r,
        }
        for k, v in fields.items():
            if v not in (None, ""):
                row[k] = v

        with st.spinner("Running ensemble model..."):
            try:
                result = predict(row)
            except FileNotFoundError:
                st.error("Model file not found. Please run `train.py` first.")
                st.stop()
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

        score = result["flood_risk_score"]
        label = result["risk_label"]
        art   = load_model()
        _log_prediction("single", json.dumps(row), score, label)

        st.markdown("---")
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown(f"""
            <div class="risk-box {RISK_CLASS[label]}">
                {label} Risk<br>
                <span style="font-size:1.1rem;font-weight:400;">Score: {score:.4f}</span>
            </div>""", unsafe_allow_html=True)
            # Show which inputs were auto-filled
            auto_keys = [k for k in row if k in af and k != "meta"]
            if auto_keys:
                st.info(f"🔄 **{len(auto_keys)} fields** were auto-filled from live APIs: {', '.join(auto_keys)}")
        with c2:
            st.plotly_chart(gauge(score, art["Y_MIN"], art["Y_MAX"]), use_container_width=True)

        with st.expander("See raw inputs sent to model"):
            st.json(row)


# ════════════════════════════════════════════════════════════════
# PAGE 2 — BATCH PREDICTION
# ════════════════════════════════════════════════════════════════
elif page == "📂 Batch Prediction":
    st.title("📂 Batch Prediction — CSV Upload")
    st.markdown("Upload a CSV with your feature columns. The model will score every row.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df_in = pd.read_csv(uploaded)
        st.write(f"**{len(df_in):,} rows × {len(df_in.columns)} columns detected**")
        st.dataframe(df_in.head(5), use_container_width=True)

        if st.button("🔮 Run Batch Predictions", use_container_width=True):
            with st.spinner(f"Scoring {len(df_in):,} rows..."):
                try:
                    result_df = predict_batch(df_in)
                except FileNotFoundError:
                    st.error("Model file not found. Please run `train.py` first.")
                    st.stop()
                except Exception as e:
                    st.error(f"Batch prediction failed: {e}")
                    st.stop()

            st.success(f"Done! Scored {len(result_df):,} rows.")

            # Log every row to the monitoring DB
            for _, r in result_df.iterrows():
                _log_prediction("batch", "{}", float(r["flood_risk_score"]), r["risk_label"])

            col1, col2 = st.columns(2)
            with col1:
                label_counts = result_df["risk_label"].value_counts().reset_index()
                label_counts.columns = ["Risk Level", "Count"]
                fig = px.pie(label_counts, names="Risk Level", values="Count",
                             color="Risk Level", color_discrete_map=RISK_COLORS,
                             title="Risk Level Distribution")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig2 = px.histogram(result_df, x="flood_risk_score", nbins=40,
                                    title="Score Distribution",
                                    color_discrete_sequence=["#1a4a8a"])
                st.plotly_chart(fig2, use_container_width=True)

            out_df = pd.concat([df_in.reset_index(drop=True),
                                 result_df[["flood_risk_score","risk_label"]].reset_index(drop=True)], axis=1)
            st.dataframe(out_df, use_container_width=True)

            csv_bytes = out_df.to_csv(index=False).encode()
            st.download_button("⬇️ Download Results CSV", csv_bytes,
                               "flood_risk_predictions.csv", "text/csv",
                               use_container_width=True)


# ════════════════════════════════════════════════════════════════
# PAGE 3 — MONITORING DASHBOARD
# ════════════════════════════════════════════════════════════════
elif page == "📊 Monitoring":
    st.title("📊 Prediction Monitoring Dashboard")

    logs = _load_logs()
    if logs.empty:
        st.info("No predictions logged yet. Make some predictions first!")
        st.stop()

    logs["ts"]    = pd.to_datetime(logs["ts"])
    logs["score"] = pd.to_numeric(logs["score"], errors="coerce")

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Predictions", f"{len(logs):,}")
    k2.metric("Avg Risk Score",    f"{logs['score'].mean():.4f}")
    k3.metric("Max Risk Score",    f"{logs['score'].max():.4f}")
    k4.metric("High/Very High",
              f"{(logs['label'].isin(['High','Very High'])).sum():,}")
    k5.metric("Unique Modes",      logs["mode"].nunique())

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        label_counts = logs["label"].value_counts().reset_index()
        label_counts.columns = ["Risk Level", "Count"]
        fig = px.bar(label_counts, x="Risk Level", y="Count",
                     color="Risk Level", color_discrete_map=RISK_COLORS,
                     title="Predictions by Risk Level")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.histogram(logs, x="score", nbins=40,
                            title="Score Distribution (all predictions)",
                            color_discrete_sequence=["#1a4a8a"])
        st.plotly_chart(fig2, use_container_width=True)

    logs_sorted = logs.sort_values("ts")
    fig3 = px.scatter(logs_sorted, x="ts", y="score", color="label",
                      color_discrete_map=RISK_COLORS,
                      title="Predictions Over Time",
                      labels={"ts": "Timestamp", "score": "Flood Risk Score"})
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Recent Prediction Log")
    st.dataframe(logs[["id","ts","mode","score","label"]].head(50),
                 use_container_width=True)

    csv_bytes = logs.to_csv(index=False).encode()
    st.download_button("⬇️ Export Full Log", csv_bytes, "prediction_log.csv", "text/csv")
