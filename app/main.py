"""
FastAPI backend — exposes:
  POST /predict          single-row prediction
  POST /predict/batch    CSV upload → predictions
  GET  /health           liveness check
  GET  /metrics          prediction log summary
  GET  /logs             raw prediction history (JSON)
"""

import io
import os
import json
import sqlite3
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# allow running from repo root
import sys
sys.path.insert(0, str(Path(__file__).parent))
from predict import predict, predict_batch, load_model

# ── SQLite prediction log ────────────────────────────────────────
DB_PATH = os.environ.get("LOG_DB", "model/predictions.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def _init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            ts        TEXT,
            mode      TEXT,
            input_json TEXT,
            score     REAL,
            label     TEXT
        )
    """)
    con.commit(); con.close()

_init_db()

def _log(mode: str, input_json: str, score: float, label: str):
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "INSERT INTO predictions (ts, mode, input_json, score, label) VALUES (?,?,?,?,?)",
        (datetime.datetime.utcnow().isoformat(), mode, input_json, score, label)
    )
    con.commit(); con.close()


# ── App ──────────────────────────────────────────────────────────
app = FastAPI(
    title="Flood Risk Prediction API",
    description="ML Opsidian: Genesis — production flood risk scoring service",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ── Request schema ───────────────────────────────────────────────
class PredictRequest(BaseModel):
    # Geographic
    district:                        Optional[str]   = None
    place_name:                      Optional[str]   = None
    latitude:                        Optional[float] = None
    longitude:                       Optional[float] = None
    elevation_m:                     Optional[float] = None
    # Hydro / weather
    rainfall_7d_mm:                  Optional[float] = None
    monthly_rainfall_mm:             Optional[float] = None
    distance_to_river_m:             Optional[float] = None
    inundation_area_sqm:             Optional[float] = None
    drainage_index:                  Optional[float] = None
    extreme_weather_index:           Optional[float] = None
    seasonal_index:                  Optional[float] = None
    historical_flood_count:          Optional[float] = None
    flood_occurrence_current_event:  Optional[str]   = None
    water_presence_flag:             Optional[str]   = None
    # Terrain / land
    terrain_roughness_index:         Optional[float] = None
    ndvi:                            Optional[float] = None
    ndwi:                            Optional[float] = None
    built_up_percent:                Optional[float] = None
    landcover:                       Optional[str]   = None
    soil_type:                       Optional[str]   = None
    # Socio
    population_density_per_km2:      Optional[float] = None
    socioeconomic_status_index:      Optional[float] = None
    infrastructure_score:            Optional[float] = None
    nearest_hospital_km:             Optional[float] = None
    nearest_evac_km:                 Optional[float] = None
    road_quality:                    Optional[str]   = None
    electricity:                     Optional[str]   = None
    water_supply:                    Optional[str]   = None
    urban_rural:                     Optional[str]   = None
    is_good_to_live:                 Optional[str]   = None
    reason_not_good_to_live:         Optional[str]   = None


# ── Endpoints ────────────────────────────────────────────────────
@app.get("/health")
def health():
    try:
        load_model()
        return {"status": "ok", "model": "loaded"}
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/predict")
def predict_single(req: PredictRequest):
    row = {k: v for k, v in req.model_dump().items() if v is not None}
    try:
        result = predict(row)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
    _log("single", json.dumps(row), result["flood_risk_score"], result["risk_label"])
    return result


@app.post("/predict/batch")
async def predict_batch_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")
    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    try:
        result_df = predict_batch(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    # log summary
    for _, r in result_df[["flood_risk_score", "risk_label"]].iterrows():
        _log("batch", file.filename, r["flood_risk_score"], r["risk_label"])

    return result_df[["flood_risk_score", "risk_label"]].to_dict(orient="records")


@app.get("/metrics")
def metrics():
    con = sqlite3.connect(DB_PATH)
    rows = con.execute("""
        SELECT
            COUNT(*)               AS total_predictions,
            ROUND(AVG(score), 4)   AS avg_score,
            ROUND(MIN(score), 4)   AS min_score,
            ROUND(MAX(score), 4)   AS max_score,
            SUM(CASE WHEN label='Low'       THEN 1 ELSE 0 END) AS low_count,
            SUM(CASE WHEN label='Moderate'  THEN 1 ELSE 0 END) AS moderate_count,
            SUM(CASE WHEN label='High'      THEN 1 ELSE 0 END) AS high_count,
            SUM(CASE WHEN label='Very High' THEN 1 ELSE 0 END) AS very_high_count
        FROM predictions
    """).fetchone()
    con.close()
    keys = ["total_predictions","avg_score","min_score","max_score",
            "low_count","moderate_count","high_count","very_high_count"]
    return dict(zip(keys, rows))


@app.get("/logs")
def logs(limit: int = 100):
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        "SELECT id, ts, mode, score, label FROM predictions ORDER BY id DESC LIMIT ?",
        (limit,)
    ).fetchall()
    con.close()
    return [{"id": r[0], "timestamp": r[1], "mode": r[2], "score": r[3], "label": r[4]}
            for r in rows]


# ── Location enrichment endpoint ─────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))
from location_enricher import enrich_location as _enrich

class EnrichRequest(BaseModel):
    place_name: Optional[str] = None
    latitude:   Optional[float] = None
    longitude:  Optional[float] = None

@app.post("/enrich",
          summary="Auto-fill feature values from a location name or coordinates",
          description=(
              "Given a place name and/or lat/lon, fetches live elevation, "
              "rainfall, weather, and terrain data from free public APIs "
              "(Open-Meteo, Nominatim/OSM) and returns pre-filled feature values "
              "ready for /predict."
          ))
def enrich(req: EnrichRequest):
    if not req.place_name and req.latitude is None:
        raise HTTPException(status_code=400, detail="Provide place_name or latitude+longitude.")
    result = _enrich(place_name=req.place_name, lat=req.latitude, lon=req.longitude)
    return result
