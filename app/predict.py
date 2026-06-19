"""
Prediction pipeline — loads the saved artifact bundle and runs inference
for both single-row (manual input) and batch (CSV upload) modes.
"""

import os
import numpy as np
import pandas as pd
import joblib

EPS       = 1e-6
MODEL_PATH = os.environ.get("MODEL_PATH", "model/flood_model.pkl")

_artifact = None   # module-level cache


def load_model():
    global _artifact
    if _artifact is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at '{MODEL_PATH}'. "
                "Run train.py first to generate the model artifact."
            )
        _artifact = joblib.load(MODEL_PATH)
    return _artifact


def _engineer_single(row: dict, art: dict) -> np.ndarray:
    """Convert a raw feature dict → 1-row feature matrix."""
    df = pd.DataFrame([row])
    return _engineer_batch(df, art)


def _engineer_batch(df: pd.DataFrame, art: dict) -> np.ndarray:
    df = df.copy()
    fit_params        = art["fit_params"]
    district_agg_table = art["district_agg_table"]
    train_medians     = art["train_medians"]
    feature_names     = art["feature_names"]
    scaler            = art["scaler"]

    # Impute district/lat/lon from training lookup if missing
    # (best effort — new districts fall back to global mean encoding)

    # ── core log transforms ──────────────────────────────────────
    if "inundation_area_sqm" in df.columns:
        df["inun_log"] = np.log1p(df["inundation_area_sqm"])
    else:
        df["inun_log"] = 0.0

    for col, src in [
        ("distance_to_river_m_log1p",  "distance_to_river_m"),
        ("rainfall_7d_mm_log1p",       "rainfall_7d_mm"),
        ("monthly_rainfall_mm_log1p",  "monthly_rainfall_mm"),
        ("nearest_hospital_km_log1p",  "nearest_hospital_km"),
        ("nearest_evac_km_log1p",      "nearest_evac_km"),
    ]:
        if col not in df.columns and src in df.columns:
            df[col] = np.log1p(df[src].clip(lower=0))

    for col in ["inun_log", "distance_to_river_m_log1p", "rainfall_7d_mm_log1p"]:
        if col in df.columns:
            df[f"{col}_sq"] = df[col] ** 2
            df[f"{col}_cb"] = df[col] ** 3

    # interactions
    def _safe(a, b=None, op="mul"):
        aa = df.get(a, pd.Series(0, index=df.index))
        if b is None: return aa
        bb = df.get(b, pd.Series(0, index=df.index))
        return aa * bb if op == "mul" else aa / (bb + EPS)

    df["inun_per_river"]      = _safe("inun_log", "distance_to_river_m_log1p", "div")
    df["inun_x_rain"]         = _safe("inun_log", "rainfall_7d_mm_log1p")
    df["inun_river_ew"]       = df["inun_per_river"] * df.get("extreme_weather_index", 0)
    df["rain_x_river"]        = _safe("rainfall_7d_mm_log1p", "distance_to_river_m_log1p")
    df["rain_x_ew"]           = _safe("rainfall_7d_mm_log1p", "extreme_weather_index")
    df["river_x_ew"]          = _safe("distance_to_river_m_log1p", "extreme_weather_index")
    df["inun_x_terrain"]      = _safe("inun_log", "terrain_roughness_index")
    df["river_x_terrain"]     = _safe("distance_to_river_m_log1p", "terrain_roughness_index")
    df["river_elev_ratio"]    = df.get("distance_to_river_m", pd.Series(0, index=df.index)) / (df.get("elevation_m", pd.Series(1, index=df.index)).clip(lower=EPS))
    df["rain_drain_ratio"]    = df.get("rainfall_7d_mm", pd.Series(0, index=df.index)) / (df.get("drainage_index", pd.Series(1, index=df.index)).clip(lower=EPS))
    df["rain_monthly_ratio"]  = df.get("rainfall_7d_mm", pd.Series(0, index=df.index)) / (df.get("monthly_rainfall_mm", pd.Series(1, index=df.index)).clip(lower=EPS))
    df["pop_built_interact"]  = _safe("population_density_per_km2", "built_up_percent")
    df["ndwi_ndvi_diff"]      = df.get("ndwi", pd.Series(0, index=df.index)) - df.get("ndvi", pd.Series(0, index=df.index))
    df["flood_rain"]          = _safe("historical_flood_count", "rainfall_7d_mm_log1p")
    df["risk_composite"]      = (df.get("extreme_weather_index", 0)  * 0.4
                                  + df.get("terrain_roughness_index", 0) * 0.3
                                  + df.get("seasonal_index", 0)          * 0.3)
    df["infra_gap"]           = df.get("nearest_hospital_km_log1p", 0) + df.get("nearest_evac_km_log1p", 0)
    df["rain_sq"]             = df.get("rainfall_7d_mm_log1p", pd.Series(0, index=df.index)) ** 2
    df["sustained_rain_ratio"] = df.get("monthly_rainfall_mm_log1p", pd.Series(0, index=df.index)) / (df.get("rainfall_7d_mm_log1p", pd.Series(EPS, index=df.index)) + EPS)
    df["inun_x_flood_cnt"]   = df.get("inun_log", 0) * df.get("historical_flood_count", pd.Series(0, index=df.index)).fillna(0)
    df["rain_x_ew_x_terrain"] = (df.get("rainfall_7d_mm_log1p", 0)
                                   * df.get("extreme_weather_index", 0)
                                   * df.get("terrain_roughness_index", 0))
    df["inun_x_rain_x_river"] = (df.get("inun_log", 0)
                                   * df.get("rainfall_7d_mm_log1p", 0)
                                   * df.get("distance_to_river_m_log1p", 0))
    df["river_proximity_log"] = np.log1p(1.0 / (df.get("distance_to_river_m", pd.Series(1, index=df.index)).clip(lower=1)))
    df["rain_anomaly"]        = df.get("rainfall_7d_mm", 0) - (df.get("monthly_rainfall_mm", pd.Series(0, index=df.index)) / 30.0).clip(lower=0)
    df["pop_exposure"]        = (df.get("population_density_per_km2", 0)
                                  * df.get("inun_log", 0)
                                  / (df.get("elevation_m", pd.Series(1, index=df.index)).clip(lower=EPS)))

    # missingness
    miss_cols = ["elevation_m", "distance_to_river_m", "drainage_index",
                 "ndvi", "ndwi", "electricity", "road_quality",
                 "infrastructure_score", "nearest_hospital_km", "latitude", "longitude"]
    existing  = [c for c in miss_cols if c in df.columns]
    df["total_missing_count"] = df[existing].isnull().sum(axis=1) if existing else 0
    for col in miss_cols:
        if col in df.columns:
            df[f"{col}_missing"] = df[col].isnull().astype(int)

    # ordinal / binary encodings
    maps = {
        "flood_occurrence_current_event": {"Yes": 1, "No": 0},
        "water_presence_flag":            {"Likely": 1, "Unlikely": 0},
        "is_good_to_live":                {"Yes": 1, "No": 0},
        "urban_rural":                    {"Urban": 1, "Rural": 0},
        "road_quality":                   {"No road access": 0, "Poor (unpaved)": 1, "Fair": 2, "Good (paved)": 3},
        "electricity":                    {"Off-grid (solar)": 0, "Mixed": 1, "Grid": 2},
    }
    for col, m in maps.items():
        if col in df.columns:
            df[col] = df[col].map(m).fillna(list(m.values())[len(m)//2])

    if "soil_type" in df.columns:
        df["soil_risk"] = df["soil_type"].map({"Peaty": 3, "Clay": 2, "Sandy": 1, "Loamy": 1, "Silty": 0}).fillna(0)
    if "water_supply" in df.columns:
        df["water_supply_risk"] = df["water_supply"].map(
            {"Surface water": 3, "Tube-well": 2, "Well": 1, "Rainwater harvesting": 1, "Municipal": 0}).fillna(0)

    # reason_not_good_to_live
    if "reason_not_good_to_live" in df.columns:
        reason = df["reason_not_good_to_live"].fillna("Other")
        df["reason_high_flood"] = reason.str.contains("High flood risk",     case=False).astype(int)
        df["reason_poor_infra"] = reason.str.contains("Poor infrastructure", case=False).astype(int)
        df["reason_no_road"]    = reason.str.contains("No road access",      case=False).astype(int)
        df["reason_flag_count"] = df[["reason_high_flood", "reason_poor_infra", "reason_no_road"]].sum(axis=1)
        df["reason_text_len"]   = df["reason_not_good_to_live"].fillna("").str.len()

    # flood occ × water presence
    if "flood_occurrence_current_event" in df.columns and "water_presence_flag" in df.columns:
        fo = df["flood_occurrence_current_event"].fillna(0)
        wp = df["water_presence_flag"].fillna(0)
        df["flood_occ_x_water_pres"] = fo * wp
        df["fo_x_extreme_wx"]        = fo * df.get("extreme_weather_index", 0)
        df["wp_x_inun"]              = wp * df.get("inun_log", 0)

    # district aggregates
    df = df.merge(district_agg_table, on="district", how="left")
    for col in ["rainfall_7d_mm_log1p", "distance_to_river_m_log1p",
                "extreme_weather_index", "terrain_roughness_index",
                "socioeconomic_status_index"]:
        mean_col = f"district_mean_{col}"
        if mean_col in df.columns and col in df.columns:
            df[f"dev_{col}"] = df[col] - df[mean_col]
    if "district_mean_inun_log" in df.columns and "inun_log" in df.columns:
        df["dev_inun_log"] = df["inun_log"] - df["district_mean_inun_log"]
    for col in ["inun_log", "rainfall_7d_mm_log1p", "distance_to_river_m_log1p",
                "extreme_weather_index", "terrain_roughness_index", "socioeconomic_status_index"]:
        mean_col = f"district_mean_{col}"; std_col = f"district_std_{col}"
        if mean_col in df.columns and std_col in df.columns and col in df.columns:
            df[f"zscore_{col}"] = (df[col] - df[mean_col]) / (df[std_col] + EPS)
    if "district_mean_extreme_weather_index" in df.columns:
        df["ew_norm_by_district"] = df.get("extreme_weather_index", 0) / (df["district_mean_extreme_weather_index"] + EPS)

    # quantile bins (use saved boundaries)
    for col in ["inun_log", "distance_to_river_m_log1p", "rainfall_7d_mm_log1p",
                "elevation_m_yeojohnson", "extreme_weather_index"]:
        key = f"{col}_qbin5_boundaries"
        if key in fit_params and col in df.columns:
            df[f"{col}_qbin5"] = np.digitize(df[col].fillna(df[col].median()), fit_params[key])

    # district rank (use saved mapping)
    for col in ["inun_log", "distance_to_river_m_log1p", "rainfall_7d_mm_log1p"]:
        key = f"{col}_dist_rank_mapping"
        if key in fit_params and col in df.columns and "district" in df.columns:
            dm = fit_params[key]; fb = (np.array([0.0]), np.array([0.5]))
            df[f"{col}_dist_rank"] = df.apply(
                lambda r: np.interp(r[col], dm.get(r["district"], fb)[0], dm.get(r["district"], fb)[1])
                if (not pd.isna(r.get(col)) and r.get("district") in dm) else 0.5, axis=1)

    # river_close_flag
    if "river_close_threshold" in fit_params and "distance_to_river_m" in df.columns:
        df["river_close_flag"] = (df["distance_to_river_m"] < fit_params["river_close_threshold"]).astype(int)

    # target encoding — use global mean for unseen values
    global_mean = art["global_mean"]
    tr_district  = art["tr_district"]
    tr_placename = art["tr_placename"]
    y            = art["y"]

    def _smooth_enc(grp_tr, grp_query, y_tr, smoothing=30):
        tmp   = pd.DataFrame({"grp": grp_tr, "t": y_tr})
        stats = tmp.groupby("grp")["t"].agg(["mean", "count"])
        stats["enc"] = ((stats["mean"] * stats["count"] + global_mean * smoothing)
                        / (stats["count"] + smoothing))
        enc_map = stats["enc"].to_dict()
        return pd.Series(grp_query).map(enc_map).fillna(global_mean).values

    def _smooth_enc_std(grp_tr, grp_query, y_tr):
        tmp   = pd.DataFrame({"grp": grp_tr, "t": y_tr})
        stats = tmp.groupby("grp")["t"].std().fillna(0)
        return pd.Series(grp_query).map(stats.to_dict()).fillna(0).values

    if "district" in df.columns:
        df["district_te"]      = _smooth_enc(tr_district,  df["district"].values,  y)
        df["district_te_std"]  = _smooth_enc_std(tr_district,  df["district"].values,  y)
    if "place_name" in df.columns:
        df["place_name_te"]    = _smooth_enc(tr_placename, df["place_name"].values, y, smoothing=50)
        df["place_name_te_std"]= _smooth_enc_std(tr_placename, df["place_name"].values, y)

    # OHE
    ohe_cols = [c for c in ["landcover", "soil_type", "water_supply"] if c in df.columns]
    if ohe_cols:
        df = pd.get_dummies(df, columns=ohe_cols, dummy_na=True)

    # drop unwanted
    drop_cols = ["record_id", "generation_date", "reason_not_good_to_live",
                 "is_synthetic", "flood_risk_score", "inundation_area_sqm",
                 "district", "place_name"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors="ignore")

    # align to training feature set
    df = df.reindex(columns=feature_names, fill_value=0)

    # impute & clean
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(train_medians.reindex(num_cols, fill_value=0))
    df = df.replace([np.inf, -np.inf], 0)

    return df.values.astype(np.float32)


def predict(row: dict) -> dict:
    """Single-row prediction. Returns score + risk label."""
    art = load_model()
    X   = _engineer_single(row, art)
    raw = _run_ensemble(X, art)
    score = float(np.clip(art["iso_final"].predict(raw), art["Y_MIN"], art["Y_MAX"])[0])
    return {"flood_risk_score": round(score, 4), "risk_label": _label(score, art)}


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Batch prediction from a DataFrame. Returns df with score + label columns."""
    art    = load_model()
    X      = _engineer_batch(df, art)
    raw    = _run_ensemble(X, art)
    scores = np.clip(art["iso_final"].predict(raw), art["Y_MIN"], art["Y_MAX"])
    out    = df.copy()
    out["flood_risk_score"] = np.round(scores, 4)
    out["risk_label"]       = [_label(s, art) for s in scores]
    return out


def _run_ensemble(X: np.ndarray, art: dict) -> np.ndarray:
    scaler = art["scaler"]
    Xs     = scaler.transform(X)
    preds  = []
    for key in ["lgb_model", "cat_model", "et_model"]:
        if key in art:
            preds.append(art[key].predict(X))
    if "mlp_model" in art:
        preds.append(art["mlp_model"].predict(Xs))
    if "xgb_model" in art:
        preds.append(art["xgb_model"].predict(X))

    arr = np.column_stack(preds)

    method  = art.get("blend_method", "nelder_mead")
    weights = art.get("blend_weights")
    meta_en = art.get("meta_en")

    if method == "elasticnet" and meta_en is not None:
        return meta_en.predict(arr)
    elif weights is not None:
        # match weight count to available models
        w = np.array(weights[:arr.shape[1]], dtype=float)
        w = np.clip(w, 0, None); w /= w.sum()
        return arr @ w
    else:
        return arr.mean(axis=1)


def _label(score: float, art: dict) -> str:
    y_min, y_max = art["Y_MIN"], art["Y_MAX"]
    span = y_max - y_min or 1
    pct  = (score - y_min) / span
    if pct < 0.25:  return "Low"
    if pct < 0.50:  return "Moderate"
    if pct < 0.75:  return "High"
    return "Very High"
