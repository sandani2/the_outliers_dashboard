import warnings
warnings.filterwarnings("ignore")

import os
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from scipy.optimize import minimize, differential_evolution
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler
import lightgbm as lgb

try:
    import catboost
    from catboost import CatBoostRegressor
except ImportError:
    os.system("pip install catboost")
    from catboost import CatBoostRegressor

try:
    import xgboost as xgb
    USE_XGB = True
except ImportError:
    print("xgboost not found — running LGB + CAT + ET only")
    USE_XGB = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    USE_OPTUNA = True
except ImportError:
    print("optuna not found — using hand-tuned params")
    USE_OPTUNA = False

SEED         = 42
N_FOLDS      = 10
TARGET       = "flood_risk_score"
ID_COL       = "record_id"
EPS          = 1e-6
N_OPTUNA     = 80
IMPORTANCE_K = 150
MODEL_DIR    = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

BEST_LGB = dict(
    n_estimators=3000, learning_rate=0.02, num_leaves=127,
    min_child_samples=30, subsample=0.80, subsample_freq=1,
    colsample_bytree=0.50, reg_alpha=0.3, reg_lambda=0.8,
    max_bin=255, random_state=SEED, n_jobs=-1, verbose=-1,
)
BEST_CAT = dict(
    iterations=2000, learning_rate=0.03, depth=8, l2_leaf_reg=2,
    bagging_temperature=0.5, random_strength=1.0,
    random_seed=SEED, loss_function="RMSE", verbose=0,
)
BEST_XGB = dict(
    n_estimators=3000, learning_rate=0.02, max_depth=7,
    min_child_weight=4, subsample=0.80, colsample_bytree=0.50,
    reg_alpha=0.3, reg_lambda=0.8, gamma=0.1,
    random_state=SEED, n_jobs=-1, tree_method="hist", verbosity=0,
    early_stopping_rounds=100,
)
BEST_ET = dict(
    n_estimators=1000, max_depth=20, min_samples_leaf=5,
    max_features=0.5, n_jobs=-1, random_state=SEED,
)

DIST_AGG_COLS = [
    "rainfall_7d_mm_log1p", "distance_to_river_m_log1p",
    "extreme_weather_index", "terrain_roughness_index",
    "socioeconomic_status_index",
]


def make_district_aggs(ref_df):
    agg = ref_df.copy()
    agg["inun_log"] = np.log1p(agg["inundation_area_sqm"])
    cols = DIST_AGG_COLS + ["inun_log"]
    stats = agg.groupby("district")[cols].agg(
        ["mean", "std", "median", "skew",
         lambda x: x.quantile(0.75) - x.quantile(0.25)]
    )
    stats.columns = [f"district_{s}_{c}" for c, s in stats.columns]
    counts = agg.groupby("district").size().rename("district_count")
    stats = stats.join(counts)
    return stats.reset_index()


def engineer(df, district_agg_table, fit_params=None):
    df = df.copy()
    if fit_params is None:
        fit_params = {}

    df["inun_log"] = np.log1p(df["inundation_area_sqm"])
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

    df["inun_per_river"]   = df["inun_log"] / (df["distance_to_river_m_log1p"] + EPS)
    df["inun_x_rain"]      = df["inun_log"] * df["rainfall_7d_mm_log1p"]
    df["inun_river_ew"]    = df["inun_per_river"] * df["extreme_weather_index"]
    df["rain_x_river"]     = df["rainfall_7d_mm_log1p"] * df["distance_to_river_m_log1p"]
    df["rain_x_ew"]        = df["rainfall_7d_mm_log1p"] * df["extreme_weather_index"]
    df["river_x_ew"]       = df["distance_to_river_m_log1p"] * df["extreme_weather_index"]
    df["inun_x_terrain"]   = df["inun_log"] * df["terrain_roughness_index"]
    df["river_x_terrain"]  = df["distance_to_river_m_log1p"] * df["terrain_roughness_index"]
    df["river_elev_ratio"] = df["distance_to_river_m"] / (df["elevation_m"].clip(lower=EPS))
    df["rain_drain_ratio"] = df["rainfall_7d_mm"] / (df["drainage_index"].clip(lower=EPS))
    df["rain_monthly_ratio"] = df["rainfall_7d_mm"] / (df["monthly_rainfall_mm"].clip(lower=EPS))
    df["pop_built_interact"] = df["population_density_per_km2"] * df["built_up_percent"]
    df["ndwi_ndvi_diff"]   = df["ndwi"] - df["ndvi"]
    df["flood_rain"]       = df["historical_flood_count"] * df["rainfall_7d_mm_log1p"]
    df["risk_composite"]   = (df["extreme_weather_index"]  * 0.4
                               + df["terrain_roughness_index"] * 0.3
                               + df["seasonal_index"]          * 0.3)
    df["infra_gap"]        = df["nearest_hospital_km_log1p"] + df["nearest_evac_km_log1p"]
    df["rain_sq"]          = df["rainfall_7d_mm_log1p"] ** 2
    df["sustained_rain_ratio"] = df["monthly_rainfall_mm_log1p"] / (df["rainfall_7d_mm_log1p"] + EPS)
    df["inun_x_flood_cnt"] = df["inun_log"] * df["historical_flood_count"].fillna(0)
    df["rain_x_ew_x_terrain"] = (df["rainfall_7d_mm_log1p"]
                                  * df["extreme_weather_index"]
                                  * df["terrain_roughness_index"])
    df["inun_x_rain_x_river"] = (df["inun_log"]
                                  * df["rainfall_7d_mm_log1p"]
                                  * df["distance_to_river_m_log1p"])
    df["river_proximity_log"] = np.log1p(1.0 / (df["distance_to_river_m"].clip(lower=1)))
    df["rain_anomaly"] = df["rainfall_7d_mm"] - (df["monthly_rainfall_mm"] / 30.0).clip(lower=0)
    df["pop_exposure"] = (df["population_density_per_km2"] * df["inun_log"]
                          / (df["elevation_m"].clip(lower=EPS)))

    if "built_up_percent_qmap" in df.columns:
        df["built_x_rain"] = df["built_up_percent_qmap"] * df["rainfall_7d_mm_log1p"]
    if "ndwi_qmap" in df.columns:
        df["ndwi_x_rain"]  = df["ndwi_qmap"] * df["rainfall_7d_mm_log1p"]
    if "elevation_m_yeojohnson" in df.columns and "drainage_index_yeojohnson" in df.columns:
        df["elev_x_drain"] = df["elevation_m_yeojohnson"] * df["drainage_index_yeojohnson"]
    if "infrastructure_score" in df.columns:
        df["socio_x_infra"] = df["socioeconomic_status_index"] * df["infrastructure_score"].fillna(0)

    bin_cols = ["inun_log", "distance_to_river_m_log1p", "rainfall_7d_mm_log1p",
                "elevation_m_yeojohnson", "extreme_weather_index"]
    for col in bin_cols:
        if col not in df.columns:
            continue
        key = f"{col}_qbin5_boundaries"
        if key not in fit_params:
            fit_params[key] = np.percentile(df[col].dropna(), np.linspace(0, 100, 6)[1:-1])
        df[f"{col}_qbin5"] = np.digitize(df[col].fillna(df[col].median()), fit_params[key])

    for col in ["inun_log", "distance_to_river_m_log1p", "rainfall_7d_mm_log1p"]:
        if col not in df.columns or "district" not in df.columns:
            continue
        key = f"{col}_dist_rank_mapping"
        if key not in fit_params:
            dist_mappings = {}
            for district in df["district"].unique():
                vals = df.loc[df["district"] == district, col].dropna().values
                if len(vals) > 1:
                    sv = np.sort(vals)
                    dist_mappings[district] = (sv, np.arange(1, len(sv)+1) / (len(sv)+1))
                else:
                    dist_mappings[district] = (np.array([0.0]), np.array([0.5]))
            fit_params[key] = dist_mappings
        dm = fit_params[key]
        fb = (np.array([0.0]), np.array([0.5]))
        df[f"{col}_dist_rank"] = df.apply(
            lambda r: np.interp(r[col], dm.get(r["district"], fb)[0], dm.get(r["district"], fb)[1])
            if (not pd.isna(r[col]) and r["district"] in dm) else 0.5, axis=1)

    miss_probe = ["elevation_m", "distance_to_river_m", "drainage_index",
                  "ndvi", "ndwi", "electricity", "road_quality",
                  "infrastructure_score", "nearest_hospital_km",
                  "latitude", "longitude", "landcover", "soil_type",
                  "water_supply", "urban_rural", "water_presence_flag"]
    existing = [c for c in miss_probe if c in df.columns]
    df["total_missing_count"] = df[existing].isnull().sum(axis=1) if existing else 0
    for col in ["elevation_m", "distance_to_river_m", "drainage_index",
                "ndvi", "ndwi", "electricity", "road_quality",
                "infrastructure_score", "nearest_hospital_km", "latitude", "longitude"]:
        if col in df.columns:
            df[f"{col}_missing"] = df[col].isnull().astype(int)

    if "reason_not_good_to_live" in df.columns:
        reason = df["reason_not_good_to_live"].fillna("Other")
        df["reason_high_flood"] = reason.str.contains("High flood risk",     case=False).astype(int)
        df["reason_poor_infra"] = reason.str.contains("Poor infrastructure", case=False).astype(int)
        df["reason_no_road"]    = reason.str.contains("No road access",      case=False).astype(int)
        df["reason_flag_count"] = df[["reason_high_flood", "reason_poor_infra", "reason_no_road"]].sum(axis=1)
        df["reason_text_len"]   = df["reason_not_good_to_live"].fillna("").str.len()

    if "flood_occurrence_current_event" in df.columns:
        df["flood_occurrence_current_event"] = df["flood_occurrence_current_event"].map({"Yes": 1, "No": 0}).fillna(0)
    if "water_presence_flag" in df.columns:
        df["water_presence_flag"] = df["water_presence_flag"].map({"Likely": 1, "Unlikely": 0}).fillna(0)
    if "is_good_to_live" in df.columns:
        df["is_good_to_live"] = df["is_good_to_live"].map({"Yes": 1, "No": 0}).fillna(0)
    if "urban_rural" in df.columns:
        df["urban_rural"] = df["urban_rural"].map({"Urban": 1, "Rural": 0}).fillna(0)
    if "road_quality" in df.columns:
        df["road_quality"] = df["road_quality"].map(
            {"No road access": 0, "Poor (unpaved)": 1, "Fair": 2, "Good (paved)": 3}).fillna(1)
    if "electricity" in df.columns:
        df["electricity"] = df["electricity"].map(
            {"Off-grid (solar)": 0, "Mixed": 1, "Grid": 2}).fillna(1)
    if "soil_type" in df.columns:
        df["soil_risk"] = df["soil_type"].map(
            {"Peaty": 3, "Clay": 2, "Sandy": 1, "Loamy": 1, "Silty": 0}).fillna(0)
    if "water_supply" in df.columns:
        df["water_supply_risk"] = df["water_supply"].map(
            {"Surface water": 3, "Tube-well": 2, "Well": 1,
             "Rainwater harvesting": 1, "Municipal": 0}).fillna(0)

    if "flood_occurrence_current_event" in df.columns and "water_presence_flag" in df.columns:
        fo = df["flood_occurrence_current_event"].fillna(0)
        wp = df["water_presence_flag"].fillna(0)
        df["flood_occ_x_water_pres"] = fo * wp
        df["fo_x_extreme_wx"]        = fo * df["extreme_weather_index"]
        df["wp_x_inun"]              = wp * df["inun_log"]

    df = df.merge(district_agg_table, on="district", how="left")
    for col in DIST_AGG_COLS:
        mean_col = f"district_mean_{col}"
        if mean_col in df.columns and col in df.columns:
            df[f"dev_{col}"] = df[col] - df[mean_col]
    if "district_mean_inun_log" in df.columns:
        df["dev_inun_log"] = df["inun_log"] - df["district_mean_inun_log"]
    for col in ["inun_log"] + DIST_AGG_COLS:
        mean_col = f"district_mean_{col}"
        std_col  = f"district_std_{col}"
        if mean_col in df.columns and std_col in df.columns and col in df.columns:
            df[f"zscore_{col}"] = (df[col] - df[mean_col]) / (df[std_col] + EPS)

    if "district_mean_extreme_weather_index" in df.columns:
        df["ew_norm_by_district"] = (df["extreme_weather_index"]
                                     / (df["district_mean_extreme_weather_index"] + EPS))

    if "river_close_threshold" not in fit_params:
        fit_params["river_close_threshold"] = df["distance_to_river_m"].quantile(0.25)
    df["river_close_flag"] = (df["distance_to_river_m"] < fit_params["river_close_threshold"]).astype(int)

    drop_cols = ["record_id", "generation_date", "reason_not_good_to_live",
                 "is_synthetic", "flood_risk_score", "inundation_area_sqm"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors="ignore")

    ohe_cols = [c for c in ["landcover", "soil_type", "water_supply"] if c in df.columns]
    if ohe_cols:
        df = pd.get_dummies(df, columns=ohe_cols, dummy_na=True)

    return df, fit_params


# ── MLflow experiment setup ─────────────────────────────────────
mlflow.set_experiment("flood_risk_model")

with mlflow.start_run(run_name="ensemble_v6") as run:
    print(f"MLflow run ID: {run.info.run_id}")

    # Log all hyperparameters
    mlflow.log_params({f"lgb_{k}": v for k, v in BEST_LGB.items()})
    mlflow.log_params({f"cat_{k}": v for k, v in BEST_CAT.items()})
    mlflow.log_params({"n_folds": N_FOLDS, "seed": SEED, "importance_k": IMPORTANCE_K})

    print("Loading data...")
    train = pd.read_csv("train.csv")
    test  = pd.read_csv("test.csv")
    y     = train[TARGET].copy().values
    Y_MIN = float(y.min())
    Y_MAX = float(y.max())
    print(f"  Target range: [{Y_MIN:.4f}, {Y_MAX:.4f}]")

    # Impute district/lat/lon
    place_to_district = train.dropna(subset=["district", "place_name"]).set_index("place_name")["district"].to_dict()
    place_lat = train.dropna(subset=["latitude", "place_name"]).groupby("place_name")["latitude"].mean().to_dict()
    place_lon = train.dropna(subset=["longitude", "place_name"]).groupby("place_name")["longitude"].mean().to_dict()
    for df in [train, test]:
        df["district"]  = df.apply(lambda r: place_to_district.get(r["place_name"], r["district"]) if pd.isna(r["district"]) else r["district"], axis=1)
        df["latitude"]  = df.apply(lambda r: place_lat.get(r["place_name"], r["latitude"]) if pd.isna(r["latitude"]) else r["latitude"], axis=1)
        df["longitude"] = df.apply(lambda r: place_lon.get(r["place_name"], r["longitude"]) if pd.isna(r["longitude"]) else r["longitude"], axis=1)

    district_agg_table = make_district_aggs(train)

    print("Engineering features...")
    X_tr_raw, fit_params = engineer(train, district_agg_table, fit_params=None)
    X_te_raw, _          = engineer(test,  district_agg_table, fit_params=fit_params)

    # OOF target encoding
    kf_te       = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    global_mean = y.mean()

    def smooth_encode(grp_tr, grp_va, y_tr, smoothing):
        tmp   = pd.DataFrame({"grp": grp_tr, "t": y_tr})
        stats = tmp.groupby("grp")["t"].agg(["mean", "count"])
        stats["enc"] = ((stats["mean"] * stats["count"] + global_mean * smoothing) / (stats["count"] + smoothing))
        return pd.Series(grp_va).map(stats["enc"]).fillna(global_mean).values

    def smooth_encode_std(grp_tr, grp_va, y_tr):
        tmp   = pd.DataFrame({"grp": grp_tr, "t": y_tr})
        stats = tmp.groupby("grp")["t"].std().rename("std_enc").fillna(0)
        return pd.Series(grp_va).map(stats).fillna(0).values

    district_te  = np.full(len(X_tr_raw), global_mean)
    placename_te = np.full(len(X_tr_raw), global_mean)
    SM_DISTRICT, SM_PLACE = 30, 50

    for tr_idx, val_idx in kf_te.split(X_tr_raw):
        district_te[val_idx]  = smooth_encode(X_tr_raw.iloc[tr_idx]["district"].values, X_tr_raw.iloc[val_idx]["district"].values, y[tr_idx], SM_DISTRICT)
        placename_te[val_idx] = smooth_encode(X_tr_raw.iloc[tr_idx]["place_name"].values, X_tr_raw.iloc[val_idx]["place_name"].values, y[tr_idx], SM_PLACE)

    tr_district  = X_tr_raw["district"].values
    tr_placename = X_tr_raw["place_name"].values
    te_district  = X_te_raw["district"].values
    te_placename = X_te_raw["place_name"].values

    dist_enc  = smooth_encode(tr_district, te_district, y, SM_DISTRICT)
    place_enc = smooth_encode(tr_placename, te_placename, y, SM_PLACE)
    tr_dist_std  = smooth_encode_std(tr_district, tr_district, y)
    tr_place_std = smooth_encode_std(tr_placename, tr_placename, y)
    te_dist_std  = smooth_encode_std(tr_district, te_district, y)
    te_place_std = smooth_encode_std(tr_placename, te_placename, y)

    X_tr_raw["district_te"]       = district_te
    X_tr_raw["place_name_te"]     = placename_te
    X_tr_raw["district_te_std"]   = tr_dist_std
    X_tr_raw["place_name_te_std"] = tr_place_std
    X_te_raw["district_te"]       = dist_enc
    X_te_raw["place_name_te"]     = place_enc
    X_te_raw["district_te_std"]   = te_dist_std
    X_te_raw["place_name_te_std"] = te_place_std

    X_tr_raw.drop(columns=["district", "place_name"], inplace=True, errors="ignore")
    X_te_raw.drop(columns=["district", "place_name"], inplace=True, errors="ignore")
    X_te_raw = X_te_raw.reindex(columns=X_tr_raw.columns, fill_value=0)

    num_cols      = X_tr_raw.select_dtypes(include=[np.number]).columns
    train_medians = X_tr_raw[num_cols].median()
    X_tr_raw[num_cols] = X_tr_raw[num_cols].fillna(train_medians)
    X_te_raw[num_cols] = X_te_raw[num_cols].fillna(train_medians)
    X_tr_raw = X_tr_raw.replace([np.inf, -np.inf], 0)
    X_te_raw = X_te_raw.replace([np.inf, -np.inf], 0)

    X      = X_tr_raw.values.astype(np.float32)
    X_test = X_te_raw.values.astype(np.float32)
    feature_names = list(X_tr_raw.columns)

    # Feature selection
    if IMPORTANCE_K is not None:
        selector = lgb.LGBMRegressor(**BEST_LGB)
        selector.fit(X, y)
        top_idx = np.sort(np.argsort(selector.feature_importances_)[::-1][:IMPORTANCE_K])
        X      = X[:, top_idx]
        X_test = X_test[:, top_idx]
        feature_names = [feature_names[i] for i in top_idx]
        mlflow.log_param("selected_features", len(feature_names))

    # CV Training
    y_bins_full = pd.qcut(y, q=10, labels=False, duplicates="drop")
    skf_main    = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    oof_lgb = np.zeros(len(X)); pred_lgb = np.zeros(len(X_test))
    oof_cat = np.zeros(len(X)); pred_cat = np.zeros(len(X_test))
    oof_xgb = np.zeros(len(X)); pred_xgb = np.zeros(len(X_test))
    oof_et  = np.zeros(len(X)); pred_et  = np.zeros(len(X_test))
    oof_mlp = np.zeros(len(X)); pred_mlp = np.zeros(len(X_test))

    scaler        = RobustScaler()
    X_scaled      = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)

    for fold, (tr_idx, val_idx) in enumerate(skf_main.split(X, y_bins_full)):
        print(f"\n── Fold {fold+1}/{N_FOLDS} ──")
        Xtr, Xval     = X[tr_idx], X[val_idx]
        ytr, yval     = y[tr_idx], y[val_idx]
        Xtr_s, Xval_s = X_scaled[tr_idx], X_scaled[val_idx]

        lgb_m = lgb.LGBMRegressor(**BEST_LGB)
        lgb_m.fit(Xtr, ytr, eval_set=[(Xval, yval)],
                  callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)])
        oof_lgb[val_idx] = lgb_m.predict(Xval)
        pred_lgb        += lgb_m.predict(X_test) / N_FOLDS

        cat_m = CatBoostRegressor(**BEST_CAT)
        cat_m.fit(Xtr, ytr, eval_set=(Xval, yval), early_stopping_rounds=80, verbose=False)
        oof_cat[val_idx] = cat_m.predict(Xval)
        pred_cat        += cat_m.predict(X_test) / N_FOLDS

        if USE_XGB:
            xgb_m = xgb.XGBRegressor(**BEST_XGB)
            xgb_m.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
            oof_xgb[val_idx] = xgb_m.predict(Xval)
            pred_xgb        += xgb_m.predict(X_test) / N_FOLDS

        et_m = ExtraTreesRegressor(**BEST_ET)
        et_m.fit(Xtr, ytr)
        oof_et[val_idx] = et_m.predict(Xval)
        pred_et        += et_m.predict(X_test) / N_FOLDS

        mlp_m = MLPRegressor(
            hidden_layer_sizes=(256, 128, 64), activation="relu", solver="adam",
            learning_rate_init=3e-4, max_iter=300, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=20, random_state=SEED + fold,
        )
        mlp_m.fit(Xtr_s, ytr)
        oof_mlp[val_idx] = mlp_m.predict(Xval_s)
        pred_mlp        += mlp_m.predict(X_test_scaled) / N_FOLDS

    # Blending
    labels    = ["LGB", "CAT", "ET", "MLP"] + (["XGB"] if USE_XGB else [])
    oof_list  = [oof_lgb, oof_cat, oof_et, oof_mlp] + ([oof_xgb] if USE_XGB else [])
    pred_list = [pred_lgb, pred_cat, pred_et, pred_mlp] + ([pred_xgb] if USE_XGB else [])
    oof_arr   = np.column_stack(oof_list)
    pred_arr  = np.column_stack(pred_list)

    def blend_rmse(w):
        w = np.clip(w, 0, None); s = w.sum()
        if s < EPS: return 1e9
        return mean_squared_error(y, oof_arr @ (w / s)) ** 0.5

    x0     = np.ones(len(oof_list)) / len(oof_list)
    res_nm = minimize(blend_rmse, x0, method="Nelder-Mead", options={"maxiter": 10000})
    best_w = np.clip(res_nm.x, 0, None); best_w /= best_w.sum()

    res_de = differential_evolution(blend_rmse, [(0, 1)] * len(oof_list), seed=SEED, maxiter=500, polish=True)
    w_de   = np.clip(res_de.x, 0, None); w_de /= w_de.sum()

    meta_en = ElasticNet(alpha=0.01, l1_ratio=0.5, fit_intercept=True, max_iter=10000, random_state=SEED)
    meta_en.fit(oof_arr, y)

    rmse_nm   = blend_rmse(best_w)
    rmse_de   = blend_rmse(w_de)
    rmse_enet = mean_squared_error(y, meta_en.predict(oof_arr)) ** 0.5

    best_rmse = min(rmse_nm, rmse_de, rmse_enet)
    if best_rmse == rmse_nm:
        blend_oof = oof_arr @ best_w; blend_pred = pred_arr @ best_w; blend_weights = best_w; blend_method = "nelder_mead"
    elif best_rmse == rmse_de:
        blend_oof = oof_arr @ w_de;   blend_pred = pred_arr @ w_de;   blend_weights = w_de;   blend_method = "diff_evolution"
    else:
        blend_oof = meta_en.predict(oof_arr); blend_pred = meta_en.predict(pred_arr); blend_weights = None; blend_method = "elasticnet"

    # Isotonic calibration
    iso_final = IsotonicRegression(out_of_bounds="clip")
    iso_final.fit(blend_oof, y)
    final_preds = np.clip(iso_final.predict(blend_pred), Y_MIN, Y_MAX)

    iso_oof = iso_final.predict(blend_oof)
    final_oof_rmse = mean_squared_error(y, iso_oof) ** 0.5

    # Log metrics to MLflow
    for name, oof in zip(labels, oof_list):
        mlflow.log_metric(f"oof_rmse_{name.lower()}", mean_squared_error(y, oof) ** 0.5)
    mlflow.log_metric("oof_rmse_blend",    mean_squared_error(y, blend_oof) ** 0.5)
    mlflow.log_metric("oof_rmse_isotonic", final_oof_rmse)
    mlflow.log_param("blend_method", blend_method)
    if blend_weights is not None:
        for name, w in zip(labels, blend_weights):
            mlflow.log_param(f"weight_{name}", round(float(w), 4))

    print(f"\n═══ FINAL OOF RMSE (+ Isotonic): {final_oof_rmse:.5f} ═══")

    # ── SAVE ALL ARTIFACTS ─────────────────────────────────────
    print("\nSaving model artifacts...")

    artifact = {
        "district_agg_table":  district_agg_table,
        "fit_params":          fit_params,
        "train_medians":       train_medians,
        "feature_names":       feature_names,
        "scaler":              scaler,
        "blend_weights":       blend_weights,
        "blend_method":        blend_method,
        "meta_en":             meta_en,
        "iso_final":           iso_final,
        "Y_MIN":               Y_MIN,
        "Y_MAX":               Y_MAX,
        "global_mean":         global_mean,
        "tr_district":         tr_district,
        "tr_placename":        tr_placename,
        "y":                   y,
        "labels":              labels,
        "USE_XGB":             USE_XGB,
        # Last-fold models for single-row inference
        "lgb_model":           lgb_m,
        "cat_model":           cat_m,
        "et_model":            et_m,
        "mlp_model":           mlp_m,
    }
    if USE_XGB:
        artifact["xgb_model"] = xgb_m

    joblib.dump(artifact, f"{MODEL_DIR}/flood_model.pkl", compress=3)
    mlflow.log_artifact(f"{MODEL_DIR}/flood_model.pkl")
    mlflow.log_param("model_path", f"{MODEL_DIR}/flood_model.pkl")

    # Save submission
    sub = pd.DataFrame({ID_COL: test[ID_COL], TARGET: final_preds})
    sub.to_csv("submission_v6.csv", index=False)
    mlflow.log_artifact("submission_v6.csv")

    print(f"Saved → {MODEL_DIR}/flood_model.pkl")
    print(f"Saved → submission_v6.csv")
    print(f"MLflow run: {run.info.run_id}")
