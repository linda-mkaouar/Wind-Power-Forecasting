# ============================================================
# pipeline.py  —  Wind Turbine Power Forecasting Backend
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# SHARED PREPROCESSING HELPER
# ─────────────────────────────────────────────────────────────
def _engineer_features(df, turbulence_bins=None):
    """
    Apply all feature engineering steps to a raw dataframe (with datetime index).
    Returns the feature-engineered dataframe with lag/rolling columns added and NaNs dropped.

    turbulence_bins: pre-fitted bin edges from the training set (from pd.qcut retbins=True).
                     When provided, pd.cut is used with those edges so test data does not
                     influence the category boundaries (prevents data leakage).
                     When None, pd.qcut is used on the supplied data directly.
    """
    df = df.copy()
    df["hour"]  = df.index.hour
    df["day"]   = df.index.day
    df["month"] = df.index.month

    df["windspeed_100m_cubed"] = df["windspeed_100m"] ** 3
    df["turbulence_intensity"] = np.where(
        df["windspeed_10m"] == 0, 0,
        (df["windgusts_10m"] - df["windspeed_10m"]) / df["windspeed_10m"],
    )
    if turbulence_bins is None:
        df["turbulence_category"] = pd.qcut(
            df["turbulence_intensity"], q=3, labels=["Low", "Moderate", "High"]
        )
    else:
        # Use pre-fitted train-only bin edges — no leakage from the test set
        df["turbulence_category"] = pd.cut(
            df["turbulence_intensity"],
            bins=turbulence_bins,
            labels=["Low", "Moderate", "High"],
        )
    df["temp_K"]      = (df["temperature_2m"] - 32) * 5.0 / 9.0 + 273.15
    df["air_density"] = 101325 / (287.05 * df["temp_K"])

    def get_season(m):
        return "Winter" if m in [12,1,2] else "Spring" if m in [3,4,5] else "Summer" if m in [6,7,8] else "Autumn"
    df["season"] = df["month"].apply(get_season)

    df["wind_dir_sin"] = np.sin(np.radians(df["winddirection_100m"]))
    df["wind_dir_cos"] = np.cos(np.radians(df["winddirection_100m"]))
    df["Power_log"]    = np.log1p(df["Power"])

    df_enc = pd.get_dummies(df, columns=["season", "turbulence_category"], drop_first=True)
    for lag in [1, 2, 3, 6, 12, 24]:
        df_enc[f"Power_lag{lag}"]        = df_enc["Power"].shift(lag)
        df_enc[f"windspeed100_lag{lag}"] = df_enc["windspeed_100m"].shift(lag)
    df_enc["Power_roll3"]        = df_enc["Power"].rolling(3).mean()
    df_enc["windspeed100_roll3"] = df_enc["windspeed_100m"].rolling(3).mean()
    df_enc.dropna(inplace=True)
    return df_enc


# ─────────────────────────────────────────────────────────────
# PIPELINE — runs when a new CSV is uploaded
# ─────────────────────────────────────────────────────────────
def run_pipeline(df_input):
    """
    Full preprocessing + model training pipeline.

    Returns a status dict (log list).
    """
    import warnings, joblib, numpy as np, pandas as pd
    from scipy.stats import skew
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

    warnings.filterwarnings("ignore")
    os.makedirs("models", exist_ok=True)
    log = []

    # ── 1. Parse & clean ────────────────────────────────────────
    df = df_input.copy()
    time_col = None
    for c in df.columns:
        if c.lower() in ("time", "datetime", "date", "timestamp"):
            time_col = c
            break
    if time_col:
        df[time_col] = pd.to_datetime(df[time_col])
        df.set_index(time_col, inplace=True)
    df.sort_index(inplace=True)
    df_raw = df.copy()
    log.append(f"✅ Loaded {len(df):,} rows  |  {df.index.min().date()} → {df.index.max().date()}")

    # ── 1b. Missing Values ──────────────────────────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    missing_before = df[numeric_cols].isnull().sum()
    total_missing = int(missing_before.sum())
    missing_pct = (missing_before / len(df) * 100).round(2)
    missing_report = pd.DataFrame({
        "Missing Count": missing_before,
        "Missing %":     missing_pct,
    })
    missing_report = missing_report[missing_report["Missing Count"] > 0]

    if total_missing > 0:
        df[numeric_cols] = df[numeric_cols].ffill().bfill()
        still_missing = int(df[numeric_cols].isnull().sum().sum())
        if still_missing > 0:
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        log.append(f"✅ Missing values: {total_missing:,} found across {len(missing_report)} columns — forward/back-filled")
    else:
        log.append("✅ Missing values: none detected")

    # ── 1c. Duplicates ──────────────────────────────────────────
    n_dupes = df.index.duplicated().sum()
    if n_dupes > 0:
        df = df[~df.index.duplicated(keep="first")]
        log.append(f"✅ Duplicates: {n_dupes:,} duplicate timestamps removed")
    else:
        log.append("✅ Duplicates: none detected")

    # ── 1d. Outlier Detection & Capping (IQR method) ────────────
    outlier_cols = [c for c in numeric_cols if c != "Power"]
    outlier_report = {}
    for col in outlier_cols:
        if col not in df.columns:
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 3.0 * IQR
        upper = Q3 + 3.0 * IQR
        mask = (df[col] < lower) | (df[col] > upper)
        n_out = int(mask.sum())
        if n_out > 0:
            outlier_report[col] = {
                "count": n_out,
                "pct":   round(n_out / len(df) * 100, 3),
                "lower": round(lower, 4),
                "upper": round(upper, 4),
            }
            df[col] = df[col].clip(lower=lower, upper=upper)
    total_outliers = sum(v["count"] for v in outlier_report.values())
    if total_outliers > 0:
        log.append(f"✅ Outliers: {total_outliers:,} values capped (IQR×3) across {len(outlier_report)} columns")
    else:
        log.append("✅ Outliers: none detected (IQR×3 threshold)")

    # Save quality report for the Data Quality page
    quality_report = {
        "original_rows":   len(df) + n_dupes,
        "final_rows":      len(df),
        "missing_report":  missing_report,
        "n_dupes":         n_dupes,
        "outlier_report":  outlier_report,
    }
    joblib.dump(quality_report, "models/quality_report.pkl")

    # Save cleaned df_raw (post missing/dupe/outlier handling)
    df_raw = df.copy()
    joblib.dump(df_raw, "models/df_raw.pkl")

    # ── 2. Feature engineering ──────────────────────────────────
    rough_split = int(0.8 * len(df))
    train_ti = np.where(
        df["windspeed_10m"].iloc[:rough_split] == 0, 0,
        (df["windgusts_10m"].iloc[:rough_split] - df["windspeed_10m"].iloc[:rough_split])
        / df["windspeed_10m"].iloc[:rough_split],
    )
    _, turbulence_bins = pd.qcut(train_ti, q=3, retbins=True)
    turbulence_bins[0]  = -np.inf
    turbulence_bins[-1] =  np.inf
    joblib.dump(turbulence_bins, "models/turbulence_bins.pkl")

    df_encoded = _engineer_features(df, turbulence_bins=turbulence_bins)
    log.append(f"✅ Feature engineering done — shape {df_encoded.shape}")

    # ── 6. Train/test split ─────────────────────────────────────
    df_encoded.sort_index(inplace=True)
    split_idx = int(0.8 * len(df_encoded))
    train = df_encoded.iloc[:split_idx].copy()
    test  = df_encoded.iloc[split_idx:].copy()
    log.append(f"✅ Train: {len(train):,} rows  |  Test: {len(test):,} rows")

    # ── 5. Drop highly correlated features (computed on train only) ──
    corr_matrix = train.corr(numeric_only=True)
    target_corr = corr_matrix["Power"].abs().drop("Power")
    to_drop = set()
    feature_list = [c for c in corr_matrix.columns if c != "Power"]
    for i, col_a in enumerate(feature_list):
        for col_b in feature_list[i+1:]:
            val = corr_matrix.loc[col_a, col_b]
            if pd.notna(val) and abs(val) > 0.85:
                if target_corr.get(col_a, 0) >= target_corr.get(col_b, 0):
                    to_drop.add(col_b)
                else:
                    to_drop.add(col_a)
    train = train.drop(columns=list(to_drop))
    test  = test.drop(columns=list(to_drop))
    log.append(f"✅ Dropped {len(to_drop)} correlated features (train-only correlation)")

    # ── 7. Features & targets ────────────────────────────────────
    exclude_cols = ["Power", "Power_log", "month", "day", "hour", "turbulence_intensity"]
    feature_cols = [c for c in train.columns if c not in exclude_cols]
    X_train     = train[feature_cols]
    y_train_raw = train["Power"]
    y_train_log = train["Power_log"]
    X_test      = test[feature_cols]
    y_test_raw  = test["Power"]
    joblib.dump(feature_cols, "models/feature_cols.pkl")

    # ── 8. Scaling ───────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    joblib.dump(scaler, "models/scaler.pkl")

    # ── 9. Train models ──────────────────────────────────────────
    def evaluate(y_true, y_pred):
        mae  = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2   = r2_score(y_true, y_pred)
        return dict(MAE=mae, RMSE=rmse, R2=r2)

    results, predictions = {}, {}

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train_log)
    y_pred_lr = np.expm1(lr.predict(X_test_scaled))
    results["Linear Regression"] = evaluate(y_test_raw, y_pred_lr)
    predictions["Linear Regression"] = dict(y_pred=y_pred_lr, y_actual=y_test_raw.values, ts=test.index)
    joblib.dump(lr, "models/lr.pkl")
    log.append("✅ Linear Regression trained")

    # Lasso
    lasso = LassoCV(cv=5, random_state=42, max_iter=10000, n_jobs=-1)
    lasso.fit(X_train_scaled, y_train_log)
    y_pred_lasso = np.expm1(lasso.predict(X_test_scaled))
    results["Lasso"] = evaluate(y_test_raw, y_pred_lasso)
    predictions["Lasso"] = dict(y_pred=y_pred_lasso, y_actual=y_test_raw.values, ts=test.index)
    joblib.dump(lasso, "models/lasso.pkl")
    log.append("✅ Lasso trained")

    # Ridge
    ridge = RidgeCV(cv=5)
    ridge.fit(X_train_scaled, y_train_log)
    y_pred_ridge = np.expm1(ridge.predict(X_test_scaled))
    results["Ridge"] = evaluate(y_test_raw, y_pred_ridge)
    predictions["Ridge"] = dict(y_pred=y_pred_ridge, y_actual=y_test_raw.values, ts=test.index)
    joblib.dump(ridge, "models/ridge.pkl")
    log.append("✅ Ridge trained")

    # Random Forest
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train_raw)
    y_pred_rf = rf.predict(X_test)
    results["Random Forest"] = evaluate(y_test_raw, y_pred_rf)
    predictions["Random Forest"] = dict(y_pred=y_pred_rf, y_actual=y_test_raw.values, ts=test.index)
    joblib.dump(rf, "models/rf.pkl")
    fi_rf = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    log.append("✅ Random Forest trained")

    # XGBoost
    try:
        from xgboost import XGBRegressor
        xgb_default = XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42, verbosity=0)
        xgb_default.fit(X_train, y_train_raw)
        y_pred_xgb = xgb_default.predict(X_test)
        results["XGBoost"] = evaluate(y_test_raw, y_pred_xgb)
        predictions["XGBoost"] = dict(y_pred=y_pred_xgb, y_actual=y_test_raw.values, ts=test.index)
        tscv = TimeSeriesSplit(n_splits=3)
        param_dist_xgb = {
            "n_estimators": [100, 200, 300], "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [3, 5, 7], "subsample": [0.7, 0.8, 1.0],
            "colsample_bytree": [0.7, 0.8, 1.0],
        }
        rs_xgb = RandomizedSearchCV(
            XGBRegressor(random_state=42, verbosity=0), param_dist_xgb,
            n_iter=10, cv=tscv, scoring="neg_mean_squared_error",
            random_state=42, n_jobs=-1,
        )
        rs_xgb.fit(X_train, y_train_raw)
        best_xgb = rs_xgb.best_estimator_
        y_pred_xgb_tuned = best_xgb.predict(X_test)
        results["XGBoost (Tuned)"] = evaluate(y_test_raw, y_pred_xgb_tuned)
        predictions["XGBoost (Tuned)"] = dict(y_pred=y_pred_xgb_tuned, y_actual=y_test_raw.values, ts=test.index)
        joblib.dump(best_xgb, "models/xgb_tuned.pkl")
        fi_xgb = pd.Series(best_xgb.feature_importances_, index=feature_cols).sort_values(ascending=False)
        log.append("✅ XGBoost trained")
    except ImportError:
        fi_xgb = fi_rf.copy()
        log.append("⚠️  XGBoost not installed — skipped")

    # LightGBM
    try:
        import lightgbm as lgb
        lgbm_default = lgb.LGBMRegressor(n_estimators=200, random_state=42, n_jobs=-1, verbosity=-1)
        lgbm_default.fit(X_train, y_train_raw)
        y_pred_lgbm = lgbm_default.predict(X_test)
        results["LightGBM"] = evaluate(y_test_raw, y_pred_lgbm)
        predictions["LightGBM"] = dict(y_pred=y_pred_lgbm, y_actual=y_test_raw.values, ts=test.index)
        param_dist_lgb = {
            "n_estimators": [100, 200, 300], "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, 7, -1], "num_leaves": [15, 31, 63],
            "subsample": [0.7, 0.8, 1.0], "colsample_bytree": [0.7, 0.8, 1.0],
        }
        rs_lgb = RandomizedSearchCV(
            lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbosity=-1),
            param_dist_lgb,
            n_iter=10, cv=tscv, scoring="neg_mean_squared_error",
            random_state=42, n_jobs=-1,
        )
        rs_lgb.fit(X_train, y_train_raw)
        best_lgbm = rs_lgb.best_estimator_
        y_pred_lgbm_tuned = best_lgbm.predict(X_test)
        results["LightGBM (Tuned)"] = evaluate(y_test_raw, y_pred_lgbm_tuned)
        predictions["LightGBM (Tuned)"] = dict(y_pred=y_pred_lgbm_tuned, y_actual=y_test_raw.values, ts=test.index)
        joblib.dump(best_lgbm, "models/lgbm_tuned.pkl")
        fi_lgbm = pd.Series(best_lgbm.feature_importances_, index=feature_cols).sort_values(ascending=False)
        log.append("✅ LightGBM trained")
    except ImportError:
        fi_lgbm = fi_rf.copy()
        log.append("⚠️  LightGBM not installed — skipped")

    # CatBoost
    try:
        from catboost import CatBoostRegressor
        cat_default = CatBoostRegressor(iterations=300, learning_rate=0.1, depth=6,
                                         random_seed=42, verbose=0, task_type="CPU")
        cat_default.fit(X_train, y_train_raw)
        y_pred_cat = cat_default.predict(X_test)
        results["CatBoost"] = evaluate(y_test_raw, y_pred_cat)
        predictions["CatBoost"] = dict(y_pred=y_pred_cat, y_actual=y_test_raw.values, ts=test.index)

        param_dist_cat = {
            "iterations": [200, 300, 500], "learning_rate": [0.01, 0.05, 0.1],
            "depth": [4, 6, 8], "l2_leaf_reg": [1, 3, 5],
        }
        rs_cat = RandomizedSearchCV(
            CatBoostRegressor(random_seed=42, verbose=0, task_type="CPU"),
            param_dist_cat,
            n_iter=10, cv=tscv, scoring="neg_mean_squared_error",
            random_state=42, n_jobs=-1,
        )
        rs_cat.fit(X_train, y_train_raw)
        best_cat = rs_cat.best_estimator_
        y_pred_cat_tuned = best_cat.predict(X_test)
        results["CatBoost (Tuned)"] = evaluate(y_test_raw, y_pred_cat_tuned)
        predictions["CatBoost (Tuned)"] = dict(y_pred=y_pred_cat_tuned, y_actual=y_test_raw.values, ts=test.index)
        joblib.dump(best_cat, "models/cat_tuned.pkl")
        fi_cat = pd.Series(best_cat.get_feature_importance(), index=feature_cols).sort_values(ascending=False)
        log.append("✅ CatBoost trained")
    except ImportError:
        fi_cat = fi_rf.copy()
        log.append("⚠️  CatBoost not installed — skipped")

    # ── 10. Save artifacts ───────────────────────────────────────
    comparison_df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "Model"})
    comparison_df = comparison_df.sort_values("RMSE").reset_index(drop=True)
    best_model_name = comparison_df.iloc[0]["Model"]

    model_map = {
        "Linear Regression": lr, "Lasso": lasso, "Ridge": ridge,
        "Random Forest": rf,
    }
    if "XGBoost (Tuned)" in results:
        model_map["XGBoost"] = xgb_default
        model_map["XGBoost (Tuned)"] = best_xgb
    if "LightGBM (Tuned)" in results:
        model_map["LightGBM"] = lgbm_default
        model_map["LightGBM (Tuned)"] = best_lgbm
    if "CatBoost (Tuned)" in results:
        model_map["CatBoost"] = cat_default
        model_map["CatBoost (Tuned)"] = best_cat

    best_model_obj = model_map.get(best_model_name, rf)
    joblib.dump({"model": best_model_obj, "name": best_model_name}, "models/best_model.pkl")
    joblib.dump(comparison_df, "models/comparison_df.pkl")
    joblib.dump(predictions,   "models/predictions.pkl")
    joblib.dump({"RF": fi_rf, "XGBoost": fi_xgb, "LightGBM": fi_lgbm, "CatBoost": fi_cat},
                "models/feature_importances.pkl")

    last_rows = df_encoded[feature_cols + ["Power"]].tail(48).copy()
    joblib.dump(last_rows, "models/last_rows.pkl")

    raw_weather_cols = [
        "windspeed_10m", "windspeed_100m", "windgusts_10m",
        "temperature_2m", "relativehumidity_2m", "winddirection_100m",
        "dewpoint_2m",
    ]
    feature_ranges = {
        col: {"min": float(df_raw[col].min()), "max": float(df_raw[col].max()), "mean": float(df_raw[col].mean())}
        for col in raw_weather_cols if col in df_raw.columns
    }
    joblib.dump(feature_ranges, "models/feature_ranges.pkl")

    meta = {
        "best_model_name": best_model_name,
        "train_start": str(train.index[0].date()),
        "train_end":   str(train.index[-1].date()),
        "test_start":  str(test.index[0].date()),
        "test_end":    str(test.index[-1].date()),
        "n_features":  len(feature_cols),
        "n_train":     len(train),
        "n_test":      len(test),
        "rated_power": 1.0,
        "cut_in":      3.0,
        "rated_speed": 12.0,
        "cut_out":     25.0,
    }
    joblib.dump(meta, "models/meta.pkl")

    log.append(f"🏆 Best model: {best_model_name}")
    log.append("✅ All artifacts saved to models/")
    return log


# ─────────────────────────────────────────────────────────────
# LOAD ARTIFACTS
# ─────────────────────────────────────────────────────────────
def load_all():
    base = "models"
    if not os.path.exists(f"{base}/df_raw.pkl"):
        return None
    return {
        "df_raw":              joblib.load(f"{base}/df_raw.pkl"),
        "comparison_df":       joblib.load(f"{base}/comparison_df.pkl"),
        "predictions":         joblib.load(f"{base}/predictions.pkl"),
        "feature_importances": joblib.load(f"{base}/feature_importances.pkl"),
        "feature_cols":        joblib.load(f"{base}/feature_cols.pkl"),
        "best_model":          joblib.load(f"{base}/best_model.pkl"),
        "scaler":              joblib.load(f"{base}/scaler.pkl"),
        "last_rows":           joblib.load(f"{base}/last_rows.pkl"),
        "feature_ranges":      joblib.load(f"{base}/feature_ranges.pkl"),
        "meta":                joblib.load(f"{base}/meta.pkl"),
        "quality_report":      joblib.load(f"{base}/quality_report.pkl") if os.path.exists(f"{base}/quality_report.pkl") else None,
    }


def load_ensemble_models():
    """Load all individual model files once and cache them for the session."""
    model_files = {
        "Linear Regression": "models/lr.pkl",
        "Lasso":             "models/lasso.pkl",
        "Ridge":             "models/ridge.pkl",
        "Random Forest":     "models/rf.pkl",
        "XGBoost (Tuned)":   "models/xgb_tuned.pkl",
        "LightGBM (Tuned)":  "models/lgbm_tuned.pkl",
        "CatBoost (Tuned)":  "models/cat_tuned.pkl",
    }
    loaded = {}
    for name, path in model_files.items():
        if os.path.exists(path):
            try:
                loaded[name] = joblib.load(path)
            except Exception:
                pass
    return loaded
