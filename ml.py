from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FEATURES = ["track", "compound", "tyre_age", "fuel_kg", "air_temp_c", "track_temp_c",
            "rainfall", "traffic", "driver_level", "setup_downforce"]
NUMERIC = ["tyre_age", "fuel_kg", "air_temp_c", "track_temp_c", "rainfall",
           "traffic", "driver_level", "setup_downforce"]
CATEGORICAL = ["track", "compound"]
TARGET_CANDIDATES = ["lap_time_seconds", "lap_time", "LapTime", "lapTime", "Lap_Time"]

def _synthetic_training_frame(n=2400, seed=42):
    """Fallback for a runnable demo only; never represents official race results."""
    rng = np.random.default_rng(seed)
    tracks = np.array(["Bahrain", "Monza", "Silverstone", "Spa", "Suzuka", "Monaco", "Austria"])
    base = {"Bahrain": 91, "Monza": 82, "Silverstone": 89, "Spa": 106, "Suzuka": 92, "Monaco": 72, "Austria": 66}
    track = rng.choice(tracks, n)
    compound = rng.choice(["Soft", "Medium", "Hard"], n, p=[.35,.4,.25])
    tyre_age = rng.integers(0, 32, n)
    fuel = rng.uniform(1, 110, n)
    air = rng.uniform(15, 35, n); track_temp = rng.uniform(20, 55, n)
    rain = rng.binomial(1, .08, n); traffic = rng.uniform(0, 1, n)
    driver = rng.uniform(0, 1, n); downforce = rng.uniform(0, 1, n)
    comp = pd.Series(compound).map({"Soft":-.45,"Medium":0,"Hard":.35}).to_numpy()
    lap = np.array([base[t] for t in track]) + comp + tyre_age*.055 + fuel*.028 + rain*rng.uniform(1.5,5,n) + traffic*1.2 + (1-driver)*.65 + np.abs(track_temp-32)*.012 - downforce*.18 + rng.normal(0,.32,n)
    return pd.DataFrame({"track":track,"compound":compound,"tyre_age":tyre_age,"fuel_kg":fuel,
        "air_temp_c":air,"track_temp_c":track_temp,"rainfall":rain,"traffic":traffic,
        "driver_level":driver,"setup_downforce":downforce,"lap_time_seconds":lap})

def _normalize_columns(df):
    aliases = {
      "LapTime":"lap_time_seconds", "lap_time":"lap_time_seconds", "lapTime":"lap_time_seconds",
      "Lap_Time":"lap_time_seconds", "Compound":"compound", "TyreCompound":"compound",
      "Track":"track", "Circuit":"track", "TyreLife":"tyre_age", "tyre_life":"tyre_age",
      "FuelLoad":"fuel_kg", "AirTemp":"air_temp_c", "TrackTemp":"track_temp_c",
      "Rainfall":"rainfall", "Traffic":"traffic", "DriverLevel":"driver_level",
      "Downforce":"setup_downforce"
    }
    return df.rename(columns={k:v for k,v in aliases.items() if k in df.columns and v not in df.columns})

def _prepare(df, fitting=False, encoders=None):
    df = _normalize_columns(df.copy())
    for col in FEATURES:
        if col not in df.columns:
            if col in CATEGORICAL: df[col] = "Unknown"
            else: df[col] = {"tyre_age":0,"fuel_kg":50,"air_temp_c":25,"track_temp_c":32,
                             "rainfall":0,"traffic":0.2,"driver_level":0.7,"setup_downforce":0.5}[col]
    df["track"] = df["track"].fillna("Unknown").astype(str)
    df["compound"] = df["compound"].fillna("Medium").astype(str)
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    if fitting:
        categories = {c: sorted(df[c].astype(str).unique().tolist()) for c in CATEGORICAL}
    else:
        categories = encoders
    out = df[NUMERIC].copy()
    for c in CATEGORICAL:
        known = categories[c]
        out[c] = df[c].astype(str).where(df[c].astype(str).isin(known), "Unknown")
        if "Unknown" not in known: known = known + ["Unknown"]
        for val in known:
            out[f"{c}={val}"] = (out[c] == val).astype(int)
        out = out.drop(columns=[c])
    return out, categories

def model_status(path):
    if not path.exists():
        return {"trained": False, "kind": "Synthetic fallback", "message": "No trained model yet; simulator uses a clearly labelled synthetic fallback."}
    try:
        m = joblib.load(path)
        return {"trained": True, "kind": m.get("kind", "Local model"), "metrics": m.get("metrics", {}),
                "trained_at": m.get("trained_at", "unknown")}
    except Exception:
        return {"trained": False, "kind": "Unavailable", "message": "Saved model could not be loaded."}

def train_model(data_dir, model_path):
    data_dir = Path(data_dir)
    files = list(data_dir.glob("*.csv")) + list((data_dir/"raw").glob("*.csv"))
    if not files:
        raise FileNotFoundError("No CSV datasets found in data/ or data/raw/. Add a lap-level dataset with a lap time column, then retry.")
    frames, ignored = [], []
    for f in files:
        try:
            df = pd.read_csv(f)
            df = _normalize_columns(df)
            if any(c in df.columns for c in TARGET_CANDIDATES):
                target = next(c for c in TARGET_CANDIDATES if c in df.columns)
                df["lap_time_seconds"] = pd.to_numeric(df[target], errors="coerce")
                frames.append(df)
            else:
                ignored.append(f.name)
        except Exception:
            ignored.append(f.name)
    if not frames:
        raise ValueError("No usable lap-time target column found. Expected lap_time_seconds, lap_time, LapTime or lapTime. See DATASET_GUIDE.md.")
    data = pd.concat(frames, ignore_index=True)
    data = data[np.isfinite(data["lap_time_seconds"]) & (data["lap_time_seconds"] > 30) & (data["lap_time_seconds"] < 300)]
    if len(data) < 30:
        raise ValueError(f"Only {len(data)} usable lap rows found; provide at least 30 for a basic train/test split.")
    X, enc = _prepare(data, fitting=True)
    y = data["lap_time_seconds"].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
    model = RandomForestRegressor(n_estimators=250, min_samples_leaf=2, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {"mae_seconds": round(float(mean_absolute_error(y_test,pred)),4),
               "rmse_seconds": round(float(np.sqrt(mean_squared_error(y_test,pred))),4),
               "r2": round(float(r2_score(y_test,pred)),4), "test_rows": int(len(y_test)), "training_rows": int(len(y_train))}
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model":model,"features":list(X.columns),"encoders":enc,"metrics":metrics,
                 "kind":"RandomForestRegressor (trained on local CSV)","trained_at":pd.Timestamp.now().isoformat()}, model_path)
    return {"success":True,"metrics":metrics,"files_used":[p.name for p in files],"ignored_files":ignored,
            "message":"Model trained and saved locally."}

def _fallback_predict(payload):
    frame = _synthetic_training_frame(2600, seed=12)
    X, enc = _prepare(frame, fitting=True)
    model = RandomForestRegressor(n_estimators=100, min_samples_leaf=3, random_state=7, n_jobs=-1).fit(X, frame["lap_time_seconds"])
    row, _ = _prepare(pd.DataFrame([payload]), fitting=False, encoders=enc)
    row = row.reindex(columns=X.columns, fill_value=0)
    return float(model.predict(row)[0]), "Synthetic demo model — not official or validated race data"

def predict_lap_time(payload, model_path):
    required = ["track","compound","tyre_age","fuel_kg","air_temp_c","track_temp_c","rainfall","traffic","driver_level","setup_downforce"]
    defaults = {"track":"Silverstone","compound":"Medium","tyre_age":5,"fuel_kg":45,"air_temp_c":25,
                "track_temp_c":32,"rainfall":0,"traffic":.2,"driver_level":.7,"setup_downforce":.5}
    values = {k:payload.get(k,defaults[k]) for k in required}
    for k in NUMERIC:
        try: values[k] = float(values[k])
        except (ValueError, TypeError): raise ValueError(f"{k} must be numeric.")
    if values["tyre_age"] < 0 or values["fuel_kg"] < 0: raise ValueError("Tyre age and fuel load cannot be negative.")
    if values["rainfall"] not in (0,1): raise ValueError("Rainfall must be 0 (dry) or 1 (wet).")
    if not 0 <= values["traffic"] <= 1 or not 0 <= values["driver_level"] <= 1 or not 0 <= values["setup_downforce"] <= 1:
        raise ValueError("Traffic, driver level and downforce must be between 0 and 1.")
    if model_path.exists():
        bundle = joblib.load(model_path)
        X, _ = _prepare(pd.DataFrame([values]), encoders=bundle["encoders"])
        X = X.reindex(columns=bundle["features"], fill_value=0)
        prediction = float(bundle["model"].predict(X)[0])
        kind = bundle.get("kind","Trained local model")
        metrics = bundle.get("metrics",{})
    else:
        prediction, kind = _fallback_predict(values); metrics = {}
    return {"predicted_lap_seconds":round(prediction,3),"predicted_lap_formatted":f"{int(prediction//60)}:{prediction%60:06.3f}",
            "model":kind,"metrics":metrics,"features":values,
            "note":"Estimate only. Accuracy depends on dataset quality and whether inputs match the training data."}
