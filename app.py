from flask import Flask, render_template, request, jsonify, send_file
from pathlib import Path
import pandas as pd
import numpy as np
import joblib, json, io, threading, socket, struct, time
from datetime import datetime
from ml import FEATURES, train_model, predict_lap_time, model_status
from strategy import compare_strategies
from telemetry import TelemetryReceiver

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
SESSION_DIR = DATA_DIR / "sessions"
MODEL_PATH = BASE / "models" / "lap_time_model.joblib"
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 40 * 1024 * 1024
receiver = TelemetryReceiver()
latest_telemetry = {"connected": False, "message": "Listener stopped", "samples": [], "latest": None}

@app.get("/")
def index():
    return render_template("index.html")

@app.get("/api/status")
def status():
    return jsonify({"ok": True, "model": model_status(MODEL_PATH), "telemetry": receiver.status(),
                    "sessions": sorted([p.name for p in SESSION_DIR.glob("*.csv")])})

@app.post("/api/predict")
def predict():
    payload = request.get_json(force=True)
    try:
        result = predict_lap_time(payload, MODEL_PATH)
        return jsonify(result)
    except (ValueError, KeyError) as exc:
        return jsonify({"error": str(exc)}), 400

@app.post("/api/strategy")
def strategy():
    payload = request.get_json(force=True)
    try:
        return jsonify(compare_strategies(payload, MODEL_PATH))
    except (ValueError, KeyError) as exc:
        return jsonify({"error": str(exc)}), 400

@app.post("/api/train")
def train():
    try:
        result = train_model(DATA_DIR, MODEL_PATH)
        return jsonify(result)
    except (ValueError, FileNotFoundError) as exc:
        return jsonify({"error": str(exc)}), 400

@app.post("/api/sessions/upload")
def upload_session():
    f = request.files.get("file")
    if not f or not f.filename.lower().endswith(".csv"):
        return jsonify({"error": "Upload a CSV file."}), 400
    name = Path(f.filename).name
    target = SESSION_DIR / name
    target.parent.mkdir(parents=True, exist_ok=True)
    f.save(target)
    try:
        df = pd.read_csv(target)
        summary = summarize_session(df)
    except Exception as exc:
        target.unlink(missing_ok=True)
        return jsonify({"error": f"Could not parse CSV: {exc}"}), 400
    return jsonify({"filename": name, "summary": summary})

@app.get("/api/sessions/<path:name>")
def get_session(name):
    safe = Path(name).name
    target = SESSION_DIR / safe
    if not target.exists() or target.suffix.lower() != ".csv":
        return jsonify({"error": "Session not found."}), 404
    try:
        df = pd.read_csv(target)
        return jsonify({"filename": safe, "columns": list(df.columns),
                        "rows": df.head(1000).replace({np.nan: None}).to_dict(orient="records"),
                        "summary": summarize_session(df)})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

@app.get("/api/sessions/<path:name>/export")
def export_session(name):
    target = SESSION_DIR / Path(name).name
    if not target.exists() or target.suffix.lower() != ".csv":
        return jsonify({"error": "Session not found."}), 404
    return send_file(target, as_attachment=True, download_name=target.name)

@app.post("/api/telemetry/start")
def telemetry_start():
    payload = request.get_json(silent=True) or {}
    port = int(payload.get("port", 20777))
    try:
        receiver.start(port)
        return jsonify(receiver.status())
    except OSError as exc:
        return jsonify({"error": str(exc)}), 400

@app.post("/api/telemetry/stop")
def telemetry_stop():
    receiver.stop()
    return jsonify(receiver.status())

@app.get("/api/telemetry/latest")
def telemetry_latest():
    return jsonify(receiver.snapshot())

@app.post("/api/demo/generate")
def generate_demo():
    # Explicitly synthetic demo session, useful before real datasets are downloaded.
    rng = np.random.default_rng(24)
    n = 120
    speed = np.clip(rng.normal(235, 28, n), 80, 330)
    throttle = np.clip(rng.normal(.76, .18, n), 0, 1)
    brake = np.clip(rng.beta(1.4, 7, n), 0, 1)
    gear = np.clip(np.round(speed / 38), 2, 8).astype(int)
    rpm = np.clip(3500 + speed * 23 + rng.normal(0, 500, n), 3500, 12500).astype(int)
    df = pd.DataFrame({"timestamp": np.arange(n)*.1, "speed_kph": speed.round(2),
                       "throttle": throttle.round(3), "brake": brake.round(3),
                       "steering": rng.uniform(-1,1,n).round(3), "gear": gear, "rpm": rpm})
    target = SESSION_DIR / "synthetic_demo_telemetry.csv"
    df.to_csv(target, index=False)
    return jsonify({"filename": target.name, "summary": summarize_session(df), "synthetic": True})

def summarize_session(df):
    numeric = df.select_dtypes(include=[np.number])
    result = {"rows": int(len(df)), "columns": list(df.columns)}
    for col in ["speed_kph", "Speed", "speed", "throttle", "Throttle", "brake", "Brake", "lap_time", "LapTime"]:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(vals):
                result[col] = {"min": round(float(vals.min()), 3), "mean": round(float(vals.mean()), 3),
                               "max": round(float(vals.max()), 3)}
    result["preview"] = numeric.head(20).replace({np.nan: None}).to_dict(orient="records")
    return result

if __name__ == "__main__":
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    print("F1 Predictive Race Engineer → http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)
