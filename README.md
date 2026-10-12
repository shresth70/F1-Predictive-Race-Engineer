# F1 24 Predictive Race Engineer
**Predictive Analytics (CSE3141) — local university project**

A Flask + vanilla HTML/CSS/JavaScript dashboard for lap-time estimates, illustrative pit-strategy comparison, saved CSV session analysis, and optional F1 24 UDP telemetry. No paid API, cloud service, Docker, or GPU is required.

## Run on Windows
1. Install Python 3.10–3.12 and VS Code.
2. Extract the ZIP.
3. Double-click `run_windows.bat`, or run:
   ```powershell
   py -3 -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   python app.py
   ```
4. Open `http://127.0.0.1:5000`.

The first start installs dependencies. Internet is needed for pip installation, not for normal local app use after installation.

## Modes
- **Race Simulator:** uses a locally trained model when one exists. Before you provide a dataset, it uses a clearly labelled synthetic fallback so the interface works. Synthetic output is not an official F1 prediction and must not be presented as validated.
- **Live Telemetry:** starts a local UDP listener (default port 20777). In F1 24 PC settings, enable UDP telemetry, choose port 20777, telemetry format 2024, and send to `127.0.0.1` if game and app are on the same PC. Firewall settings may need to allow local Python UDP traffic. This receiver currently targets the 2024 Car Telemetry packet (packet ID 6); it is not a complete parser for every F1 packet.
- **Saved Session / Demo:** upload CSV, generate a synthetic demo telemetry CSV, inspect summary/preview, and export the uploaded session.

## Dataset setup — when you download data
Put raw CSV files in `data/raw/` (recommended). You can also put a training CSV directly in `data/`. Do not delete `data/sessions/`; it stores session uploads.

The training endpoint looks for a lap-level target column named one of:
- `lap_time_seconds` (preferred)
- `lap_time`
- `LapTime`
- `lapTime`
- `Lap_Time`

Recognized input aliases include `Track`/`Circuit` → `track`, `Compound`/`TyreCompound` → `compound`, `TyreLife` → `tyre_age`, `FuelLoad` → `fuel_kg`, `AirTemp` → `air_temp_c`, `TrackTemp` → `track_temp_c`, `Rainfall`, `Traffic`, `DriverLevel`, and `Downforce`. If the dataset lacks some predictors, defaults are filled in; this makes the prototype runnable but can reduce scientific validity. Prefer lap-level rows with a real lap-time target and relevant weather, tyre, fuel, circuit, and driver/session features.

### Suggested public data sources
- **FastF1**: https://docs.fastf1.dev/ — Python library to access timing, telemetry, weather, and session data. Review its API/data licensing and caching requirements. Data generally represents real F1 sessions, not F1 24 game physics.
- **Kaggle Formula 1 datasets**: https://www.kaggle.com/datasets — search for lap times, race results, tyre strategies, weather, and pit stops. Check the dataset's licence and column definitions before use.
- **Ergast/Jolpica historical data**: https://github.com/jolpica/jolpica-f1 — useful for race results and historical timing-related records, but not a drop-in source of every simulator feature.
- **F1 24 UDP telemetry**: best for your own game sessions. Record laps and telemetry locally, then engineer lap-level rows; the game UDP data is not the same as official real-world race data.

Avoid combining real-world F1 timing with game telemetry without clearly separating domains. F1 game telemetry does not automatically provide labelled lap times in the exact tabular shape needed by the model.

## Training
1. Add one or more lap-level CSVs under `data/raw/`.
2. Ensure each row represents a lap and includes a target lap-time column in seconds.
3. Start the app, open **Race Simulator**, and click **Train model from local CSVs**.
4. Review MAE/RMSE/R² shown by the API. Save the model locally under `models/`.
5. Report train/test split and limitations in your university report. For stronger evaluation, split by race/session rather than randomly when the same session contributes many correlated rows.

Training uses a RandomForestRegressor with a held-out 20% random test split. This is a baseline, not evidence of causal effects. A robust final project should add group-based validation by session/season and compare against a simple baseline.

## API routes
- `GET /api/status`
- `POST /api/predict`
- `POST /api/strategy`
- `POST /api/train`
- `POST /api/sessions/upload` (multipart field `file`)
- `GET /api/sessions/<filename>`
- `GET /api/sessions/<filename>/export`
- `POST /api/telemetry/start` JSON `{ "port": 20777 }`
- `POST /api/telemetry/stop`
- `GET /api/telemetry/latest`
- `POST /api/demo/generate`

## Limitations
- Strategy comparison is a transparent heuristic: predicted stint pace + simplified tyre degradation + fixed pit loss. It does not model safety cars, traffic evolution, tyre rules, undercut/overcut, weather changes, or all compound constraints.
- UDP parsing is intentionally narrow and should be validated against actual F1 24 packets before being relied on. It only handles Car Telemetry packet ID 6 and uses the player car's leading telemetry fields.
- Demo session data is synthetic and labelled as such.
- This is a local educational project, not a real-time team race-engineering tool.

### Prepare the downloaded historical archive
1. Extract the downloaded `archive.zip`.
2. Copy its CSV files directly into this project's `data/raw/` folder (so `data/raw/lap_times.csv` exists).
3. In the project folder's VS Code terminal, run `python prepare_dataset.py`.
4. This creates `data/f1_lap_training.csv` with lap times converted from milliseconds to seconds and circuit names joined in. It samples up to 150,000 valid laps for local training speed.
5. Start the app with `python app.py`, open `http://127.0.0.1:5000`, go to Race Simulator, and click **Train model from local CSVs**.

Important: the archive does not contain per-lap tyre compound, fuel load, weather, or setup for these records. The preparation script uses explicit placeholders for those missing inputs. Treat this as a historical baseline, not a model that has learned those effects. Do not claim the placeholders are measured data.
