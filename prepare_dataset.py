"""Prepare the Jolpica/Ergast-style historical F1 archive for the local lap-time model.

Expected source CSVs in data/raw/: lap_times.csv, races.csv, circuits.csv, drivers.csv.
Writes data/f1_lap_training.csv. It uses milliseconds as the reliable lap-time target
and samples up to 150,000 valid laps to keep local training practical.
"""
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent
RAW = BASE / "data" / "raw"
OUT = BASE / "data" / "f1_lap_training.csv"
MAX_ROWS = 150_000


def main():
    required = ["lap_times.csv", "races.csv", "circuits.csv", "drivers.csv"]
    missing = [name for name in required if not (RAW / name).exists()]
    if missing:
        raise SystemExit(
            "Missing source files in data/raw/: " + ", ".join(missing)
            + "\nExtract archive.zip so the CSV files are directly inside data/raw/."
        )

    laps = pd.read_csv(RAW / "lap_times.csv")
    races = pd.read_csv(RAW / "races.csv", na_values=["\\N"])
    circuits = pd.read_csv(RAW / "circuits.csv", na_values=["\\N"])
    drivers = pd.read_csv(RAW / "drivers.csv", na_values=["\\N"])

    # milliseconds is the dependable numeric lap-time target in this archive.
    laps["lap_time_seconds"] = pd.to_numeric(laps["milliseconds"], errors="coerce") / 1000.0
    laps = laps.dropna(subset=["lap_time_seconds", "raceId", "driverId", "lap"])
    laps = laps[laps["lap_time_seconds"].between(30, 300)]

    race_cols = races[["raceId", "year", "round", "circuitId", "name"]].rename(
        columns={"name": "grand_prix"}
    )
    circuit_cols = circuits[["circuitId", "name"]].rename(columns={"name": "track"})
    driver_cols = drivers[["driverId", "forename", "surname"]].copy()
    driver_cols["driver_name"] = (
        driver_cols["forename"].fillna("").astype(str).str.strip()
        + " " + driver_cols["surname"].fillna("").astype(str).str.strip()
    ).str.strip()
    driver_cols = driver_cols[["driverId", "driver_name"]]

    data = laps.merge(race_cols, on="raceId", how="left")
    data = data.merge(circuit_cols, on="circuitId", how="left")
    data = data.merge(driver_cols, on="driverId", how="left")
    data["track"] = data["track"].fillna(data["grand_prix"]).fillna("Unknown")

    # Historical archive lacks per-lap tyre compound, fuel load and weather values.
    # Keep neutral placeholders explicit; do not pretend these are measured values.
    data["compound"] = "Unknown"
    data["tyre_age"] = 0
    data["fuel_kg"] = 50.0
    data["air_temp_c"] = 25.0
    data["track_temp_c"] = 32.0
    data["rainfall"] = 0
    data["traffic"] = 0.2
    data["driver_level"] = 0.7
    data["setup_downforce"] = 0.5

    keep = ["track", "compound", "tyre_age", "fuel_kg", "air_temp_c", "track_temp_c",
            "rainfall", "traffic", "driver_level", "setup_downforce", "lap_time_seconds",
            "lap", "position", "year", "round", "driver_name", "grand_prix"]
    data = data[keep].dropna(subset=["lap_time_seconds"])
    if len(data) > MAX_ROWS:
        data = data.sample(n=MAX_ROWS, random_state=42)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUT, index=False)
    print(f"Prepared {len(data):,} lap records: {OUT}")
    print("Target: lap_time_seconds (seconds).")
    print("Note: tyre compound, fuel, weather and setup are placeholders, not measured archive values.")
    print("Next: run the app, open Race Simulator, then click Train model from local CSVs.")


if __name__ == "__main__":
    main()
