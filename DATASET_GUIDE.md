# Dataset checklist
- Use CSV; put files in `data/raw/`.
- One row per completed lap is the recommended training unit.
- Target: lap duration in **seconds**, column `lap_time_seconds` (preferred) or a supported alias.
- Include `track`, `compound`, `tyre_age`, `fuel_kg`, `air_temp_c`, `track_temp_c`, `rainfall`, `traffic`, `driver_level`, and `setup_downforce` when available.
- Keep units consistent; encode rainfall as 0/1. Traffic, driver_level, setup_downforce should be scaled to 0–1.
- Do not train on rows with pit/in-lap/out-lap times unless you intentionally model them.
- Never claim the synthetic fallback is trained on actual F1 data.
- Record source URL, retrieval date, licence, session/year coverage, and data cleaning decisions in your report.
