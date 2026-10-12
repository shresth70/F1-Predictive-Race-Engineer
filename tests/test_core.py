import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml import predict_lap_time
from strategy import compare_strategies

def test_fallback_prediction(tmp_path):
    out = predict_lap_time({"track":"Silverstone","compound":"Medium","tyre_age":5}, tmp_path/"missing.joblib")
    assert 30 < out["predicted_lap_seconds"] < 180
    assert "Synthetic" in out["model"]

def test_strategy_order_and_stints(tmp_path):
    out = compare_strategies({"race_laps":20}, tmp_path/"missing.joblib")
    assert len(out["strategies"]) == 4
    assert out["strategies"][0]["rank"] == 1
    assert all(s["total_seconds"] > 0 for s in out["strategies"])
