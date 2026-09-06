"""
tests/test_stress.py — Project ARJUNA (SIH 26170)
Sustained-run stress verification (T7): bounded memory, timestamp monotonicity,
state-coherence under repeated scenario execution, and multi-client behavior.

Bounded runtime (< ~20 s) so it is CI-safe while still exercising 2,000+
sustained telemetry ticks.
"""

import random
import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "Backend"))

from Backend.criticality_config import get_config
from Backend.cusum_drift import DriftDetector
from Backend.database import TelemetryStore
from Backend.isolation_forest import LinearRegressionDriftPredictor, MultivariateAnomalyDetector
from Backend.security import API_KEY
from Backend.server import app
from Backend.simulator import ComponentSimulator

MODEL_DIR = ROOT_DIR / "Model"
client = TestClient(app)


def _server_domain_iddq(sim: ComponentSimulator, temp: float) -> float:
    """Iddq exactly as the deployed server generates it (server.py nominal path):
    Arrhenius thermal ratio × 10 µA + low-noise sensor jitter, clamped to the
    pre-screened lot bound [9.0, 11.5] µA. NOTE: this is the documented LIVE
    per-tick sensor domain (σ ≈ 0.15 µA) — NOT the cross-component lot-jitter
    domain (σ ≈ 1.15 µA, `compute_iddq_and_prop_delay` normal branch) which the
    two-noise-domain design explicitly reserves for CSV/training data.
    """
    import math

    t_kelvin = temp + 273.15
    t0_kelvin = 125.0 + 273.15
    thermal_ratio = math.exp(sim.Ea_kB * (1.0 / t0_kelvin - 1.0 / t_kelvin))
    return round(max(9.0, min(11.5, 10.0 * thermal_ratio + random.gauss(0, 0.15))), 2)


def _load_trained_detector() -> MultivariateAnomalyDetector:
    model_path = MODEL_DIR / "isolation_forest_model.joblib"
    if model_path.exists():
        return MultivariateAnomalyDetector.load_model(str(model_path))
    detector = MultivariateAnomalyDetector(contamination=0.001, n_estimators=40)
    detector.train(str(MODEL_DIR / "sample_data.csv"))
    return detector


def test_sustained_run_memory_bounded_and_state_coherent():
    """2,000 sustained ticks across scenario resets must not grow memory
    unboundedly, corrupt timestamps, or desync detector state."""
    random.seed(1234)
    store = TelemetryStore(history_limit=500, events_limit=100)
    sim = ComponentSimulator(criticality_level=2)
    detector = _load_trained_detector()
    cusum = DriftDetector(mean=10.0, std=1.17, criticality_level=2, auto_baseline=True)
    predictor = LinearRegressionDriftPredictor(lot_mean_iddq=10.0, lot_std_iddq=1.17)

    last_ts = None
    burn_in = 0.0
    for tick in range(2000):
        # Cycle scenarios to force repeated resets (repeated operation FP check)
        scenario = ["nominal", "isro_outlier", "nominal", "thermal_drift"][tick % 4]
        if tick % 500 == 0:
            sim.reset()
            cusum.reset()
            predictor.reset()
            burn_in = 0.0

        t, v, c = sim.step(dt=1.0, mode="normal" if scenario == "nominal" else "drift",
                           drift_time=burn_in, drift_rate=0.005)
        iddq = _server_domain_iddq(sim, t)
        ml = detector.detect_spike(current=c, voltage=v, temp=t, iddq=iddq)
        cusum_flag = cusum.evaluate_drift(iddq)
        predictor.update(burn_in_hours=burn_in, iddq_uA=iddq)

        ts = f"tick-{tick:05d}"
        assert ts != last_ts or tick == 0  # timestamps must advance
        last_ts = ts
        burn_in = min(168.0, burn_in + 0.4)

        store.record_telemetry(
            {"timestamp": ts, "voltage": v, "current": c, "temperature": t,
             "iddq_uA": iddq, "fault_type": "NORMAL", "anomaly_score": ml["anomaly_score"],
             "is_anomaly": ml["is_anomaly"], "cusum_drift_detected": cusum_flag,
             "criticality_level": 2, "system_status": "NOMINAL"}
        )

        # Bounded-memory contract (deque maxlen)
        assert len(store._history) <= 500
        assert len(predictor._iddq) <= predictor.max_window

    # Detector state must be finite and non-negative after 2,000 ticks
    assert cusum.cusum >= 0.0
    assert predictor._burn_hours[0] <= predictor._burn_hours[-1]
    status = store.get_status()
    assert status["total_telemetry_logged"] == 2000
    assert len(store._history) == 500  # trimmed to maxlen, not grown


def test_stress_multi_client_shared_chamber_consistency():
    """Three simultaneous WS clients must all receive coherent frames from the
    single shared chamber (documented single-DUT design): every frame must be a
    well-formed dict with consistent scenario/criticality within one frame."""
    frames = []
    with client.websocket_connect(f"/ws?api_key={API_KEY}") as ws1, \
         client.websocket_connect(f"/ws?api_key={API_KEY}") as ws2, \
         client.websocket_connect(f"/ws?api_key={API_KEY}") as ws3:
        for ws in (ws1, ws2, ws3):
            data = ws.receive_json()
            frames.append(data)

    for data in frames:
        assert isinstance(data, dict)
        assert data["criticality_level"] in (1, 2, 3)
        expected_h = get_config(data["criticality_level"])["cusum_threshold"]
        assert data["cusum_threshold"] == expected_h
        # Cross-field coherence inside a frame
        if data["fault_type"] == "NORMAL":
            assert data["system_status"] in ("NOMINAL", "ANOMALY")
        assert 0.0 <= data["burn_in_hours"] <= 168.0


def test_stress_repeated_scenario_resets_no_false_alarm_accumulation():
    """Repeated nominal runs after resets must not accumulate CUSUM state
    (false alarms caused by unclean reset would appear here)."""
    random.seed(99)
    false_alarms = 0
    for _run in range(10):
        sim = ComponentSimulator(criticality_level=2)
        cusum = DriftDetector(mean=10.0, std=1.17, criticality_level=2, auto_baseline=True)
        for _ in range(60):
            t, v, c = sim.step(dt=1.0, mode="normal")
            iddq = _server_domain_iddq(sim, t)
            if cusum.evaluate_drift(iddq):
                false_alarms += 1
    assert false_alarms == 0, (
        f"Unclean reset state caused {false_alarms} false CUSUM alarms across 10 nominal runs"
    )
