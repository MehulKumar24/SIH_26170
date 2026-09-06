"""
tests/test_supabase.py — Project ARJUNA (SIH 26170)
Automated tests for Supabase Schema Alignment, Payload Formatting, Filtering, and Outage Fallback.
"""

import logging

from Backend.database import TelemetryStore, telemetry_store


def test_telemetry_payload_schema_compliance():
    """Verify that formatted payload strictly matches migrations/supabase_schema.sql."""
    raw_frame = {
        "timestamp": "2026-03-30T00:00:01.000Z",
        "voltage": 4.982,
        "current": 1.205,
        "temperature": 125.3,
        "iddq_uA": 45.2,
        "prop_delay": 4.52,
        "anomaly_score": 0.892,
        "is_anomaly": True,
        "cusum_drift_detected": False,
        "fault_type": "ELECTRICAL_SPIKE",
        "criticality_level": 3,
        "system_status": "ANOMALY",
    }

    formatted = TelemetryStore._format_telemetry_payload(raw_frame)

    # Exact column names must match SQL schema
    expected_cols = {
        "timestamp",
        "voltage",
        "current",
        "temperature",
        "iddq_uA",
        "prop_delay",
        "anomaly_score",
        "isolation_anomaly",
        "drift_anomaly",
        "fault_type",
        "criticality_level",
        "system_status",
    }
    assert set(formatted.keys()) == expected_cols
    assert formatted["voltage"] == 4.982
    assert formatted["iddq_uA"] == 45.2
    assert formatted["criticality_level"] == 3
    assert formatted["isolation_anomaly"] is True
    assert formatted["drift_anomaly"] is False
    assert formatted["fault_type"] == "ELECTRICAL_SPIKE"


def test_telemetry_store_offline_in_memory_persistence():
    """Verify offline fallback records telemetry and events without database connection."""
    store = TelemetryStore(history_limit=100)
    store.client = None  # Explicit offline mode
    store.http_client = None  # Explicit offline mode

    test_frame_1 = {
        "timestamp": "2026-03-30T00:00:01Z",
        "voltage": 5.0,
        "current": 1.2,
        "temperature": 125.0,
        "iddq_uA": 10.0,
        "fault_type": "NORMAL",
    }
    test_frame_2 = {
        "timestamp": "2026-03-30T00:00:02Z",
        "voltage": 5.0,
        "current": 1.2,
        "temperature": 125.0,
        "iddq_uA": 45.0,
        "fault_type": "ELECTRICAL_SPIKE",
    }

    store.record_telemetry(test_frame_1)
    store.record_telemetry(test_frame_2)
    store.record_event("INJECTION", "HIGH", "Spike injected", criticality_level=2)

    # Recent retrieval without filter
    recent_all = store.recent(limit=10)
    assert len(recent_all) == 2
    assert recent_all[0]["iddq_uA"] == 45.0  # Most recent first

    # Recent retrieval with fault_type filter
    spikes_only = store.recent(limit=10, fault_type="ELECTRICAL_SPIKE")
    assert len(spikes_only) == 1
    assert spikes_only[0]["fault_type"] == "ELECTRICAL_SPIKE"

    nominal_only = store.recent(limit=10, fault_type="NORMAL")
    assert len(nominal_only) == 1
    assert nominal_only[0]["fault_type"] == "NORMAL"

    # Events retrieval
    events = store.recent_events(limit=10)
    assert len(events) == 1
    assert events[0]["event_type"] == "INJECTION"


def test_telemetry_store_status_reporting():
    """Verify telemetry store provides comprehensive persistence status."""
    status = telemetry_store.get_status()
    assert "supabase_enabled" in status
    assert "supabase_available" in status
    assert "in_memory_records" in status
    assert "total_telemetry_logged" in status


# =============================================================================
# S5: Persistence failures must be OBSERVABLE, not silently swallowed.
# A non-2xx PostgREST response (e.g. RLS 401/403 rejection) or a transport
# exception must surface via last_error and a throttled WARNING while the
# local in-memory buffer still absorbs the frame (non-blocking preserved).
# =============================================================================

class _FakeHttpResponse:
    """Minimal stand-in for an httpx.Response."""

    def __init__(self, status_code: int, text: str = ""):
        self.status_code = status_code
        self.text = text


class _FailingHttpClient:
    """Stands in for the PostgREST httpx client."""

    def __init__(self, status_code: int = 403):
        self.status_code = status_code
        self.calls = 0

    def post(self, url, json=None):
        self.calls += 1
        return _FakeHttpResponse(self.status_code, text="RLS violation")


def test_rls_rejection_surfaces_in_last_error(caplog):
    """A non-2xx (RLS-style) response must set last_error — never silent."""
    store = TelemetryStore(history_limit=10)
    store.url = "https://example.supabase.co"
    store.http_client = _FailingHttpClient(status_code=403)  # type: ignore[assignment]
    store.last_error = None

    with caplog.at_level(logging.WARNING, logger="project_arjuna.database"):
        store.record_telemetry(
            {"timestamp": "2026-03-30T00:00:01Z", "voltage": 5.0, "current": 1.2,
             "temperature": 125.0, "iddq_uA": 10.0, "fault_type": "NORMAL"}
        )

    assert store.last_error is not None
    assert "403" in store.last_error
    # Frame must still be absorbed by the in-memory buffer (non-blocking intact)
    assert len(store.recent(limit=10)) == 1


def test_persistence_failure_warning_is_throttled(caplog):
    """First failure warns; subsequent failures do not flood the log."""
    store = TelemetryStore(history_limit=100)
    store.url = "https://example.supabase.co"
    store.http_client = _FailingHttpClient(status_code=403)  # type: ignore[assignment]
    store.last_error = None

    with caplog.at_level(logging.WARNING, logger="project_arjuna.database"):
        for _ in range(60):
            store.record_telemetry(
                {"timestamp": "2026-03-30T00:00:01Z", "voltage": 5.0, "current": 1.2,
                 "temperature": 125.0, "iddq_uA": 10.0, "fault_type": "NORMAL"}
            )

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    # 1st failure + the 50th failure = exactly 2 warnings out of 60 failures
    assert len(warnings) == 2
    assert all("PERSISTENCE FALLBACK" in r.getMessage() for r in warnings)


def test_transport_exception_surfaces_in_last_error(caplog):
    """A raised transport exception must set last_error and warn, not vanish."""
    store = TelemetryStore(history_limit=10)
    store.url = "https://example.supabase.co"

    class _RaisingClient:
        def post(self, url, json=None):
            raise ConnectionError("network down")

    store.http_client = _RaisingClient()  # type: ignore[assignment]
    store.last_error = None

    with caplog.at_level(logging.WARNING, logger="project_arjuna.database"):
        store.record_telemetry(
            {"timestamp": "2026-03-30T00:00:01Z", "voltage": 5.0, "current": 1.2,
             "temperature": 125.0, "iddq_uA": 10.0, "fault_type": "NORMAL"}
        )

    assert store.last_error is not None
    assert "network down" in store.last_error
    assert any(
        "PERSISTENCE FALLBACK" in r.getMessage()
        for r in caplog.records if r.levelno == logging.WARNING
    )
    assert len(store.recent(limit=10)) == 1


def test_event_rejection_surfaces_in_last_error():
    """Non-2xx event insertion must set last_error (previously fully silent)."""
    store = TelemetryStore(events_limit=10)
    store.url = "https://example.supabase.co"
    store.http_client = _FailingHttpClient(status_code=401)  # type: ignore[assignment]
    store.last_error = None

    store.record_event("INJECTION", "HIGH", "Spike injected", criticality_level=2)

    assert store.last_error is not None
    assert "401" in store.last_error
    assert store.total_inserted_events == 1  # buffered locally regardless
