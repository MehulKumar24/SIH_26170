"""
scripts/check_supabase_rls.py — Project ARJUNA (SIH 26170)
Opt-in live verification of Supabase RLS behavior (SECURITY_REMEDIATION.md §3).

Runs ONLY when SUPABASE_URL and SUPABASE_KEY are present in the environment
(or .env). Never hardcode credentials here.

Checks:
  1. INSERT into telemetry_logs with the configured key.
     - service_role / authenticated key  -> expected SUCCEED
     - anon / publishable key            -> expected REJECTED (401/403), proving
                                            RLS blocks anonymous writes.
  2. Anonymous-style SELECT on telemetry_logs -> expected SUCCEED (read is
     intentionally public per migrations/supabase_schema.sql).

Exit codes: 0 = behavior matches the documented RLS model; 1 = mismatch;
2 = missing configuration (script skipped).
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import httpx  # noqa: E402  (after dotenv so .env values are visible)


def main() -> int:
    url = os.getenv("SUPABASE_URL", "").strip()
    key = os.getenv("SUPABASE_KEY", "").strip()
    if not url or not key:
        print("SKIP: SUPABASE_URL / SUPABASE_KEY not configured (opt-in script).")
        return 2

    from Backend.database import normalize_supabase_url

    url = normalize_supabase_url(url)
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    insert_headers = {**headers, "Prefer": "return=minimal"}
    client = httpx.Client(timeout=8.0)

    ok = True

    # ---- Check 1: INSERT must follow RLS ------------------------------------
    payload = {
        "timestamp": "1970-01-01T00:00:00+00:00",  # sentinel row, easy to purge
        "voltage": 5.0,
        "current": 1.2,
        "temperature": 125.0,
        "iddq_ua": 10.0,
        "prop_delay": 4.5,
        "anomaly_score": 0.0,
        "isolation_anomaly": False,
        "drift_anomaly": False,
        "fault_type": "NORMAL",
        "criticality_level": 2,
        "system_status": "NOMINAL",
    }
    try:
        res = client.post(f"{url}/rest/v1/telemetry_logs", json=payload, headers=insert_headers)
    except Exception as exc:
        print(f"FAIL: insert request raised {exc}")
        return 1

    is_service_role = "service_role" in key or "sb_secret_" in key
    if res.status_code in (200, 201):
        print(f"PASS: INSERT succeeded (key class: {'service_role' if is_service_role else 'authenticated/publishable'}).")
        if not is_service_role:
            print("WARNING: a non-service-role key was able to INSERT — check RLS policies!")
            ok = False
    elif res.status_code in (401, 403):
        print(f"PASS: INSERT rejected with HTTP {res.status_code} — RLS is enforcing the write policy.")
        if is_service_role:
            print("FAIL: service_role key was rejected — service_role must bypass RLS.")
            ok = False
    else:
        print(f"FAIL: unexpected HTTP {res.status_code} on insert: {res.text[:200]}")
        ok = False

    # ---- Check 2: public SELECT (documented model) ---------------------------
    try:
        res = client.get(
            f"{url}/rest/v1/telemetry_logs?select=*&limit=1", headers=headers
        )
        if res.status_code == 200:
            print("PASS: public SELECT succeeded (read is intentionally public).")
        else:
            print(f"NOTE: SELECT returned HTTP {res.status_code} — verify this matches your intended read policy.")
    except Exception as exc:
        print(f"NOTE: SELECT request raised {exc}")
        ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
