"""Probe Finam Trade API symbols (dev only)."""
from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

API = "https://api.finam.ru"


def _token() -> str:
    secret = os.environ.get("FINAM_API_SECRET", "")
    if not secret:
        raise SystemExit("Set FINAM_API_SECRET")
    req = urllib.request.Request(
        f"{API}/v1/sessions",
        data=json.dumps({"secret": secret}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return json.loads(urllib.request.urlopen(req, timeout=30).read())["token"]


def _get(path: str, token: str) -> object:
    url = API + path
    req = urllib.request.Request(url, headers={"Authorization": token, "Accept": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


def main() -> None:
    token = _token()
    for q in ("NASDAQ", "SP500", "S&P", "IXIC", "SPX", "GAZP", "IMOEX", "DAX", "NDX"):
        for path in (
            f"/v1/assets?query={urllib.parse.quote(q)}",
            f"/v1/assets?q={urllib.parse.quote(q)}",
            f"/v1/instruments?query={urllib.parse.quote(q)}",
        ):
            try:
                data = _get(path, token)
            except Exception:
                continue
            print("\n==", path, "==")
            print(json.dumps(data, ensure_ascii=False)[:800])
            break

    sym = "GAZP@MISX"
    enc = urllib.parse.quote(sym, safe="")
    path = (
        f"/v1/instruments/{enc}/bars?timeframe=TIME_FRAME_M5"
        "&interval.start_time=2025-06-01T00:00:00Z&interval.end_time=2025-06-03T00:00:00Z"
    )
    try:
        data = _get(path, token)
        bars = data.get("bars", []) if isinstance(data, dict) else []
        print(f"\nGAZP bars: {len(bars)}")
    except Exception as exc:
        print(f"\nGAZP bars failed: {exc}")


if __name__ == "__main__":
    main()
