"""CLI: download maximum Finam 5-minute history for key instruments."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_platform.finam_bars import DEFAULT_FINAM_UNIVERSE, download_finam


def main() -> int:
    p = argparse.ArgumentParser(description="Download Finam 5-minute OHLCV (4-month chunks).")
    p.add_argument(
        "--symbols",
        default=",".join(DEFAULT_FINAM_UNIVERSE),
        help="Comma-separated Finam aliases (GAZP,SBER,IMOEX,NASDAQ,SP500,DAX,GER40)",
    )
    p.add_argument("--years-back", type=int, default=10, help="History depth to request")
    p.add_argument("--force", action="store_true", help="Rebuild cache even if parquet exists")
    args = p.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    print(f"Finam 5Min download @ {datetime.now(timezone.utc).isoformat()}", flush=True)
    counts = download_finam(symbols, years_back=args.years_back, force=args.force)
    ok = sum(1 for n in counts.values() if n > 0)
    print("Done:", counts, flush=True)
    print(f"Summary: {ok}/{len(symbols)} symbols with data", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
