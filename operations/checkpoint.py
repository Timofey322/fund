"""Pipeline checkpoint store (SQLite + JSON blobs per agent step)."""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import OUT_DIR

CHECKPOINT_ROOT = OUT_DIR / "checkpoints"
DEFAULT_DB = CHECKPOINT_ROOT / "pipeline.db"


@dataclass
class PipelineContext:
    """Shared state passed between role agents."""

    run_id: str
    tickers: list[str]
    artifacts: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def new(cls, tickers: list[str], run_id: str | None = None) -> PipelineContext:
        return cls(
            run_id=run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8],
            tickers=[t.upper() for t in tickers],
        )


class PipelineCheckpoint:
    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or DEFAULT_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS steps (
                    run_id TEXT NOT NULL,
                    step TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, step)
                )
                """
            )

    def save(self, run_id: str, step: str, payload: dict[str, Any], *, status: str = "done") -> None:
        now = datetime.now(timezone.utc).isoformat()
        blob = json.dumps(payload, default=str)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO steps (run_id, step, status, payload, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(run_id, step) DO UPDATE SET
                    status=excluded.status,
                    payload=excluded.payload,
                    updated_at=excluded.updated_at
                """,
                (run_id, step, status, blob, now),
            )

    def load(self, run_id: str, step: str) -> dict[str, Any] | None:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT payload FROM steps WHERE run_id=? AND step=?",
                (run_id, step),
            ).fetchone()
        if not row or not row[0]:
            return None
        return json.loads(row[0])

    def last_completed_step(self, run_id: str) -> str | None:
        order = ["data", "hmm", "fusion", "plot", "monte_carlo"]
        done = set()
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT step, status FROM steps WHERE run_id=?",
                (run_id,),
            ).fetchall()
        for step, status in rows:
            if status == "done":
                done.add(step)
        last = None
        for s in order:
            if s in done:
                last = s
        return last

    def list_runs(self, limit: int = 20) -> list[dict[str, str]]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT run_id, MAX(updated_at) AS updated
                FROM steps GROUP BY run_id
                ORDER BY updated DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [{"run_id": r[0], "updated_at": r[1]} for r in rows]
