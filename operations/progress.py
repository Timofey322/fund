"""Dependency-free progress reporting for long pipeline loops.

Designed for captured (non-TTY) logs: prints throttled, newline-terminated
status lines (no carriage returns) so terminal capture files stay readable.
Throttling is by both wall-clock interval and percentage step to avoid spam.

Usage:
    for x in track(items, label="walk-forward"):
        ...

    rep = ProgressReporter(total=n, label="impulse grid")
    for combo in grid:
        rep.update()
        ...
    rep.close()

Disable globally via env ALGO_PROGRESS=0.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Iterable, Iterator, TypeVar

T = TypeVar("T")


def _enabled() -> bool:
    return os.environ.get("ALGO_PROGRESS", "1") not in ("0", "false", "False", "")


def _fmt_secs(s: float) -> str:
    s = int(max(0.0, s))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


class ProgressReporter:
    """Throttled counter-style progress printer."""

    def __init__(
        self,
        total: int | None,
        label: str = "",
        *,
        stream=None,
        indent: str = "    ",
        min_interval: float = 5.0,
        min_step_pct: float = 5.0,
        enabled: bool | None = None,
    ) -> None:
        self.total = int(total) if total else None
        self.label = label
        self.stream = stream if stream is not None else sys.stdout
        self.indent = indent
        self.min_interval = float(min_interval)
        self.min_step_pct = float(min_step_pct)
        self.enabled = _enabled() if enabled is None else bool(enabled)
        self.count = 0
        self._start = time.monotonic()
        self._last_t = self._start
        self._last_pct = -min_step_pct  # force first print after interval

    def update(self, n: int = 1) -> None:
        self.count += int(n)
        if not self.enabled:
            return
        now = time.monotonic()
        if now - self._last_t < self.min_interval:
            return
        if self.total:
            pct = 100.0 * self.count / self.total
            if pct - self._last_pct < self.min_step_pct and self.count < self.total:
                return
            self._last_pct = pct
        self._last_t = now
        self._emit(now)

    def _emit(self, now: float) -> None:
        elapsed = now - self._start
        rate = self.count / elapsed if elapsed > 0 else 0.0
        if self.total:
            pct = 100.0 * self.count / self.total
            eta = (self.total - self.count) / rate if rate > 0 else 0.0
            msg = (
                f"{self.indent}[{self.label}] {self.count:,}/{self.total:,} "
                f"({pct:.1f}%) elapsed {_fmt_secs(elapsed)} eta {_fmt_secs(eta)} "
                f"{rate:.1f}/s"
            )
        else:
            msg = (
                f"{self.indent}[{self.label}] {self.count:,} "
                f"elapsed {_fmt_secs(elapsed)} {rate:.1f}/s"
            )
        print(msg, file=self.stream, flush=True)

    def close(self) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        elapsed = now - self._start
        rate = self.count / elapsed if elapsed > 0 else 0.0
        total = self.total if self.total else self.count
        print(
            f"{self.indent}[{self.label}] done {self.count:,}/{total:,} "
            f"in {_fmt_secs(elapsed)} ({rate:.1f}/s)",
            file=self.stream,
            flush=True,
        )


def track(
    iterable: Iterable[T],
    *,
    total: int | None = None,
    label: str = "",
    stream=None,
    indent: str = "    ",
    min_interval: float = 5.0,
    min_step_pct: float = 5.0,
    enabled: bool | None = None,
) -> Iterator[T]:
    """Wrap an iterable, emitting throttled progress; yields items unchanged."""
    if total is None:
        try:
            total = len(iterable)  # type: ignore[arg-type]
        except (TypeError, AttributeError):
            total = None
    rep = ProgressReporter(
        total, label, stream=stream, indent=indent,
        min_interval=min_interval, min_step_pct=min_step_pct, enabled=enabled,
    )
    try:
        for item in iterable:
            yield item
            rep.update()
    finally:
        rep.close()
