"""Explicit pipeline step logging (visible between long silent phases)."""

from __future__ import annotations


def stage_log(
    step: str,
    *,
    fold: int | str | None = None,
    detail: str | None = None,
    indent: int = 6,
) -> None:
    """Print a single-line stage marker, e.g. ``fold 0 >> calibrating edge``."""
    prefix = " " * indent
    if fold is not None:
        line = f"{prefix}fold {fold} >> {step}"
    else:
        line = f"{prefix}>> {step}"
    if detail:
        line = f"{line} — {detail}"
    print(line, flush=True)
