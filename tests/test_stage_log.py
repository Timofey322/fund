"""Tests for explicit stage logging."""

from __future__ import annotations

from common.stage_log import stage_log


def test_stage_log_fold(capsys):
    stage_log("calibrating edge", fold=2, detail="80k rows")
    out = capsys.readouterr().out
    assert "fold 2 >> calibrating edge" in out
    assert "80k rows" in out


def test_stage_log_global(capsys):
    stage_log("decile gate audit", detail="1M rows")
    out = capsys.readouterr().out
    assert ">> decile gate audit" in out
    assert "fold" not in out
