"""Tests for Optuna progress callback."""

from __future__ import annotations

from common.optuna_progress import optuna_progress_callback


class _Trial:
    def __init__(self, number: int, value: float | None, user_attrs: dict | None = None):
        self.number = number
        self.value = value
        self.user_attrs = user_attrs or {}


class _Study:
    def __init__(self, best_value: float, best_attrs: dict | None = None):
        self.best_value = best_value
        self.best_trial = _Trial(0, best_value, best_attrs or {})


def test_progress_shows_bar_and_marks_improvement(capsys):
    cb = optuna_progress_callback(fold=0, label="model", n_trials=30, progress_every=2)
    study = _Study(0.42, {"top_decile_net_bps": 8.0})
    cb(study, _Trial(0, 0.42, {"top_decile_net_bps": 8.0}))
    out = capsys.readouterr().out
    assert "trial#1" in out
    assert "composite=0.4200" in out
    assert "best=0.4200" in out
    assert "#" in out
    assert "*" in out

    study.best_value = 0.42
    cb(study, _Trial(1, 0.40, {"top_decile_net_bps": 2.0}))
    out2 = capsys.readouterr().out
    assert "trial#2" in out2
    assert "composite=0.4000" in out2
    assert "best=0.4200" in out2
    assert "*" not in out2
