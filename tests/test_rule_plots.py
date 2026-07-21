"""Tests for rule strategy visual artifacts."""

from __future__ import annotations

from rule.web_export import _web_plot_path


def test_web_plot_path_normalization():
    assert _web_plot_path("/output/plots/rule/rule_equity_curve.png") == "/output/plots/rule/rule_equity_curve.png"
    win = "D:\\me\\output\\plots\\rule\\rule_equity_curve.png"
    assert _web_plot_path(win).replace("\\", "/") == "/output/plots/rule/rule_equity_curve.png"
