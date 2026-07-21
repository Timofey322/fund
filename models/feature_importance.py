"""Aggregate and persist ML feature importance after walk-forward training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

import config as _cfg
from config import OUT_DIR
from research.features.registry import aggregate_importance_by_group, merge_fold_importances

IMPORTANCE_CACHE_PATH = OUT_DIR / "cache" / "ml_feature_importance.json"


def build_ml_feature_importance_report(
    wf_folds: list[dict],
    feat_cols: list[str] | None = None,
    *,
    top_n: int = 30,
) -> dict[str, Any]:
    """Merge per-fold LightGBM importances into a ranked report."""
    active = [f for f in wf_folds if not f.get("skipped")]
    merged = merge_fold_importances(active)
    ranked = sorted(merged.items(), key=lambda x: abs(float(x[1])), reverse=True)
    top_features = [
        {"feature": name, "importance": round(float(val), 6)}
        for name, val in ranked[:top_n]
    ]
    per_fold: list[dict[str, Any]] = []
    for fold in active:
        tops = fold.get("top_features") or {}
        if not tops:
            continue
        top5 = sorted(tops.items(), key=lambda x: abs(float(x[1])), reverse=True)[:5]
        per_fold.append({
            "fold": int(fold.get("fold", -1)),
            "test_start": fold.get("test_start"),
            "top_features": {k: round(float(v), 6) for k, v in top5},
            "auc": (fold.get("oos_metrics") or {}).get("auc"),
        })
    by_group = aggregate_importance_by_group(merged) if merged else {"groups": [], "n_features": 0}
    return {
        "n_folds": len(active),
        "n_features_tracked": len(merged),
        "feature_columns": list(feat_cols or []),
        "top_features": top_features,
        "per_fold_top5": per_fold,
        "by_group": by_group,
    }


def persist_feature_importance_report(report: dict[str, Any], path: Path | None = None) -> Path:
    out = path or IMPORTANCE_CACHE_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    summary_path = out.parent / "ml_feature_importance_summary.md"
    lines = [
        "# ML feature importance (walk-forward aggregate)",
        "",
        f"Folds: {report.get('n_folds', 0)} | Features tracked: {report.get('n_features_tracked', 0)}",
        "",
        "## Top features",
        "",
        "| Rank | Feature | Importance |",
        "|------|---------|------------|",
    ]
    for i, row in enumerate(report.get("top_features") or [], 1):
        lines.append(f"| {i} | {row['feature']} | {row['importance']:.4f} |")
    groups = (report.get("by_group") or {}).get("groups") or []
    if groups:
        lines.extend(["", "## By feature group", ""])
        for g in groups:
            label = g.get("label") or g.get("group") or g.get("id") or "?"
            pct = g.get("importance_pct", g.get("importance_share_pct", 0))
            lines.append(f"- **{label}**: {float(pct):.1f}%")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def build_and_save_feature_importance_report(
    wf_folds: list[dict],
    feat_cols: list[str] | None = None,
) -> dict[str, Any]:
    report = build_ml_feature_importance_report(wf_folds, feat_cols)
    persist_feature_importance_report(report)
    return report
