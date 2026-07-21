"""Rule-based index strategy (no ML) — isolated pipeline package."""

from rule.config import RULE_DEFAULT_TICKERS, RULE_NAME
from rule.pipeline import run_rule_pipeline

__all__ = ["RULE_NAME", "RULE_DEFAULT_TICKERS", "run_rule_pipeline"]
