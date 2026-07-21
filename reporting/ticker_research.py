"""Per-instrument research bundle for the web dashboard (JSON + HTML)."""

from __future__ import annotations

import html
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as _cfg
from config import CRYPTO_DISPLAY_NAMES, OUT_DIR, TRADFI_DISPLAY_NAMES
from data_platform.universe import is_crypto_symbol, is_tradfi_symbol

RESEARCH_DIR = OUT_DIR / "research"

_PLOT_STEMS = ("ret_z_explainer", "hmm_scatter", "hmm_timeline")


def _read_fusion() -> dict:
    path = OUT_DIR / "fusion_pipeline_report.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _display_name(sym: str) -> str:
    sym_u = sym.upper()
    if sym_u in CRYPTO_DISPLAY_NAMES:
        return CRYPTO_DISPLAY_NAMES[sym_u]
    if sym_u in TRADFI_DISPLAY_NAMES:
        return TRADFI_DISPLAY_NAMES[sym_u]
    return sym_u


def _asset_class(sym: str) -> str:
    if is_crypto_symbol(sym):
        return "crypto"
    if is_tradfi_symbol(sym):
        return "tradfi"
    return "other"


def _fold_rows(fusion: dict, sym: str) -> list[dict]:
    sym_u = sym.upper()
    rows: list[dict] = []
    for fold in fusion.get("walk_forward_folds") or []:
        if fold.get("skipped"):
            continue
        fd = (fold.get("fold_diagnostics") or {}).get("tickers") or {}
        entry = fd.get(sym_u) or fd.get(sym)
        if not entry:
            continue
        train = entry.get("train") or {}
        oos = entry.get("oos") or {}
        rows.append(
            {
                "fold": fold.get("fold"),
                "train_net_bps": train.get("net"),
                "oos_net_bps": oos.get("net"),
                "oos_gross_bps": oos.get("gross"),
                "bottleneck": entry.get("bottleneck"),
            }
        )
    return rows


def _plot_paths(sym: str) -> list[dict]:
    plots_dir = OUT_DIR / "plots"
    sym_u = sym.upper()
    out: list[dict] = []
    for stem in _PLOT_STEMS:
        name = f"{stem}_{sym_u}.png"
        p = plots_dir / name
        if p.is_file():
            out.append(
                {
                    "id": stem,
                    "title": stem.replace("_", " ").title(),
                    "path": f"/output/plots/{name}",
                }
            )
    return out


def _verdict(tradeable: bool, oos_net: float | None, backtest_return: float | None) -> str:
    if tradeable and (oos_net or 0) >= float(getattr(_cfg, "FUSION_MIN_TOP_DECILE_NET_BPS", 5.0)):
        return "TRADE"
    if (oos_net or 0) > 0 or (backtest_return or 0) > 0:
        return "WATCH"
    return "SKIP"


def build_ticker_research(sym: str, fusion: dict | None = None) -> dict:
    """Structured research payload for one ticker."""
    fusion = fusion if fusion is not None else _read_fusion()
    sym_u = sym.upper()
    decile = (fusion.get("decile_audit") or {}).get("by_ticker") or {}
    dec = decile.get(sym_u) or {}
    bt = (fusion.get("per_ticker_backtest") or {}).get(sym_u) or {}
    go = fusion.get("go_no_go") or {}
    tradeable_list = [str(t).upper() for t in (go.get("tradeable_tickers") or [])]
    tradeable = sym_u in tradeable_list or bool(dec.get("tradeable"))
    oos_net = dec.get("top_decile_net_bps")
    ret = bt.get("total_return_pct")
    folds = _fold_rows(fusion, sym_u)
    ok_folds = sum(1 for f in folds if f.get("bottleneck") == "ok")
    return {
        "ticker": sym_u,
        "display_name": _display_name(sym_u),
        "asset_class": _asset_class(sym_u),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "verdict": _verdict(tradeable, oos_net, ret),
        "tradeable": tradeable,
        "decile": {
            "top_decile_net_bps": dec.get("top_decile_net_bps"),
            "top_decile_gross_bps": dec.get("top_decile_gross_bps"),
            "tradeable": dec.get("tradeable"),
            "monotonic": dec.get("monotonic"),
        },
        "backtest": {
            k: bt.get(k)
            for k in (
                "total_return_pct",
                "sharpe",
                "max_drawdown_pct",
                "n_trades",
                "n_signals",
                "benchmark_return_pct",
                "excess_return_pct",
                "no_trades",
                "skipped",
                "reason",
            )
            if k in bt
        },
        "fold_history": folds,
        "fold_ok_count": ok_folds,
        "plots": _plot_paths(sym_u),
        "research_html": f"/output/research/{sym_u}.html",
        "research_json": f"/output/research/{sym_u}.json",
    }


def _fmt_num(val, *, suffix: str = "", digits: int = 2) -> str:
    if val is None:
        return "—"
    try:
        return f"{float(val):.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return str(val)


def _render_html(payload: dict) -> str:
    sym = payload["ticker"]
    esc = html.escape
    verdict = payload.get("verdict", "SKIP")
    dec = payload.get("decile") or {}
    bt = payload.get("backtest") or {}
    plots = payload.get("plots") or []
    folds = payload.get("fold_history") or []
    is_crypto = payload.get("asset_class") == "crypto"
    hero_cls = "hero btc" if is_crypto else "hero"

    fold_rows = ""
    for f in folds:
        fold_rows += (
            f"<tr><td>{esc(str(f.get('fold')))}</td>"
            f"<td>{_fmt_num(f.get('train_net_bps'))}</td>"
            f"<td>{_fmt_num(f.get('oos_net_bps'))}</td>"
            f"<td>{esc(str(f.get('bottleneck') or '—'))}</td></tr>"
        )
    if not fold_rows:
        fold_rows = '<tr><td colspan="4" class="muted">Нет fold-диагностики</td></tr>'

    plot_blocks = ""
    for p in plots:
        plot_blocks += (
            f'<figure class="plot-card"><img src="{esc(p["path"])}" alt="{esc(p["title"])}" />'
            f'<figcaption>{esc(p["title"])}</figcaption></figure>'
        )
    if not plot_blocks:
        plot_blocks = '<p class="muted">Диагностические графики появятся после plot-этапа pipeline.</p>'

    eyebrow = "Macro leg" if is_crypto else "Index research"

    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{esc(sym)} · Index Research</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500&family=Source+Serif+4:opsz,wght@8..60,500&display=swap" rel="stylesheet" />
  <style>
    :root {{ --paper:#fcfcfc; --linen:#faf8f5; --stone:#e4e2e1; --graphite:#32302f; --pebble:#686664; --charcoal:#09090a; --bronze:#3a3525; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:Inter,sans-serif; font-size:16px; line-height:1.5; color:var(--graphite); background:var(--paper); }}
    .wrap {{ max-width:1200px; margin:0 auto; padding:48px 24px 80px; }}
    .hero {{ padding:48px; border-radius:100px; background:var(--linen); margin-bottom:48px; }}
    .hero.btc {{ background:var(--bronze); color:var(--paper); }}
    .eyebrow {{ font-size:14px; color:var(--pebble); margin:0 0 12px; }}
    .hero.btc .eyebrow {{ color:rgba(252,252,252,0.7); }}
    h1 {{ font-family:"Source Serif 4",serif; font-size:56px; font-weight:500; letter-spacing:-0.01em; margin:0 0 8px; line-height:1.08; }}
    .subtitle {{ color:var(--pebble); margin:0 0 24px; }}
    .hero.btc .subtitle {{ color:rgba(252,252,252,0.8); }}
    .badge {{ display:inline-block; padding:6px 14px; border-radius:1600px; font-size:12px; letter-spacing:0.025em; text-transform:uppercase; border:1px solid var(--stone); margin-right:8px; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:16px; margin-top:24px; }}
    .stat {{ border-top:1px solid var(--stone); padding-top:16px; }}
    .hero.btc .stat {{ border-color:rgba(252,252,252,0.2); }}
    .stat label {{ display:block; font-size:12px; color:var(--pebble); margin-bottom:4px; }}
    .hero.btc .stat label {{ color:rgba(252,252,252,0.6); }}
    .stat strong {{ font-family:"Source Serif 4",serif; font-size:28px; font-weight:500; }}
    section {{ margin-top:48px; padding-top:48px; border-top:1px solid var(--stone); }}
    h2 {{ font-family:"Source Serif 4",serif; font-size:36px; font-weight:500; margin:0 0 24px; }}
    table {{ width:100%; border-collapse:collapse; }}
    th,td {{ padding:12px 8px; border-bottom:1px solid var(--stone); text-align:left; }}
    th {{ color:var(--pebble); font-weight:400; font-size:14px; }}
    .plots img {{ width:100%; border-radius:48px; background:var(--linen); }}
    .muted {{ color:var(--pebble); }}
    .back {{ display:inline-block; margin-bottom:24px; color:var(--graphite); font-size:14px; text-decoration:none; }}
  </style>
</head>
<body>
  <div class="wrap">
    <a class="back" href="/">← Index Lattice Desk</a>
    <header class="{hero_cls}">
      <p class="eyebrow">{eyebrow}</p>
      <h1>{esc(sym)}</h1>
      <p class="subtitle">{esc(payload.get("display_name", sym))}</p>
      <span class="badge">{esc(verdict)}</span>
      <span class="badge">{esc(payload.get("asset_class", ""))}</span>
      <div class="grid">
        <div class="stat"><label>Top-decile net</label><strong>{_fmt_num(dec.get('top_decile_net_bps'))} bps</strong></div>
        <div class="stat"><label>OOS return</label><strong>{_fmt_num(bt.get('total_return_pct'), suffix='%')}</strong></div>
        <div class="stat"><label>Sharpe</label><strong>{_fmt_num(bt.get('sharpe'))}</strong></div>
        <div class="stat"><label>Folds OK</label><strong>{payload.get('fold_ok_count', 0)}</strong></div>
      </div>
    </header>
    <section><h2>Walk-forward</h2><table><thead><tr><th>Fold</th><th>Train net</th><th>OOS net</th><th>Bottleneck</th></tr></thead><tbody>{fold_rows}</tbody></table></section>
    <section><h2>Диагностика</h2><div class="plots">{plot_blocks}</div></section>
  </div>
</body>
</html>"""


def write_ticker_research(sym: str, fusion: dict | None = None) -> tuple[Path, Path]:
    """Write JSON + HTML research for one ticker."""
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    payload = build_ticker_research(sym, fusion)
    sym_u = payload["ticker"]
    json_path = RESEARCH_DIR / f"{sym_u}.json"
    html_path = RESEARCH_DIR / f"{sym_u}.html"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    html_path.write_text(_render_html(payload), encoding="utf-8")
    return json_path, html_path


def write_ticker_research_bundle(
    tickers: list[str] | None = None,
    fusion: dict | None = None,
) -> list[Path]:
    """Generate research artifacts for all pipeline tickers."""
    fusion = fusion if fusion is not None else _read_fusion()
    if tickers is None:
        tickers = fusion.get("tickers") or fusion.get("symbols") or []
        if not tickers:
            tickers = list(getattr(_cfg, "FLOW_DEFAULT_TICKERS", []))
    written: list[Path] = []
    for sym in tickers:
        try:
            jp, hp = write_ticker_research(str(sym), fusion)
            written.extend([jp, hp])
        except Exception as exc:
            print(f"    research [{sym}]: skipped — {exc}", flush=True)
    index = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "instruments": [build_ticker_research(str(t), fusion) for t in tickers],
    }
    index_path = RESEARCH_DIR / "index.json"
    index_path.write_text(json.dumps(index, indent=2, default=str), encoding="utf-8")
    written.append(index_path)
    print(f"    ticker research: {len(tickers)} instruments -> {RESEARCH_DIR}", flush=True)
    return written
