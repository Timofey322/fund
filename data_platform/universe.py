"""Multi-asset universe: crypto (Binance) + US-listed ETFs / indices (yfinance)."""

from __future__ import annotations

import config as _cfg
from data_platform.binance import CRYPTO_SYMBOLS, is_crypto_symbol

# US-listed ETFs + Finam desk aliases (MOEX / yfinance index).
TRADFI_UNIVERSE = list(getattr(_cfg, "TRADFI_UNIVERSE", ["SPY"]))
TRADFI_SYMBOLS = frozenset(TRADFI_UNIVERSE)

_FINAM_YF = {
    "NASDAQ": "^IXIC",
    "SP500": "^GSPC",
    "DAX": "^GDAXI",
    "GER40": "^GDAXI",
}
_MOEX_SYMBOLS = frozenset({"GAZP", "SBER", "IMOEX"})
YFINANCE_TICKERS: dict[str, str] = {sym: sym for sym in TRADFI_UNIVERSE}
YFINANCE_TICKERS.update({k: v for k, v in _FINAM_YF.items() if k in TRADFI_SYMBOLS})

_cfg_names = getattr(_cfg, "TRADFI_DISPLAY_NAMES", {})
TRADFI_DISPLAY_NAMES: dict[str, str] = {
    sym: str(_cfg_names.get(sym, sym)) for sym in TRADFI_UNIVERSE
}


def is_tradfi_symbol(symbol: str) -> bool:
    return symbol.upper() in TRADFI_SYMBOLS


def is_supported_symbol(symbol: str) -> bool:
    sym = symbol.upper()
    return is_crypto_symbol(sym) or is_tradfi_symbol(sym)


def needs_tick_flow(symbol: str) -> bool:
    """Only crypto has real taker buy/sell counts; tradfi uses candle-imputed flow."""
    return is_crypto_symbol(symbol)


def is_moex_symbol(symbol: str) -> bool:
    return symbol.upper() in _MOEX_SYMBOLS


def yfinance_ticker(symbol: str) -> str:
    sym = symbol.upper()
    if sym in _MOEX_SYMBOLS:
        raise ValueError(f"{sym} uses MOEX ISS, not yfinance")
    if sym not in YFINANCE_TICKERS:
        raise ValueError(f"No yfinance mapping for {symbol!r} (supported: {sorted(YFINANCE_TICKERS)})")
    return YFINANCE_TICKERS[sym]


def parse_tickers(raw: str | list[str]) -> list[str]:
    if isinstance(raw, str):
        parts = [p.strip().upper() for p in raw.split(",") if p.strip()]
    else:
        parts = [str(p).strip().upper() for p in raw if str(p).strip()]
    out: list[str] = []
    for sym in parts:
        if not is_supported_symbol(sym):
            raise ValueError(
                f"Unsupported ticker {sym!r}. "
                f"Crypto: {sorted(CRYPTO_SYMBOLS)} | TradFi: {sorted(TRADFI_SYMBOLS)}"
            )
        if sym not in out:
            out.append(sym)
    return out


def split_tickers(tickers: list[str]) -> tuple[list[str], list[str]]:
    crypto = [t for t in tickers if is_crypto_symbol(t)]
    tradfi = [t for t in tickers if is_tradfi_symbol(t)]
    return crypto, tradfi


def commission_bps_for_ticker(ticker: str) -> float:
    """Per-side commission in bps (crypto 0.011%, tradfi 0.04%)."""
    sym = str(ticker).upper()
    if is_tradfi_symbol(sym):
        return float(getattr(_cfg, "COMMISSION_BPS_TRADFI", 4.0))
    return float(getattr(_cfg, "COMMISSION_BPS_CRYPTO", 1.1))
