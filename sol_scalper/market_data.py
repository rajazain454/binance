"""Multi-asset 5-minute market data feed and technical indicators for JUP, SOL, etc."""

import logging
from typing import Dict, Any, List, Optional
import requests
import numpy as np

logger = logging.getLogger("sol_scalper")


def fetch_ohlcv_candles(ticker: str, interval: str = "5m", limit: int = 100) -> Optional[List[Dict[str, float]]]:
    """Fetches institutional klines directly from Binance public API."""
    url = f"https://api.binance.com/api/v3/klines?symbol={ticker}&interval={interval}&limit={limit}"
    headers = {"User-Agent": "SolScalperMulti/1.0"}
    
    try:
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code == 200:
            raw_klines = r.json()
            candles = []
            for k in raw_klines:
                candles.append({
                    "timestamp": int(k[0]),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                })
            return candles
    except Exception as e:
        logger.warning("Error fetching %s candles: %s", ticker, e)
    return None


def calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
    ema = np.zeros_like(prices)
    alpha = 2.0 / (period + 1.0)
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1.0 - alpha) * ema[i - 1]
    return ema


def calculate_rsi_series(prices: np.ndarray, period: int = 14) -> np.ndarray:
    deltas = np.diff(prices)
    rsi = np.full(len(prices), 50.0)
    if len(prices) <= period + 1:
        return rsi

    up = deltas.copy()
    down = deltas.copy()
    up[up < 0] = 0.0
    down[down > 0] = 0.0
    down = np.abs(down)

    roll_up = np.zeros(len(deltas))
    roll_down = np.zeros(len(deltas))
    roll_up[period - 1] = np.mean(up[:period])
    roll_down[period - 1] = np.mean(down[:period])

    for i in range(period, len(deltas)):
        roll_up[i] = (roll_up[i - 1] * (period - 1) + up[i]) / period
        roll_down[i] = (roll_down[i - 1] * (period - 1) + down[i]) / period

    for i in range(period, len(deltas)):
        if roll_down[i] == 0:
            rsi[i + 1] = 100.0
        else:
            rs = roll_up[i] / roll_down[i]
            rsi[i + 1] = 100.0 - 100.0 / (1.0 + rs)

    return rsi


def calculate_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculates Wilder's Average True Range (ATR)."""
    tr = np.zeros(len(closes))
    tr[0] = highs[0] - lows[0]
    for i in range(1, len(closes)):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr[i] = max(hl, hc, lc)

    atr = np.zeros(len(closes))
    if len(closes) < period:
        return tr
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, len(closes)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def fetch_btc_macro_regime(strategy_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluates Bitcoin macro trend to prevent buying altcoins during market-wide dump regimes."""
    if not strategy_cfg.get("btc_macro_filter", True):
        return {"is_safe": True, "price": 0.0, "ema50": 0.0, "rsi": 50.0, "reason": "Filter Disabled"}

    tf = strategy_cfg.get("btc_timeframe", "1h")
    min_rsi = float(strategy_cfg.get("btc_min_rsi", 40.0))
    candles = fetch_ohlcv_candles("BTCUSDT", interval=tf, limit=70)
    if not candles or len(candles) < 55:
        return {"is_safe": True, "price": 0.0, "ema50": 0.0, "rsi": 50.0, "reason": "BTC Data Fallback"}

    closes = np.array([c["close"] for c in candles], dtype=float)
    ema50 = calculate_ema(closes, 50)
    rsi = calculate_rsi_series(closes, 14)

    curr_p = float(closes[-1])
    curr_ema = float(ema50[-1])
    curr_rsi = float(rsi[-1])

    is_safe = (curr_p >= curr_ema * 0.995) and (curr_rsi >= min_rsi)
    reason = "🟢 BTC Macro Safe" if is_safe else f"🔴 BTC Breakdown (${curr_p:,.0f} < EMA50 ${curr_ema:,.0f} or RSI {curr_rsi:.1f} < {min_rsi:.1f})"

    return {
        "is_safe": is_safe,
        "price": curr_p,
        "ema50": curr_ema,
        "rsi": curr_rsi,
        "reason": reason,
    }


def compute_symbol_snapshot(ticker: str, symbol: str, strategy_cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Calculates live 5m technical indicators for a single asset."""
    timeframe = strategy_cfg.get("timeframe", "5m")
    candles = fetch_ohlcv_candles(ticker, interval=timeframe, limit=100)
    if not candles or len(candles) < 55:
        return None

    closes = np.array([c["close"] for c in candles], dtype=float)
    highs = np.array([c["high"] for c in candles], dtype=float)
    lows = np.array([c["low"] for c in candles], dtype=float)
    volumes = np.array([c["volume"] for c in candles], dtype=float)

    curr_p = float(closes[-1])
    prev_p = float(closes[-2])

    # Indicator calculations
    ema_fast = calculate_ema(closes, strategy_cfg.get("ema_fast", 9))
    ema_slow = calculate_ema(closes, strategy_cfg.get("ema_slow", 21))
    ema_trend = calculate_ema(closes, strategy_cfg.get("ema_trend", 50))
    rsi = calculate_rsi_series(closes, strategy_cfg.get("rsi_period", 14))

    # ATR (Volatility) calculation
    atr_period = int(strategy_cfg.get("atr_period", 14))
    atr_series = calculate_atr(highs, lows, closes, period=atr_period)
    atr_val = float(atr_series[-1])
    atr_pct = float(atr_val / max(1e-6, curr_p) * 100.0)

    # Volume Confirmation calculation
    avg_vol = float(np.mean(volumes[-21:-1])) if len(volumes) >= 21 else float(np.mean(volumes))
    vol_ratio = float(volumes[-1] / max(1e-6, avg_vol))

    # Trend regime determination
    is_bullish = (ema_fast[-1] > ema_slow[-1]) and (ema_slow[-1] > ema_trend[-1]) and (curr_p > ema_trend[-1])
    is_bearish = (ema_fast[-1] < ema_slow[-1]) and (curr_p < ema_trend[-1])
    trend_state = "BULLISH_UPTREND" if is_bullish else ("BEARISH_DOWNTREND" if is_bearish else "NEUTRAL_CHOP")

    return {
        "symbol": symbol,
        "ticker": ticker,
        "timestamp": candles[-1]["timestamp"],
        "price": curr_p,
        "prev_price": prev_p,
        "open": float(candles[-1]["open"]),
        "prev_open": float(candles[-2]["open"]),
        "high": highs[-1],
        "low": lows[-1],
        "prev_low": lows[-2],
        "volume": volumes[-1],
        "avg_volume": avg_vol,
        "volume_ratio": vol_ratio,
        "atr": atr_val,
        "atr_pct": atr_pct,
        "ema_fast": float(ema_fast[-1]),
        "ema_slow": float(ema_slow[-1]),
        "ema_trend": float(ema_trend[-1]),
        "rsi": float(rsi[-1]),
        "prev_rsi": float(rsi[-2]),
        "trend_state": trend_state,
        "is_bullish": is_bullish,
        "is_bearish": is_bearish,
    }


def compute_basket_snapshots(basket: List[Dict[str, Any]], strategy_cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Fetches snapshots for all assets in the active basket."""
    snapshots = {}
    for item in basket:
        sym = item["symbol"]
        ticker = item["ticker"]
        snap = compute_symbol_snapshot(ticker, sym, strategy_cfg)
        if snap:
            snapshots[sym] = snap
    return snapshots
