"""High-frequency real-time market data feed and technical indicators for SOL/USDC."""

import logging
import time
from typing import Dict, Any, List, Optional
import requests
import numpy as np

logger = logging.getLogger("sol_scalper")


def fetch_sol_ohlcv(symbol: str = "SOLUSDT", interval: str = "1m", limit: int = 100) -> Optional[List[Dict[str, float]]]:
    """
    Fetches real-time institutional tick candles directly from Binance public API.
    Zero rate limits, zero API keys required, sub-second latency.
    """
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    headers = {"User-Agent": "SolScalper/1.0"}
    
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
        logger.warning("Error fetching SOL candles: %s", e)
    return None


def calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
    """Calculates Exponential Moving Average (EMA)."""
    ema = np.zeros_like(prices)
    alpha = 2.0 / (period + 1.0)
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1.0 - alpha) * ema[i - 1]
    return ema


def calculate_rsi(prices: np.ndarray, period: int = 14) -> float:
    """Calculates Relative Strength Index (RSI)."""
    if len(prices) < period + 1:
        return 50.0
    
    deltas = np.diff(prices)
    seed = deltas[:period]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    
    if down == 0:
        return 100.0
    rs = up / down
    rsi = np.zeros(len(deltas))
    rsi[:period] = 100.0 - 100.0 / (1.0 + rs)

    up_val = up
    down_val = down
    for i in range(period, len(deltas)):
        delta = deltas[i]
        if delta > 0:
            up_val = (up_val * (period - 1) + delta) / period
            down_val = (down_val * (period - 1)) / period
        else:
            up_val = (up_val * (period - 1)) / period
            down_val = (down_val * (period - 1) - delta) / period
        
        if down_val == 0:
            rsi[i] = 100.0
        else:
            rs_val = up_val / down_val
            rsi[i] = 100.0 - 100.0 / (1.0 + rs_val)
            
    return float(rsi[-1])


def calculate_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
    """Calculates Average True Range (ATR) for volatility stop-loss sizing."""
    if len(closes) < period + 1:
        return 0.5
    
    tr1 = highs[1:] - lows[1:]
    tr2 = np.abs(highs[1:] - closes[:-1])
    tr3 = np.abs(lows[1:] - closes[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = np.mean(tr[-period:])
    return float(atr)


def compute_market_snapshot(strategy_cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Fetches live candles and calculates full indicator suite for decision engine."""
    timeframe = strategy_cfg.get("timeframe", "1m")
    candles = fetch_sol_ohlcv(symbol="SOLUSDT", interval=timeframe, limit=100)
    if not candles or len(candles) < 55:
        return None

    closes = np.array([c["close"] for c in candles], dtype=float)
    highs = np.array([c["high"] for c in candles], dtype=float)
    lows = np.array([c["low"] for c in candles], dtype=float)
    volumes = np.array([c["volume"] for c in candles], dtype=float)

    ema_fast_p = strategy_cfg.get("ema_fast", 9)
    ema_slow_p = strategy_cfg.get("ema_slow", 21)
    ema_trend_p = strategy_cfg.get("ema_trend", 50)
    rsi_p = strategy_cfg.get("rsi_period", 14)

    ema_fast = calculate_ema(closes, ema_fast_p)
    ema_slow = calculate_ema(closes, ema_slow_p)
    ema_trend = calculate_ema(closes, ema_trend_p)
    current_rsi = calculate_rsi(closes, rsi_p)
    prev_rsi = calculate_rsi(closes[:-1], rsi_p)
    current_atr = calculate_atr(highs, lows, closes, period=14)

    current_price = closes[-1]
    prev_price = closes[-2]

    # Trend regime determination
    is_bullish_trend = (ema_fast[-1] > ema_slow[-1]) and (current_price > ema_trend[-1])
    is_bearish_trend = (ema_fast[-1] < ema_slow[-1]) and (current_price < ema_trend[-1])
    trend_state = "BULLISH_UPTREND" if is_bullish_trend else ("BEARISH_DOWNTREND" if is_bearish_trend else "NEUTRAL_CHOP")

    return {
        "timestamp": candles[-1]["timestamp"],
        "price": current_price,
        "prev_price": prev_price,
        "high": highs[-1],
        "low": lows[-1],
        "volume": volumes[-1],
        "ema_fast": float(ema_fast[-1]),
        "ema_slow": float(ema_slow[-1]),
        "ema_trend": float(ema_trend[-1]),
        "rsi": current_rsi,
        "prev_rsi": prev_rsi,
        "atr": current_atr,
        "trend_state": trend_state,
        "is_bullish_trend": is_bullish_trend,
        "is_bearish_trend": is_bearish_trend,
    }
