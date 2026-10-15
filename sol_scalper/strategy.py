"""Systematic High-Probability Trend Following & Pullback Strategy Engine for SOL/USDC."""

import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger("sol_scalper")


def evaluate_scalp_entry(snapshot: Dict[str, Any], strategy_cfg: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Evaluates whether institutional momentum and pullback criteria align for a BUY scalp.
    Criteria:
    1. Trend Filter: SOL is in an active Bullish Trend (EMA 9 > EMA 21 and Price > EMA 50).
    2. Pullback Zone: RSI entered the dip zone (<= 45) and is now ticking upward.
    3. Price Action: Current price is rebounding back above the fast EMA (9).
    """
    price = snapshot["price"]
    ema_fast = snapshot["ema_fast"]
    ema_slow = snapshot["ema_slow"]
    ema_trend = snapshot["ema_trend"]
    rsi = snapshot["rsi"]
    prev_rsi = snapshot["prev_rsi"]

    require_trend = strategy_cfg.get("require_trend_alignment", True)
    rsi_oversold = float(strategy_cfg.get("rsi_oversold_threshold", 42.0))

    # 1. Trend Direction Check
    if require_trend and not snapshot["is_bullish_trend"]:
        return False, "Waiting for bullish trend alignment (EMA 9 > 21 > 50)"

    # 2. Oversold Pullback or EMA Support Bounce
    is_rsi_rebound = (rsi <= 48.0) and (rsi > prev_rsi)
    is_ema_bounce = (price >= ema_slow) and (snapshot["low"] <= ema_slow * 1.002)

    if not (is_rsi_rebound or is_ema_bounce):
        return False, f"Price in momentum zone but awaiting pullback trigger (RSI: {rsi:.1f})"

    # 3. Candle Confirmation
    if price < ema_fast * 0.999:
        return False, f"Awaiting candle reclaim above 9 EMA (${ema_fast:.2f})"

    reason = f"Bullish Pullback Reversal | RSI: {rsi:.1f} (Ticking Up) | EMA 9 (${ema_fast:.2f}) > EMA 21 (${ema_slow:.2f})"
    return True, reason


def evaluate_active_position(
    pos: Dict[str, Any], 
    snapshot: Dict[str, Any], 
    trading_cfg: Dict[str, Any]
) -> Tuple[bool, str, str]:
    """
    Evaluates open scalp position for automated exit triggers (TP, SL, Trailing, or Invalidation).
    Returns (should_exit, outcome, reason).
    """
    entry_p = pos["entry_price_usd"]
    curr_p = snapshot["price"]
    peak_p = max(pos.get("peak_price_usd", entry_p), curr_p)
    pos["peak_price_usd"] = peak_p

    gain_pct = (curr_p - entry_p) / entry_p * 100.0
    peak_gain_pct = (peak_p - entry_p) / entry_p * 100.0

    tp_target = float(trading_cfg.get("take_profit_pct", 1.8))
    sl_target = float(trading_cfg.get("stop_loss_pct", 0.9))
    trailing_on = trading_cfg.get("trailing_stop_enabled", True)
    trailing_act = float(trading_cfg.get("trailing_activation_pct", 1.2))
    trailing_cb = float(trading_cfg.get("trailing_callback_pct", 0.4))

    # 1. Trailing Stop Exit (Locks in runners)
    if trailing_on and peak_gain_pct >= trailing_act:
        trailing_stop_price = peak_p * (1.0 - (trailing_cb / 100.0))
        if curr_p <= trailing_stop_price:
            return True, "WIN", f"🚀 Trailing Profit Secured (+{gain_pct:.2f}% | Peaked at +{peak_gain_pct:.2f}%)"

    # 2. Hard Take-Profit Hit (+1.8%)
    if gain_pct >= tp_target:
        return True, "WIN", f"🎯 Target Take Profit Hit (+{gain_pct:.2f}% >= +{tp_target:.1f}%)"

    # 3. Strict Stop-Loss Hit (-0.9%)
    if gain_pct <= -sl_target:
        return True, "LOSS", f"🛡️ Risk Guard Stop Loss (-{abs(gain_pct):.2f}% <= -{sl_target:.1f}%)"

    # 4. Trend Breakdown Invalidation (Emergency Exit)
    if curr_p < snapshot["ema_trend"] and snapshot["rsi"] < 34.0:
        return True, "LOSS", f"⚠️ Macro Trend Invalidation (Dropped below 50 EMA at ${snapshot['ema_trend']:.2f})"

    # 5. Stale Micro-Scalp Timeout (8 minutes: cut dead sideways trades)
    import time
    max_hold_sec = float(trading_cfg.get("max_hold_seconds", 480))
    entry_ts = pos.get("entry_time_ts", time.time())
    hold_duration_sec = time.time() - entry_ts
    if hold_duration_sec >= max_hold_sec:
        outcome = "WIN" if gain_pct > 0 else "LOSS"
        return True, outcome, f"⏳ Micro-Scalp Timeout ({hold_duration_sec/60:.1f}m reached | {gain_pct:+.2f}%)"

    return False, "HOLD", "Active trade tracking target"
