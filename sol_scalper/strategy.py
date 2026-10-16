"""Quantitative 15-Minute Institutional Strategy Engine with BTC Macro Gate, ATR Targets, and Volume Confirmation."""

import time
import logging
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger("sol_scalper")


def evaluate_basket_entry(
    snapshot: Dict[str, Any], 
    strategy_cfg: Dict[str, Any],
    btc_regime: Optional[Dict[str, Any]] = None
) -> Tuple[bool, str, Dict[str, float]]:
    """
    Evaluates institutional 2-bar pullback inside macro uptrend with 3 institutional filters:
    1. BTC Macro Gate: Ensures Bitcoin is not in a market-wide dump breakdown.
    2. Macro Bullish Trend: EMA 9 > EMA 21 > EMA 50 and Price > EMA 50.
    3. Pullback Bar (t-1): Dipped into 21 EMA support or RSI dipped <= 48.
    4. Confirmed Green Reversal Bar (t): Close > Open AND Close > Prev_Close AND Close >= 9 EMA.
    5. Volume Confirmation: Reversal candle volume >= 1.1x 20-period average volume.
    6. Dynamic ATR Targets: Calculates volatility-adjusted TP and SL levels.
    """
    # 1. Bitcoin Macro Gate Check
    if btc_regime and not btc_regime.get("is_safe", True):
        return False, f"Blocked by BTC Macro Gate: {btc_regime.get('reason', 'BTC dump regime')}", {}

    price = snapshot["price"]
    prev_price = snapshot["prev_price"]
    ema_fast = snapshot["ema_fast"]
    ema_slow = snapshot["ema_slow"]
    ema_trend = snapshot["ema_trend"]
    prev_rsi = snapshot["prev_rsi"]
    prev_low = snapshot["prev_low"]

    # 2. Macro Trend Check (Must be aligned in uptrend)
    if not snapshot.get("is_bullish", False):
        return False, "Awaiting macro bullish trend alignment (EMA 9 > 21 > 50)", {}

    # 3. Pullback Check (Bar t-1 tested 21 EMA or oversold dip)
    dip_threshold = float(strategy_cfg.get("rsi_dip_threshold", 48.0))
    touched_support = (prev_low <= ema_slow * 1.002) or (prev_rsi <= dip_threshold)
    if not touched_support:
        return False, f"In bull trend but awaiting pullback to 21 EMA / RSI dip (RSI: {snapshot['rsi']:.1f})", {}

    # 4. Upper RSI Safety Check (Do not buy overextended tops)
    max_entry_rsi = float(strategy_cfg.get("rsi_max_entry", 58.0))
    if snapshot["rsi"] > max_entry_rsi:
        return False, f"Pullback bounce already overextended (RSI: {snapshot['rsi']:.1f} > {max_entry_rsi:.1f})", {}

    # 5. Confirmed Green Reversal Bar Check
    candle_is_green = price > snapshot.get("open", prev_price)
    reversal_candle = (price > prev_price) and candle_is_green and (price >= ema_fast * 0.999)
    if not reversal_candle:
        return False, f"Awaiting confirmed green reversal close above 9 EMA (${ema_fast:.4f})", {}

    # 6. Volume Confirmation Check
    min_vol_ratio = float(strategy_cfg.get("min_volume_ratio", 1.10))
    vol_ratio = float(snapshot.get("volume_ratio", 1.0))
    if vol_ratio < min_vol_ratio:
        return False, f"Awaiting volume confirmation ({vol_ratio:.2f}x < {min_vol_ratio:.2f}x avg)", {}

    # 7. Calculate Dynamic ATR Targets
    use_atr = strategy_cfg.get("use_dynamic_atr", True)
    atr_pct = float(snapshot.get("atr_pct", 1.5))
    if use_atr:
        sl_mult = float(strategy_cfg.get("atr_sl_multiplier", 1.5))
        tp_mult = float(strategy_cfg.get("atr_tp_multiplier", 3.5))
        sl_pct = max(1.5, min(3.5, round(atr_pct * sl_mult, 2)))
        tp_pct = max(3.5, min(8.0, round(atr_pct * tp_mult, 2)))
    else:
        sl_pct = float(strategy_cfg.get("stop_loss_pct", 2.00))
        tp_pct = float(strategy_cfg.get("take_profit_pct", 5.00))

    dynamic_targets = {
        "tp_pct": tp_pct,
        "sl_pct": sl_pct,
        "trailing_act": round(sl_pct * 1.25, 2),
        "trailing_cb": round(sl_pct * 0.50, 2),
        "target_tp_price": round(price * (1.0 + tp_pct / 100.0), 6),
        "target_sl_price": round(price * (1.0 - sl_pct / 100.0), 6),
    }

    reason = (
        f"Confirmed 15M Bounce | Vol: {vol_ratio:.2f}x | ATR: {atr_pct:.2f}% "
        f"(TP: +{tp_pct:.1f}% / SL: -{sl_pct:.1f}%) | RSI: {prev_rsi:.1f} -> {snapshot['rsi']:.1f}"
    )
    return True, reason, dynamic_targets


def evaluate_active_position(
    pos: Dict[str, Any], 
    snapshot: Dict[str, Any], 
    trading_cfg: Dict[str, Any]
) -> Tuple[bool, str, str]:
    """
    Evaluates open scalp position for automated exit triggers using tailored ATR targets:
    1. Take-Profit: Volatility target hit (+3.5% to +8.0%)
    2. Trailing Profit Lock: Secured runner if price pulled back after expansion
    3. Stop-Loss: Hard cut at ATR stop (-1.5% to -3.5%)
    4. Timeout: Stale momentum cut after 24 bars (6 hours)
    """
    entry_p = pos["entry_price_usd"]
    curr_p = snapshot["price"]
    peak_p = max(pos.get("peak_price_usd", entry_p), curr_p)
    pos["peak_price_usd"] = peak_p

    gain_pct = (curr_p - entry_p) / entry_p * 100.0
    peak_gain_pct = (peak_p - entry_p) / entry_p * 100.0

    # Use tailored dynamic targets from position if available, fallback to config
    tp_target = float(pos.get("tp_pct", trading_cfg.get("take_profit_pct", 5.00)))
    sl_target = float(pos.get("sl_pct", trading_cfg.get("stop_loss_pct", 2.00)))
    trailing_on = trading_cfg.get("trailing_stop_enabled", True)
    trailing_act = float(pos.get("trailing_act", trading_cfg.get("trailing_activation_pct", 2.50)))
    trailing_cb = float(pos.get("trailing_cb", trading_cfg.get("trailing_callback_pct", 1.00)))

    # 1. Take Profit Hit
    if gain_pct >= tp_target:
        return True, "WIN", f"🎯 Target Take Profit Hit (+{gain_pct:.2f}% >= +{tp_target:.2f}%)"

    # 2. Trailing Profit Lock (Secures runners)
    if trailing_on and peak_gain_pct >= trailing_act:
        trailing_stop_price = peak_p * (1.0 - (trailing_cb / 100.0))
        if curr_p <= trailing_stop_price:
            outcome = "WIN" if gain_pct > 0 else "LOSS"
            return True, outcome, f"🚀 Trailing Profit Secured (+{gain_pct:.2f}% | Peaked at +{peak_gain_pct:.2f}%)"

    # 3. Hard Stop Loss Cut
    if gain_pct <= -sl_target:
        return True, "LOSS", f"🛡️ Risk Guard Stop Loss (-{abs(gain_pct):.2f}% <= -{sl_target:.2f}%)"

    # 4. Timeout Cut (24 bars = 6 hours on 15m)
    entry_ts = pos.get("entry_time_ts", time.time())
    mins_held = (time.time() - entry_ts) / 60.0
    bar_mins = 15.0 if "15" in str(trading_cfg.get("timeframe", "15m")) else 5.0
    max_mins = float(trading_cfg.get("max_hold_bars", 24)) * bar_mins
    if mins_held >= max_mins:
        outcome = "WIN" if gain_pct > 0 else "LOSS"
        return True, outcome, f"⏳ Stale Scalp Cut ({mins_held:.0f}m held | {gain_pct:+.2f}%)"

    return False, "HOLD", "Tracking targets"
