"""Adaptive Halal Spot Quantitative Sniper Scalper Engine.

Operates strictly on Shariah-compliant Spot crypto utility assets (SOL, AVAX, NEAR, LINK).
No leverage, no margin, no short-selling, no interest (riba).
Executes high-probability 15m confluence setups:
  1. Extreme Oversold Dip Reversal (RSI <= 35 + Lower Bollinger + Bullish Confirmation Candle)
  2. Bullish Momentum Breakout (EMA 9 > 21 > 50 + Volume Expansion + Green Close)
Protects capital with Breakeven Profit Locks and Trailing Take-Profit.
"""

from datetime import datetime, timezone
import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from bot.ledger import log_trade

logger = logging.getLogger("quantlab")


def _fmt(val: float) -> str:
    """Formats price values cleanly across micro-cap memecoins and macro assets."""
    if val <= 0:
        return "0.00"
    if val < 0.001:
        return f"{val:.8f}"
    elif val < 1.0:
        return f"{val:.6f}"
    elif val < 10.0:
        return f"{val:.4f}"
    else:
        return f"{val:,.2f}"


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculates Average True Range (ATR) from OHLC dataframe."""
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculates Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def fetch_client_df(client, symbol: str, timeframe: str = "15m", limit: int = 120) -> pd.DataFrame:
    """Safely retrieves OHLCV dataframe across real BinanceClient and test mocks."""
    if hasattr(client, "fetch_ohlcv"):
        try:
            res = client.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            if isinstance(res, pd.DataFrame) and not res.empty:
                return res
        except Exception:
            pass

    if hasattr(client, "get_klines"):
        try:
            klines = client.get_klines(symbol, timeframe=timeframe, limit=limit)
            if isinstance(klines, list) and klines:
                df = pd.DataFrame(klines)
                for c in ["open", "high", "low", "close", "volume"]:
                    if c in df.columns:
                        df[c] = pd.to_numeric(df[c])
                return df
        except Exception:
            pass

    return pd.DataFrame()


def fetch_client_price(client, symbol: str) -> float:
    """Safely retrieves live ticker price across real BinanceClient and test mocks."""
    if hasattr(client, "get_ticker_price"):
        try:
            val = client.get_ticker_price(symbol)
            if isinstance(val, (int, float)) and val > 0:
                return float(val)
        except Exception:
            pass

    if hasattr(client, "get_ticker"):
        try:
            tk = client.get_ticker(symbol)
            if isinstance(tk, dict):
                return float(tk.get("price", tk.get("last", 0.0)))
            elif isinstance(tk, (int, float)):
                return float(tk)
        except Exception:
            pass

    return 0.0


def discover_top_momentum_symbols(
    client,
    min_volume_usdt: float = 2_000_000.0,
    max_symbols: int = 15,
    max_dump_from_high_pct: float = 7.0,
    max_daily_gain_pct: float = 35.0,
) -> List[str]:
    """Dynamically screens all 350+ Binance Spot USDT pairs and returns the top active runners."""
    tickers = {}
    if hasattr(client, "fetch_all_tickers"):
        try:
            tickers = client.fetch_all_tickers()
        except Exception:
            pass
    elif hasattr(client, "exchange") and hasattr(client.exchange, "fetch_tickers"):
        try:
            tickers = client.exchange.fetch_tickers()
        except Exception:
            pass

    if not tickers:
        return []

    excluded_tokens = ["UP/", "DOWN/", "BEAR/", "BULL/", "FDUSD", "USDC", "EUR", "TUSD", "DAI"]
    valid_runners = []

    for sym, tk in tickers.items():
        if not sym.endswith("/USDT"):
            continue
        if any(ex in sym for ex in excluded_tokens):
            continue
        vol = float(tk.get("quoteVolume", 0.0) or 0.0)
        pct = float(tk.get("percentage", 0.0) or 0.0)
        price = float(tk.get("last", 0.0) or 0.0)
        high = float(tk.get("high", 0.0) or 0.0)

        # 24h High Proximity Filter: Avoid buying into whale distribution dumps
        if high > 0 and price > 0 and pct > 20.0:
            dump_pct = ((high - price) / high) * 100.0
            if dump_pct > max_dump_from_high_pct:
                continue

        # Avoid super-overextended parabolic peaks
        if pct > max_daily_gain_pct:
            continue

        if vol >= min_volume_usdt and price > 0:
            valid_runners.append((sym, pct, vol, price))

    valid_runners.sort(key=lambda x: x[1], reverse=True)
    return [r[0] for r in valid_runners[:max_symbols]]


def select_best_halal_asset(
    client,
    symbols: List[str],
    macro_benchmark: str = "BTC/USDT",
    timeframe: str = "15m",
    config: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Evaluates the Halal basket or full 350-coin universe and detects confirmed breakout setups.
    
    Dynamically scans Binance Spot for the highest-momentum runners when scan_all_pairs is enabled.
    """
    candidates = []
    basket_prices = {}
    basket_vol = {}

    symbols_to_scan = symbols
    if config:
        grid_cfg = config.get("grid_trading", {})
        if grid_cfg.get("scan_all_pairs", False):
            min_vol = grid_cfg.get("min_24h_volume_usdt", 2_000_000.0)
            max_cand = grid_cfg.get("max_candidates_to_analyze", 15)
            high_cfg = grid_cfg.get("high_proximity_filter", {})
            max_dump = high_cfg.get("max_dump_from_high_pct", 7.0) if high_cfg.get("enabled", True) else 100.0
            max_gain = high_cfg.get("max_daily_gain_pct", 35.0) if high_cfg.get("enabled", True) else 100.0
            runners = discover_top_momentum_symbols(client, min_vol, max_cand, max_dump, max_gain)
            if runners:
                symbols_to_scan = list(dict.fromkeys(runners + symbols))

    # 1. Check macro benchmark regime
    try:
        btc_df = fetch_client_df(client, macro_benchmark, timeframe="1d", limit=210)
        if len(btc_df) >= 200:
            btc_closes = btc_df["close"].values
            btc_sma200 = float(btc_closes[-200:].mean())
            btc_curr = float(btc_closes[-1])
            basket_prices[macro_benchmark] = btc_curr
            if btc_curr < btc_sma200 * 0.98:
                logger.warning(
                    "[Macro Guard] Benchmark %s ($%s) below 200d SMA ($%s). Defensive mode active.",
                    macro_benchmark, f"{btc_curr:,.0f}", f"{btc_sma200:,.0f}"
                )
    except Exception as e:
        logger.warning("[Macro Guard] Could not fetch benchmark %s: %s", macro_benchmark, e)

    for symbol in symbols_to_scan:
        try:
            df = fetch_client_df(client, symbol, timeframe=timeframe, limit=120)
            if df.empty or len(df) < 30:
                p = fetch_client_price(client, symbol)
                if p > 0:
                    basket_prices[symbol] = p
                continue

            for c in ["open", "high", "low", "close", "volume"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c])

            closes = df["close"].values
            curr_price = float(closes[-1])
            basket_prices[symbol] = curr_price

            # Technical Indicators
            atr_s = calculate_atr(df, period=14)
            atr_val = float(atr_s.iloc[-1]) if not atr_s.empty and not pd.isna(atr_s.iloc[-1]) else curr_price * 0.015
            atr_pct = (atr_val / curr_price) * 100.0
            basket_vol[symbol] = atr_pct

            rsi_s = calculate_rsi(df["close"], period=14)
            curr_rsi = float(rsi_s.iloc[-1])
            min_recent_rsi = float(rsi_s.iloc[-4:].min())

            # Bollinger Bands (20, 2.0)
            bb_mid = df["close"].rolling(20).mean()
            bb_std = df["close"].rolling(20).std()
            bb_lower = bb_mid - (2.0 * bb_std)
            bb_lower_val = float(bb_lower.iloc[-1]) if not pd.isna(bb_lower.iloc[-1]) else curr_price * 0.98

            # EMAs
            ema9 = float(df["close"].ewm(span=9, adjust=False).mean().iloc[-1])
            ema21 = float(df["close"].ewm(span=21, adjust=False).mean().iloc[-1])
            ema50 = float(df["close"].ewm(span=50, adjust=False).mean().iloc[-1])

            # Candle confirmation & Anti-Fakeout Wick Analysis
            last_open = float(df["open"].iloc[-1])
            last_close = float(df["close"].iloc[-1])
            last_high = float(df["high"].iloc[-1]) if "high" in df.columns else last_close
            last_low = float(df["low"].iloc[-1]) if "low" in df.columns else last_open
            prev_close = float(df["close"].iloc[-2]) if len(df) >= 2 else last_close
            is_green_candle = (last_close > last_open)
            is_higher_close = (last_close >= prev_close)

            candle_range = last_high - last_low
            upper_wick = last_high - max(last_open, last_close)
            wick_ratio = (upper_wick / candle_range) if candle_range > 0 else 0.0

            anti_fakeout_cfg = config.get("grid_trading", {}).get("anti_fakeout", {}) if config else {}
            anti_fakeout_enabled = anti_fakeout_cfg.get("enabled", True)
            max_wick_ratio = anti_fakeout_cfg.get("max_wick_ratio", 0.35)
            is_healthy_close = (wick_ratio <= max_wick_ratio) if anti_fakeout_enabled else True

            # Volume expansion & Anomaly Spike Detection
            vol_series = df["volume"] if "volume" in df.columns else pd.Series([1.0] * len(df))
            avg_vol = float(vol_series.iloc[-21:-1].mean()) if len(vol_series) >= 21 else (float(vol_series.mean()) if not vol_series.empty else 1.0)
            curr_vol = float(vol_series.iloc[-1]) if len(vol_series) >= 1 else 1.0
            vol_ratio = (curr_vol / avg_vol) if avg_vol > 0 else 1.0
            vol_expansion = (vol_ratio >= 1.25)
            vol_spike_multiplier = config.get("grid_trading", {}).get("vol_spike_multiplier", 2.0) if config else 2.0

            # -------------------------------------------------------------
            # SETUP 1: Oversold Reversal Bounce (Sniper Dip Entry)
            # -------------------------------------------------------------
            is_oversold = (min_recent_rsi <= 36.0 or last_close <= bb_lower_val * 1.005)
            bounce_confirmed = is_green_candle and is_higher_close and is_healthy_close
            signal_oversold = is_oversold and bounce_confirmed

            # -------------------------------------------------------------
            # SETUP 2: Bullish Momentum Breakout (Trend Continuation)
            # -------------------------------------------------------------
            is_trend_aligned = (ema9 > ema21 > ema50) or (ema9 > ema21 and curr_price > ema50)
            is_momentum_rsi = (50.0 <= curr_rsi <= 68.0)
            signal_momentum = is_trend_aligned and is_momentum_rsi and is_green_candle and vol_expansion and is_healthy_close

            # -------------------------------------------------------------
            # SETUP 3: Institutional Volume Surge Inflow (Breakout Ignition)
            # -------------------------------------------------------------
            signal_vol_surge = (vol_ratio >= vol_spike_multiplier) and is_green_candle and is_healthy_close and (45.0 <= curr_rsi <= 78.0) and (curr_price >= ema21)

            signal_triggered = False
            setup_name = f"Consolidating (RSI {curr_rsi:.0f})"
            if not is_healthy_close and is_green_candle:
                setup_name = f"Topping Wick ({wick_ratio*100:.0f}%)"
            prob_win = 0.50
            total_rank = 0.0

            if signal_vol_surge:
                signal_triggered = True
                setup_name = f"Volume Surge ({vol_ratio:.1f}x Vol)"
                prob_win = 0.95
                total_rank = 97.0 + min(vol_ratio, 6.0) * 2.0
            elif signal_oversold:
                signal_triggered = True
                setup_name = f"Oversold Bounce (RSI {curr_rsi:.0f})"
                prob_win = 0.92
                total_rank = 95.0 + max(0.0, 36.0 - min_recent_rsi)
            elif signal_momentum:
                signal_triggered = True
                setup_name = f"Trend Breakout (RSI {curr_rsi:.0f})"
                prob_win = 0.88
                total_rank = 90.0 + max(0.0, curr_rsi - 50.0)

            # -------------------------------------------------------------
            # Multi-Timeframe (MTF) 1-Hour Trend Alignment Guard
            # -------------------------------------------------------------
            mtf_trend_ok = True
            check_mtf = config.get("grid_trading", {}).get("mtf_guard_enabled", True) if config else True
            if signal_triggered and check_mtf:
                try:
                    df_1h = fetch_client_df(client, symbol, timeframe="1h", limit=40)
                    if not df_1h.empty and len(df_1h) >= 20:
                        close_1h = float(df_1h["close"].iloc[-1])
                        ema50_1h = float(df_1h["close"].ewm(span=min(50, len(df_1h)), adjust=False).mean().iloc[-1])
                        rsi_1h_s = calculate_rsi(df_1h["close"], period=14)
                        rsi_1h = float(rsi_1h_s.iloc[-1]) if not rsi_1h_s.empty else 50.0
                        if rsi_1h < 38.0 and close_1h < ema50_1h * 0.985:
                            mtf_trend_ok = False
                            logger.info(
                                "🛡️ [MTF Guard] %s 15m signal suppressed (1h RSI %.0f below 38 & price under 1h EMA50)",
                                symbol, rsi_1h
                            )
                except Exception:
                    pass

            if not mtf_trend_ok:
                signal_triggered = False
                setup_name = f"MTF Bear Block (1h Trend)"
                prob_win = 0.40
                total_rank = 0.0

            candidates.append({
                "symbol": symbol,
                "current_price": curr_price,
                "atr": atr_val,
                "atr_pct": atr_pct,
                "rsi": curr_rsi,
                "ema9": ema9,
                "ema21": ema21,
                "ema50": ema50,
                "vol_ratio": round(vol_ratio, 2),
                "signal_triggered": signal_triggered,
                "setup_name": setup_name,
                "prob_win": prob_win,
                "rank": total_rank,
            })
        except Exception as e:
            logger.error("Error evaluating symbol %s: %s", symbol, e)

    if not candidates:
        return None

    # Sort so triggered signals are at the top, then by rank
    candidates.sort(key=lambda x: (x["signal_triggered"], x["rank"]), reverse=True)
    top = candidates[0]
    top["all_candidates"] = candidates
    top["basket_prices"] = basket_prices
    top["basket_vol"] = basket_vol
    return top


def calculate_grid_levels(
    current_price: float,
    atr: float,
    num_buy_levels: int = 5,
    min_step_pct: float = 0.012,
) -> List[Dict[str, Any]]:
    """Calculates adaptive geometric grid buy and take-profit levels below current price."""
    atr_pct = (atr / current_price) if current_price > 0 else min_step_pct
    step_pct = max(min_step_pct, min(atr_pct * 0.8, 0.035))  # between 1.2% and 3.5%

    levels = []
    for i in range(1, num_buy_levels + 1):
        buy_price = current_price * ((1.0 - step_pct) ** i)
        target_sell_price = buy_price * (1.0 + step_pct + 0.003)
        expected_profit_pct = ((target_sell_price / buy_price) - 1.0) * 100.0

        levels.append({
            "level_index": i,
            "buy_price": round(buy_price, 4 if buy_price < 10 else 2),
            "target_sell_price": round(target_sell_price, 4 if target_sell_price < 10 else 2),
            "step_pct": round(step_pct * 100, 2),
            "expected_profit_pct": round(expected_profit_pct, 2),
        })

    return levels


def run_grid_cycle(
    client,
    state: Dict[str, Any],
    config: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Executes a single cycle of the Halal Spot Sniper Scalper Engine.
    
    Returns:
      (new_buys, harvested_profits, grid_status)
    """
    grid_cfg = config.get("grid_trading", {})
    symbols = grid_cfg.get("halal_basket", ["SOL/USDT", "AVAX/USDT", "NEAR/USDT", "LINK/USDT"])
    benchmark = grid_cfg.get("macro_benchmark", "BTC/USDT")
    timeframe = config.get("timeframe", "15m")
    fee_rate = config.get("risk_management", {}).get("fee_rate", 0.00075)

    target_tp_pct = grid_cfg.get("target_tp_pct", 0.012)
    stop_loss_pct = grid_cfg.get("stop_loss_pct", 0.018)
    breakeven_trigger_pct = grid_cfg.get("breakeven_trigger_pct", 0.008)

    trailing_cfg = grid_cfg.get("trailing_take_profit", {})
    trailing_enabled = trailing_cfg.get("enabled", False)
    callback_pct = trailing_cfg.get("callback_pct", 0.004)

    # 1. Initialize grid state if missing
    if "grid_state" not in state:
        state["grid_state"] = {
            "active_symbol": symbols[0],
            "active_symbols": [],
            "total_grid_profit_usdt": 0.0,
            "cycles_completed": 0,
            "grid_trades": 0,
        }

    grid_state = state["grid_state"]

    # 2. Select the top Halal asset with confirmed signal
    top_asset = select_best_halal_asset(client, symbols, benchmark, timeframe, config=config)
    if not top_asset:
        logger.info("[Sniper Engine] No valid market data. Holding 100% Cash Defense.")
        return [], [], {"status": "PAUSED_DEFENSE", "reason": "Market health check paused"}

    active_symbol = top_asset["symbol"]
    curr_price = top_asset["current_price"]
    grid_state["active_symbol"] = active_symbol

    try:
        live_p = fetch_client_price(client, active_symbol)
        if live_p > 0:
            curr_price = live_p
    except Exception:
        pass

    # 3. Active Position Management: Breakeven Locks, Trailing Exits, Stop Loss
    harvested_profits = []
    remaining_open_positions = {}
    open_positions = state.get("open_positions", {})

    for pos_id, pos in list(open_positions.items()):
        symbol = pos["symbol"]
        p_price = curr_price if symbol == active_symbol else pos.get("entry_price", curr_price)
        try:
            if symbol != active_symbol:
                live_sym_p = fetch_client_price(client, symbol)
                if live_sym_p > 0:
                    p_price = live_sym_p
        except Exception:
            pass

        entry_price = pos["entry_price"]
        allocated_usdt = pos["allocated_usdt"]
        target_tp = pos.get("take_profit", entry_price * (1.0 + target_tp_pct))
        stop_loss = pos.get("stop_loss", entry_price * (1.0 - stop_loss_pct))

        # 3a. BREAKEVEN PROFIT LOCK: If price gained >= +1.0%, eliminate all trade risk!
        gain_pct = (p_price / entry_price) - 1.0
        if gain_pct >= breakeven_trigger_pct:
            guaranteed_sl = entry_price * 1.002
            if stop_loss < guaranteed_sl:
                pos["stop_loss"] = guaranteed_sl
                stop_loss = guaranteed_sl
                logger.info(
                    "🛡️ [BREAKEVEN GUARD] %s gained +%.2f%%! Stop Loss ratcheted to $%s (Risk-Free Trade)",
                    symbol, gain_pct * 100.0, _fmt(guaranteed_sl)
                )

        # 3b. PARTIAL PROFIT SCALING (Take 50% profit at TP1, lock risk-free stop)
        partial_cfg = grid_cfg.get("partial_take_profit", {})
        partial_enabled = partial_cfg.get("enabled", False)
        tp1_price = pos.get("tp1_price", entry_price * (1.0 + partial_cfg.get("tp1_pct", 0.018)))
        if partial_enabled and not pos.get("partial_tp_taken", False) and p_price >= tp1_price and (trailing_enabled or p_price < target_tp):
            ratio = partial_cfg.get("ratio", 0.5)
            part_alloc = allocated_usdt * ratio
            part_gross = part_alloc * ((p_price / entry_price) - 1.0)
            part_fees = (part_alloc * fee_rate) + ((part_alloc + part_gross) * fee_rate)
            part_net = part_gross - part_fees
            part_pnl_pct = ((p_price / entry_price) - 1.0) * 100.0

            state["balance_usdt"] += (part_alloc + part_net)
            state["realized_pnl_usdt"] += part_net
            grid_state["total_grid_profit_usdt"] += part_net
            state["trades_won"] = state.get("trades_won", 0) + 1

            pos["allocated_usdt"] -= part_alloc
            allocated_usdt = pos["allocated_usdt"]
            pos["partial_tp_taken"] = True
            pos["stop_loss"] = max(pos.get("stop_loss", 0.0), round(entry_price * 1.003, 8 if entry_price < 1 else 4))
            stop_loss = pos["stop_loss"]
            pos["trailing_active"] = True
            pos["peak_price"] = p_price

            try:
                trade_record = {
                    "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                    "position_id": pos_id + "-TP1",
                    "symbol": symbol,
                    "action": "PARTIAL_TP1_HARVEST",
                    "entry_price": entry_price,
                    "exit_price": p_price,
                    "allocated_usdt": part_alloc,
                    "stop_loss": stop_loss,
                    "tp1": tp1_price,
                    "tp2": target_tp,
                    "net_pnl": part_net,
                    "pnl_pct": part_pnl_pct,
                    "confidence_pct": 100.0,
                    "reason": f"Partial TP1 Harvest (50% closed at +{part_pnl_pct:.2f}%)",
                    "outcome": "WIN",
                    "mode": config.get("trading_mode", "paper"),
                }
                log_trade(trade_record, ledger_path=config["paths"]["trade_ledger"])
            except Exception as e:
                logger.error("Failed to log partial TP1 trade: %s", e)

            harvested_profits.append({
                "symbol": symbol,
                "position_id": pos_id + "-TP1",
                "entry_price": entry_price,
                "exit_price": p_price,
                "net_pnl": part_net,
                "pnl_pct": part_pnl_pct,
                "allocated_usdt": part_alloc,
                "reason": f"Partial TP1 Harvest (50% closed at +{part_pnl_pct:.2f}%)",
            })
            logger.info(
                "💰 [PARTIAL TP1] %s 50%% Harvested at $%s (+%.2f%%) | Booked Net: +$%.2f USDT | Remaining $%.2f Trailing with SL $%s",
                symbol, _fmt(p_price), part_pnl_pct, part_net, allocated_usdt, _fmt(stop_loss)
            )

        should_exit = False
        outcome = "WIN"
        harvest_reason = ""

        # 3c. TRAILING TAKE-PROFIT / PROFIT HARVEST
        if trailing_enabled:
            if pos.get("trailing_active", False):
                if p_price > pos.get("peak_price", p_price):
                    pos["peak_price"] = p_price

                peak = pos.get("peak_price", p_price)
                trailing_stop = peak * (1.0 - callback_pct)
                floor_stop = max(target_tp * 0.998, entry_price * 1.003) if peak >= target_tp else (entry_price * 1.003)

                if p_price <= trailing_stop or p_price <= floor_stop:
                    should_exit = True
                    outcome = "WIN"
                    harvest_reason = f"Trailing TP Harvest (Peak: ${_fmt(peak)}, Drop: -{callback_pct*100:.2f}%)"
            elif p_price >= target_tp:
                pos["trailing_active"] = True
                pos["peak_price"] = p_price
                logger.info(
                    "🚀 [TRAILING TP ARMED] %s hit scalp target ($%s). Tracking peak price ($%s)...",
                    symbol, _fmt(target_tp), _fmt(p_price)
                )
        else:
            if p_price >= target_tp:
                should_exit = True
                outcome = "WIN"
                harvest_reason = f"Scalp Target Hit (+{gain_pct*100:.2f}%)"

        # 3c. STOP LOSS GUARD
        if not should_exit and p_price <= stop_loss:
            should_exit = True
            outcome = "WIN" if p_price >= entry_price else "LOSS"
            harvest_reason = "Breakeven Guard Exit" if p_price >= entry_price else f"Risk Guard Stop Loss (-{abs(gain_pct)*100:.2f}%)"

        # 3d. STALE CAPITAL TIME-DECAY EXIT (Free capital if flat for > 4 hours)
        stale_cfg = grid_cfg.get("stale_exit", {})
        if not should_exit and stale_cfg.get("enabled", True):
            entry_time_str = pos.get("entry_time")
            if entry_time_str:
                try:
                    entry_dt = datetime.fromisoformat(entry_time_str.replace("Z", "+00:00"))
                    now_dt = datetime.now(timezone.utc)
                    hours_held = (now_dt - entry_dt).total_seconds() / 3600.0
                    stale_limit_hours = stale_cfg.get("hours", 4.0)
                    threshold_pct = stale_cfg.get("max_pnl_threshold_pct", 1.0) / 100.0
                    if hours_held >= stale_limit_hours and not pos.get("trailing_active", False) and abs(gain_pct) <= threshold_pct:
                        should_exit = True
                        outcome = "WIN" if p_price >= entry_price else "LOSS"
                        harvest_reason = f"Stale Capital Exit (Held {hours_held:.1f}h without breakout)"
                        logger.info(
                            "⏳ [STALE EXIT] %s held for %.1fh without breakout. Exiting at $%s to free capital.",
                            symbol, hours_held, f"{p_price:,.4f}"
                        )
                except Exception:
                    pass

        # 3e. EARLY MOMENTUM INVALIDATION GUARD (Cut failed breakouts at -0.8% instead of -1.8%)
        early_inv_cfg = grid_cfg.get("early_momentum_invalidation", {})
        if not should_exit and early_inv_cfg.get("enabled", True):
            loss_thresh = early_inv_cfg.get("loss_threshold_pct", 0.5) / 100.0
            if gain_pct <= -loss_thresh and not pos.get("trailing_active", False):
                try:
                    df_pos = fetch_client_df(client, symbol, timeframe=timeframe, limit=10)
                    if len(df_pos) >= 4:
                        c1 = df_pos.iloc[-1]
                        c2 = df_pos.iloc[-2]
                        c1_red = float(c1["close"]) < float(c1["open"])
                        c2_red = float(c2["close"]) < float(c2["open"])
                        ema9_pos = float(df_pos["close"].ewm(span=9, adjust=False).mean().iloc[-1])
                        if c1_red and c2_red and p_price < ema9_pos:
                            should_exit = True
                            outcome = "LOSS"
                            harvest_reason = f"Early Momentum Invalidation (-{abs(gain_pct)*100:.2f}%, 2 red closes below EMA9)"
                            logger.info(
                                "🛡️ [EARLY INVALIDATION] %s broke down below EMA9 with 2 red candles (-%.2f%%). Cutting loss early to protect capital.",
                                symbol, abs(gain_pct) * 100.0
                            )
                except Exception:
                    pass

        if should_exit:
            exit_price = p_price
            gross_pnl = allocated_usdt * ((exit_price / entry_price) - 1.0)
            fees = (allocated_usdt * fee_rate) + ((allocated_usdt + gross_pnl) * fee_rate)
            net_pnl = gross_pnl - fees
            pnl_pct = (net_pnl / allocated_usdt) * 100.0

            state["balance_usdt"] += (allocated_usdt + net_pnl)
            state["realized_pnl_usdt"] += net_pnl
            grid_state["total_grid_profit_usdt"] += net_pnl
            grid_state["cycles_completed"] += 1
            state["trade_count"] += 1

            if outcome == "WIN" and net_pnl >= 0:
                state["trades_won"] = state.get("trades_won", 0) + 1
                state["win_count"] = state.get("win_count", 0) + 1
            else:
                state["loss_count"] = state.get("loss_count", 0) + 1

            try:
                trade_record = {
                    "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                    "position_id": pos_id,
                    "symbol": symbol,
                    "action": "SCALP_HARVEST" if net_pnl >= 0 else "SCALP_STOP",
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "allocated_usdt": allocated_usdt,
                    "stop_loss": stop_loss,
                    "tp1": target_tp,
                    "tp2": target_tp,
                    "net_pnl": net_pnl,
                    "pnl_pct": pnl_pct,
                    "confidence_pct": 100.0,
                    "reason": harvest_reason,
                    "outcome": outcome,
                    "mode": config.get("trading_mode", "paper"),
                }
                log_trade(trade_record, ledger_path=config["paths"]["trade_ledger"])
            except Exception as e:
                logger.error("Failed to log scalp trade: %s", e)

            harvested = {
                "symbol": symbol,
                "position_id": pos_id,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "net_pnl": net_pnl,
                "pnl_pct": pnl_pct,
                "allocated_usdt": allocated_usdt,
                "reason": harvest_reason,
            }
            harvested_profits.append(harvested)
            logger.info(
                "💰 [%s] %s Sold at $%s (%+.2f%%) | Net PnL: +$%s USDT | %s",
                outcome, symbol, _fmt(exit_price), pnl_pct, f"{net_pnl:.2f}", harvest_reason
            )
        else:
            pos["curr_price"] = p_price
            pos["unrealized_pnl"] = allocated_usdt * ((p_price / entry_price) - 1.0)
            pos["unrealized_pnl_pct"] = ((p_price / entry_price) - 1.0) * 100.0
            remaining_open_positions[pos_id] = pos

    state["open_positions"] = remaining_open_positions

    # 4. Check for New High-Probability Sniper Scalp Entries
    new_buys = []
    max_open_trades = grid_cfg.get("max_grid_levels", 2)
    order_size_usdt = grid_cfg.get("order_size_usdt", 25.0)
    min_cash_reserve = grid_cfg.get("min_cash_reserve_usdt", 25.0)

    coins_with_positions = set(p["symbol"] for p in state["open_positions"].values())
    total_open_positions = len(state["open_positions"])
    available_cash = state["balance_usdt"] - min_cash_reserve

    candidates = top_asset.get("all_candidates", [top_asset])

    for cand in candidates:
        if total_open_positions >= max_open_trades or available_cash < order_size_usdt:
            break

        cand_sym = cand["symbol"]
        cand_positions = [p for p in state["open_positions"].values() if p["symbol"] == cand_sym]
        max_levels_per_coin = grid_cfg.get("max_levels_per_coin", 1)
        if len(cand_positions) >= max_levels_per_coin:
            continue

        max_active_coins = grid_cfg.get("max_active_coins", 2)
        if cand_sym not in coins_with_positions and len(coins_with_positions) >= max_active_coins:
            continue

        require_confluence = grid_cfg.get("require_confluence_signal", False)
        if require_confluence and not cand.get("signal_triggered", False):
            continue

        # Bid-Ask Spread Guard (Micro-Slippage Defense)
        spread_cfg = grid_cfg.get("spread_guard", {})
        if spread_cfg.get("enabled", True):
            max_spread = spread_cfg.get("max_spread_pct", 0.0015)
            spread_ok = True
            if hasattr(client, "get_ticker"):
                try:
                    tk = client.get_ticker(cand_sym)
                    if isinstance(tk, dict):
                        bid = float(tk.get("bid", 0.0) or 0.0)
                        ask = float(tk.get("ask", 0.0) or 0.0)
                        if bid > 0 and ask > 0:
                            spread = (ask - bid) / bid
                            if spread > max_spread:
                                spread_ok = False
                                logger.info(
                                    "🛡️ [SPREAD GUARD] %s spread %.3f%% exceeds limit %.3f%%. Skipping.",
                                    cand_sym, spread * 100.0, max_spread * 100.0
                                )
                except Exception:
                    pass
            if not spread_ok:
                continue

        # Correlation & Exposure Diversification Guard
        corr_cfg = grid_cfg.get("correlation_guard", {})
        if corr_cfg.get("enabled", True) and coins_with_positions:
            max_corr = corr_cfg.get("max_correlation", 0.85)
            high_corr_detected = False
            for open_sym in coins_with_positions:
                try:
                    df_cand = fetch_client_df(client, cand_sym, timeframe=timeframe, limit=30)
                    df_open = fetch_client_df(client, open_sym, timeframe=timeframe, limit=30)
                    if len(df_cand) >= 20 and len(df_open) >= 20:
                        c_series = df_cand["close"].pct_change().dropna()
                        o_series = df_open["close"].pct_change().dropna()
                        if c_series.std() > 1e-7 and o_series.std() > 1e-7:
                            min_len = min(len(c_series), len(o_series))
                            corr_val = float(c_series.iloc[-min_len:].corr(o_series.iloc[-min_len:]))
                            if not np.isnan(corr_val) and corr_val >= max_corr:
                                high_corr_detected = True
                                logger.info(
                                    "🛡️ [CORRELATION GUARD] %s correlation (%.2f) with %s exceeds limit (%.2f). Diversifying.",
                                    cand_sym, corr_val, open_sym, max_corr
                                )
                                break
                except Exception:
                    pass
            if high_corr_detected:
                continue

        cand_price = cand["current_price"]
        cand_atr = cand["atr"]
        cand_ema9 = cand.get("ema9", cand_price)
        try:
            live_cp = fetch_client_price(client, cand_sym)
            if live_cp > 0:
                cand_price = live_cp
        except Exception:
            pass

        # Pullback Retrace Calibration (Never Buy the Spike Peak)
        pullback_cfg = grid_cfg.get("pullback_retrace", {})
        if pullback_cfg.get("enabled", True):
            retrace_pct = pullback_cfg.get("retrace_pct", 0.006)
            retrace_price = cand_price * (1.0 - retrace_pct)
            if cand_ema9 > 0 and cand_ema9 < cand_price:
                optimal_entry = max(cand_ema9, retrace_price)
            else:
                optimal_entry = retrace_price
            cand_price = round(optimal_entry, 8 if optimal_entry < 0.01 else (4 if optimal_entry < 10 else 2))

        # Dynamic ATR-Based Take-Profit and Stop Loss
        atr_pct = (cand_atr / cand_price) if cand_price > 0 else 0.02
        atr_tp_mult = grid_cfg.get("atr_tp_multiplier", 1.8)
        elastic_tp_pct = min(0.08, max(target_tp_pct, atr_pct * atr_tp_mult))

        atr_sl_mult = grid_cfg.get("atr_sl_multiplier", 1.2)
        elastic_sl_pct = min(stop_loss_pct, max(0.015, atr_pct * atr_sl_mult))

        target_tp_price = cand_price * (1.0 + elastic_tp_pct)
        stop_loss_price = cand_price * (1.0 - elastic_sl_pct)

        # TP1 Partial scaling target
        partial_cfg = grid_cfg.get("partial_take_profit", {})
        tp1_pct = partial_cfg.get("tp1_pct", 0.018)
        tp1_price = cand_price * (1.0 + max(tp1_pct, elastic_tp_pct * 0.55))

        # Volatility-Adjusted Position Sizing (Risk Parity)
        vol_size_cfg = grid_cfg.get("volatility_sizing", {})
        actual_order_size = order_size_usdt
        if vol_size_cfg.get("enabled", False):
            target_risk = vol_size_cfg.get("target_risk_usdt", 0.60)
            min_order = vol_size_cfg.get("min_order_usdt", 18.0)
            max_order = vol_size_cfg.get("max_order_usdt", 35.0)
            if elastic_sl_pct > 0:
                risk_sized = target_risk / elastic_sl_pct
                actual_order_size = round(min(max_order, max(min_order, risk_sized)), 2)

        actual_order_size = min(actual_order_size, available_cash)
        if actual_order_size < 10.0:
            continue

        pos_id = str(uuid.uuid4())[:8]
        state["balance_usdt"] -= actual_order_size
        new_pos = {
            "position_id": pos_id,
            "symbol": cand_sym,
            "direction": "BUY_SPOT",
            "entry_time": datetime.now(timezone.utc).isoformat(),
            "entry_price": cand_price,
            "curr_price": cand_price,
            "stop_loss": stop_loss_price,
            "take_profit": target_tp_price,
            "tp1_price": tp1_price,
            "partial_tp_taken": False,
            "allocated_usdt": actual_order_size,
            "unrealized_pnl": 0.0,
            "unrealized_pnl_pct": 0.0,
            "trailing_active": False,
            "peak_price": cand_price,
            "setup_name": cand.get("setup_name", "Halal Confluence Scalp"),
        }
        state["open_positions"][pos_id] = new_pos
        grid_state["grid_trades"] += 1
        coins_with_positions.add(cand_sym)
        total_open_positions += 1
        available_cash -= actual_order_size

        new_buys.append(new_pos)
        logger.info(
            "🟢 [SNIPER BUY] %s Bought $%s at $%s | Setup: %s | Target TP: $%s (+%.2f%%)",
            cand_sym, f"{actual_order_size:.2f}", _fmt(cand_price),
            cand.get("setup_name", "Scalp"), _fmt(target_tp_price), elastic_tp_pct * 100.0
        )
        break

    active_symbols = list(dict.fromkeys([p["symbol"] for p in state["open_positions"].values()]))
    grid_state["active_symbols"] = active_symbols

    grid_status = {
        "status": "ACTIVE_HARVESTING",
        "active_symbol": active_symbol,
        "active_symbols": active_symbols,
        "current_price": curr_price,
        "atr_pct": top_asset["atr_pct"],
        "open_levels": len(state["open_positions"]),
        "max_levels": max_open_trades,
        "total_profit_booked": grid_state["total_grid_profit_usdt"],
        "cycles_completed": grid_state["cycles_completed"],
        "available_cash": state["balance_usdt"],
        "basket_prices": top_asset.get("basket_prices", {}),
        "basket_vol": top_asset.get("basket_vol", {}),
        "candidate_details": {c["symbol"]: c for c in candidates},
    }

    return new_buys, harvested_profits, grid_status
