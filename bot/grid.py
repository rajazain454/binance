"""Adaptive Halal Spot Volatility Grid Engine.

Operates strictly on Shariah-compliant Spot crypto utility assets (SOL, AVAX, NEAR, LINK).
No leverage, no margin, no short-selling, no interest (riba).
Harvests daily market oscillations into locked-in USDT profits.
"""

from datetime import datetime, timezone
import logging
import math
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from bot.ledger import log_trade

logger = logging.getLogger("quantlab")


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


def fetch_client_df(client, symbol: str, timeframe: str = "1h", limit: int = 120) -> pd.DataFrame:
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


def select_best_halal_asset(
    client,
    symbols: List[str],
    macro_benchmark: str = "BTC/USDT",
    timeframe: str = "1h",
) -> Optional[Dict[str, Any]]:
    """Evaluates the Halal asset basket and selects the healthiest, most volatile pair.
    
    Returns a dict with:
      symbol, current_price, atr, atr_pct, trend_score, daily_volatility
    or None if all pairs fail the health check.
    """
    candidates = []
    basket_prices = {}
    basket_vol = {}
    
    # 1. Check macro benchmark regime if requested
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

    for symbol in symbols:
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

            # Calculate ATR (14 bars)
            atr_s = calculate_atr(df, period=14)
            atr_val = float(atr_s.iloc[-1]) if not atr_s.empty and not pd.isna(atr_s.iloc[-1]) else curr_price * 0.015
            atr_pct = (atr_val / curr_price) * 100.0
            basket_vol[symbol] = atr_pct

            # Trend Check: 50 EMA vs 20 EMA on current timeframe
            ema20 = float(df["close"].ewm(span=20, adjust=False).mean().iloc[-1])
            ema50 = float(df["close"].ewm(span=50, adjust=False).mean().iloc[-1])
            is_healthy_trend = curr_price >= ema50 * 0.97  # Not in a severe freefall

            # 24h Volume and Volatility Score
            vol_24h = float(df["volume"].iloc[-24:].sum() * curr_price)
            # Volatility sweet spot: 1.0% to 5.0% ATR per 1h bar is ideal for grid trading
            vol_score = min(atr_pct, 5.0)

            # Trend health bonus
            trend_score = 1.0 if curr_price >= ema20 else (0.5 if is_healthy_trend else 0.0)

            total_rank = (vol_score * 0.6) + (trend_score * 0.4)

            if is_healthy_trend and vol_24h > 100_000:
                candidates.append({
                    "symbol": symbol,
                    "current_price": curr_price,
                    "atr": atr_val,
                    "atr_pct": atr_pct,
                    "ema20": ema20,
                    "ema50": ema50,
                    "vol_score": vol_score,
                    "rank": total_rank,
                })
        except Exception as e:
            logger.error("Error evaluating symbol %s: %s", symbol, e)

    if not candidates:
        return None

    candidates.sort(key=lambda x: x["rank"], reverse=True)
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
    """Calculates adaptive geometric grid buy and take-profit levels below current price.
    
    Each level represents an entry price and a corresponding target sell price.
    """
    atr_pct = (atr / current_price) if current_price > 0 else min_step_pct
    step_pct = max(min_step_pct, min(atr_pct * 0.8, 0.035))  # between 1.2% and 3.5%

    levels = []
    for i in range(1, num_buy_levels + 1):
        buy_price = current_price * ((1.0 - step_pct) ** i)
        # Sell target is at least 1 step above the buy price + safety margin for fees (0.15% round-trip)
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
    """Executes a single cycle of the Halal Spot Volatility Grid.
    
    Returns:
      (new_buys, harvested_profits, grid_status)
    """
    grid_cfg = config.get("grid_trading", {})
    symbols = grid_cfg.get("halal_basket", ["SOL/USDT", "AVAX/USDT", "NEAR/USDT", "LINK/USDT"])
    benchmark = grid_cfg.get("macro_benchmark", "BTC/USDT")
    timeframe = config.get("timeframe", "1h")
    fee_rate = config.get("risk_management", {}).get("fee_rate", 0.00075)

    # 1. Initialize grid state if missing
    if "grid_state" not in state:
        state["grid_state"] = {
            "active_symbol": symbols[0],
            "active_levels": {},
            "total_grid_profit_usdt": 0.0,
            "cycles_completed": 0,
            "grid_trades": 0,
        }

    grid_state = state["grid_state"]

    # 2. Select the top Halal asset
    top_asset = select_best_halal_asset(client, symbols, benchmark, timeframe)
    if not top_asset:
        logger.info("[Grid Engine] No suitable asset found passing health checks. Holding cash.")
        return [], [], {"status": "PAUSED_DEFENSE", "reason": "Market health check paused buys"}

    active_symbol = top_asset["symbol"]
    curr_price = top_asset["current_price"]
    atr = top_asset["atr"]
    grid_state["active_symbol"] = active_symbol

    # Fetch live price if possible
    try:
        live_p = fetch_client_price(client, active_symbol)
        if live_p > 0:
            curr_price = live_p
    except Exception:
        pass

    # 3. Check existing open grid inventory for take-profit exits
    harvested_profits = []
    remaining_open_positions = {}
    trailing_cfg = grid_cfg.get("trailing_take_profit", {})
    trailing_enabled = trailing_cfg.get("enabled", False)
    callback_pct = trailing_cfg.get("callback_pct", 0.005)
    
    # We inspect state["open_positions"]
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

        target_tp = pos.get("take_profit", pos["entry_price"] * 1.015)
        entry_price = pos["entry_price"]
        allocated_usdt = pos["allocated_usdt"]

        should_harvest = False
        harvest_reason = ""

        if trailing_enabled:
            if pos.get("trailing_active", False):
                if p_price > pos.get("peak_price", p_price):
                    pos["peak_price"] = p_price

                peak = pos.get("peak_price", p_price)
                trailing_stop = peak * (1.0 - callback_pct)
                floor_stop = max(target_tp * 0.998, entry_price * 1.003)

                if p_price <= trailing_stop or p_price <= floor_stop:
                    should_harvest = True
                    harvest_reason = f"Trailing TP Harvest (Peak: ${peak:,.4f}, Drop: -{callback_pct*100:.2f}%)"
            elif p_price >= target_tp:
                # Arm trailing take profit
                pos["trailing_active"] = True
                pos["peak_price"] = p_price
                logger.info(
                    "🚀 [TRAILING TP ARMED] %s hit target TP ($%s). Tracking peak price (curr: $%s, callback: %.2f%%)...",
                    symbol, f"{target_tp:,.4f}", f"{p_price:,.4f}", callback_pct * 100.0
                )
        else:
            if p_price >= target_tp:
                should_harvest = True
                harvest_reason = f"Grid Level TP Hit (+{((p_price / entry_price) - 1.0) * 100.0:.2f}%)"

        if should_harvest:
            # HARVEST PROFIT!
            exit_price = p_price
            gross_pnl = allocated_usdt * ((exit_price / entry_price) - 1.0)
            fees = (allocated_usdt * fee_rate) + ((allocated_usdt + gross_pnl) * fee_rate)
            net_pnl = gross_pnl - fees
            pnl_pct = (net_pnl / allocated_usdt) * 100.0

            # Update balances
            state["balance_usdt"] += (allocated_usdt + net_pnl)
            state["realized_pnl_usdt"] += net_pnl
            grid_state["total_grid_profit_usdt"] += net_pnl
            grid_state["cycles_completed"] += 1
            state["trades_won"] += 1
            state["trade_count"] += 1

            # Log to trade ledger
            try:
                trade_record = {
                    "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                    "position_id": pos_id,
                    "symbol": symbol,
                    "action": "GRID_HARVEST",
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "allocated_usdt": allocated_usdt,
                    "stop_loss": entry_price * 0.95,
                    "tp1": target_tp,
                    "tp2": target_tp,
                    "net_pnl": net_pnl,
                    "pnl_pct": pnl_pct,
                    "confidence_pct": 100.0,
                    "reason": harvest_reason,
                    "outcome": "WIN",
                    "mode": config.get("trading_mode", "paper"),
                }
                log_trade(trade_record, ledger_path=config["paths"]["trade_ledger"])
            except Exception as e:
                logger.error("Failed to log grid trade: %s", e)

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
                "💰 [GRID HARVEST] %s Sold at $%s (+%.2f%%) | Net Profit: +$%s USDT | %s",
                symbol, f"{exit_price:,.4f}", pnl_pct, f"{net_pnl:.2f}", harvest_reason
            )
        else:
            # Position stays open and tracks unrealized PnL
            pos["curr_price"] = p_price
            pos["unrealized_pnl"] = allocated_usdt * ((p_price / entry_price) - 1.0)
            pos["unrealized_pnl_pct"] = ((p_price / entry_price) - 1.0) * 100.0
            remaining_open_positions[pos_id] = pos

    state["open_positions"] = remaining_open_positions

    # 4. Check if we should execute a new grid buy level
    new_buys = []
    max_open_grid_levels = grid_cfg.get("max_grid_levels", 5)
    max_active_coins = grid_cfg.get("max_active_coins", 2)
    max_levels_per_coin = grid_cfg.get("max_levels_per_coin", 3)
    order_size_usdt = grid_cfg.get("order_size_usdt", 12.0)
    min_cash_reserve = grid_cfg.get("min_cash_reserve_usdt", 15.0)

    coins_with_positions = set(p["symbol"] for p in state["open_positions"].values())
    total_open_positions = len(state["open_positions"])
    available_cash = state["balance_usdt"] - min_cash_reserve

    candidates = top_asset.get("all_candidates", [top_asset])

    for cand in candidates:
        if total_open_positions >= max_open_grid_levels or available_cash < order_size_usdt:
            break

        cand_sym = cand["symbol"]
        cand_positions = [p for p in state["open_positions"].values() if p["symbol"] == cand_sym]

        # Respect max active coin diversification limit
        if cand_sym not in coins_with_positions and len(coins_with_positions) >= max_active_coins:
            continue

        # Respect max levels per coin limit
        if len(cand_positions) >= max_levels_per_coin:
            continue

        cand_price = cand["current_price"]
        cand_atr = cand["atr"]
        try:
            live_cp = fetch_client_price(client, cand_sym)
            if live_cp > 0:
                cand_price = live_cp
        except Exception:
            pass

        should_enter = False
        target_tp_price = cand_price * (1.0 + max(0.012, (cand_atr / cand_price) * 0.9))

        if not cand_positions:
            should_enter = True
            entry_price = cand_price
        else:
            existing_entry_prices = [p["entry_price"] for p in cand_positions]
            lowest_entry = min(existing_entry_prices)
            drop_from_lowest = ((cand_price / lowest_entry) - 1.0) * 100.0
            step_threshold = -max(1.2, (cand_atr / cand_price) * 75.0)

            if drop_from_lowest <= step_threshold:
                should_enter = True
                entry_price = cand_price
                target_tp_price = entry_price * (1.0 + max(0.012, (cand_atr / cand_price) * 0.9))

        if should_enter:
            pos_id = str(uuid.uuid4())[:8]
            state["balance_usdt"] -= order_size_usdt
            new_pos = {
                "position_id": pos_id,
                "symbol": cand_sym,
                "direction": "BUY_SPOT",
                "entry_time": datetime.now(timezone.utc).isoformat(),
                "entry_price": entry_price,
                "curr_price": entry_price,
                "stop_loss": entry_price * 0.95,
                "take_profit": target_tp_price,
                "allocated_usdt": order_size_usdt,
                "unrealized_pnl": 0.0,
                "unrealized_pnl_pct": 0.0,
                "trailing_active": False,
                "peak_price": entry_price,
            }
            state["open_positions"][pos_id] = new_pos
            grid_state["grid_trades"] += 1
            coins_with_positions.add(cand_sym)
            total_open_positions += 1
            available_cash -= order_size_usdt

            new_buys.append(new_pos)
            logger.info(
                "🟢 [GRID BUY] %s Bought $%s at $%s | Target TP: $%s",
                cand_sym, f"{order_size_usdt:.2f}", f"{entry_price:,.4f}", f"{target_tp_price:,.4f}"
            )
            break

    active_symbols = list(dict.fromkeys([p["symbol"] for p in state["open_positions"].values()]))
    if not active_symbols:
        active_symbols = [active_symbol]
    grid_state["active_symbols"] = active_symbols

    grid_status = {
        "status": "ACTIVE_HARVESTING",
        "active_symbol": active_symbol,
        "active_symbols": active_symbols,
        "current_price": curr_price,
        "atr_pct": top_asset["atr_pct"],
        "open_levels": len(state["open_positions"]),
        "max_levels": max_open_grid_levels,
        "total_profit_booked": grid_state["total_grid_profit_usdt"],
        "cycles_completed": grid_state["cycles_completed"],
        "available_cash": state["balance_usdt"],
        "basket_prices": top_asset.get("basket_prices", {}),
        "basket_vol": top_asset.get("basket_vol", {}),
    }

    return new_buys, harvested_profits, grid_status
