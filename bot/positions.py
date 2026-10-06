"""Position monitoring, dynamic take-profit, trailing stops, and exits."""

import logging
from datetime import datetime, timezone, timedelta
from bot.ledger import log_trade

logger = logging.getLogger("quantlab")


def render_trade_progress(curr_p, entry_p, tp1, sl, length=10):
    """
    Renders visual ASCII progress gauge indicating progress towards TP1 or SL.
    Example: `[██████░░░░] 60% to TP1 ($793.38)`
    """
    if tp1 <= entry_p or sl >= entry_p:
        return f"TP1: ${tp1:,.2f} | SL: ${sl:,.2f}"

    if curr_p >= entry_p:
        tp_dist = tp1 - entry_p
        gain = curr_p - entry_p
        pct = min(1.0, max(0.0, gain / max(1e-9, tp_dist)))
        filled = int(round(pct * length))
        bar = "█" * filled + "░" * (length - filled)
        pct_label = int(round(pct * 100))
        return f"`[{bar}]` **{pct_label}% to TP1** (`${tp1:,.2f}`)"
    else:
        sl_dist = entry_p - sl
        loss = entry_p - curr_p
        pct = min(1.0, max(0.0, loss / max(1e-9, sl_dist)))
        filled = int(round(pct * length))
        bar = "▓" * filled + "░" * (length - filled)
        pct_label = int(round(pct * 100))
        return f"`[{bar}]` **{pct_label}% to SL** (`${sl:,.2f}`)"


def check_and_update_positions(client, state, config):
    """
    Monitors open positions for:
    1. Multi-Stage Take-Profit (TP1 50% + Breakeven Lock, TP2 100%)
    2. Dynamic Mark-to-Market Unrealized PnL calculation
    3. Stop-Loss invalidation
    4. Max-Holding-Time decay exit (closes stagnant chop trades after N hours)
    """
    open_pos = state.get("open_positions", {})
    if not open_pos:
        return []

    cost_rate = config["risk_management"]["fee_rate"] + config["risk_management"]["slippage_rate"]
    max_hold_hours = config.get("risk_management", {}).get("max_holding_hours", 48)
    now_utc = datetime.now(timezone.utc)
    closed_signals = []
    symbols_to_remove = []

    for sym, pos in open_pos.items():
        try:
            curr_price = client.get_ticker_price(sym)
        except Exception as e:
            logger.warning("Failed to fetch ticker price for %s: %s", sym, e)
            continue

        entry_price = pos["entry_price"]
        sl = pos["stop_loss"]
        tp1 = pos.get("take_profit_1", pos.get("take_profit", entry_price * 1.02))
        tp2 = pos.get("take_profit_2", entry_price * 1.04)
        units = pos["units"]
        allocated = pos["allocated_usdt"]

        # Update live mark-to-market metrics & peak price
        pos["curr_price"] = curr_price
        gross_unrealized = units * (curr_price - entry_price)
        pos["unrealized_pnl"] = round(gross_unrealized, 2)
        pos["unrealized_pnl_pct"] = round(((curr_price / entry_price) - 1.0) * 100.0, 2)
        highest_p = max(pos.get("highest_price", entry_price), curr_price)
        pos["highest_price"] = highest_p

        # Chandelier Dynamic Trailing Stop for runners (after TP1 is banked)
        trail_cfg = config.get("trailing_stop", {})
        use_chandelier = trail_cfg.get("enabled", False)
        trail_mult = trail_cfg.get("atr_mult", 2.2)

        if use_chandelier and pos.get("tp1_reached", False):
            pos_atr = pos.get("atr", (tp1 - entry_price) / config.get("ai_model", {}).get("tp1_atr_mult", 1.2))
            chandelier_stop = highest_p - (trail_mult * pos_atr)
            be_level = entry_price * 1.001
            pos["stop_loss"] = max(pos["stop_loss"], chandelier_stop, be_level)
            sl = pos["stop_loss"]

        # Check maximum holding time decay
        is_stale = False
        entry_time_str = pos.get("entry_time")
        if entry_time_str and max_hold_hours > 0:
            try:
                entry_dt = datetime.fromisoformat(entry_time_str.replace(" ", "T"))
                if entry_dt.tzinfo is None:
                    entry_dt = entry_dt.replace(tzinfo=timezone.utc)
                if (now_utc - entry_dt).total_seconds() >= max_hold_hours * 3600:
                    is_stale = True
            except Exception:
                pass

        # Stage 1: Check TP1 Partial Exit (50%) & Lock Stop to Breakeven / Profit Floor
        if not pos.get("tp1_reached", False) and curr_price >= tp1:
            sell_ratio = config.get("ai_model", {}).get("partial_tp_ratio", 0.50)
            sell_units = units * sell_ratio
            sell_allocated = allocated * sell_ratio

            # Small account guard: If 50% split is below Binance minNotional,
            # don't split order into rejected dust. Ratchet Stop Loss to secure profit floor!
            try:
                min_notional = float(client.get_min_notional(sym))
            except Exception:
                min_notional = float(config.get("risk_management", {}).get("min_notional_usdt", 5.0))
            if sell_allocated < min_notional:
                profit_floor = entry_price * 1.001
                pos["stop_loss"] = max(pos["stop_loss"], profit_floor)
                pos["tp1_reached"] = True
                record = {
                    "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                    "symbol": sym,
                    "action": "BREAKEVEN LOCK (SMALL ACCOUNT)",
                    "entry_price": round(entry_price, 4),
                    "exit_price": round(curr_price, 4),
                    "allocated_usdt": round(allocated, 2),
                    "stop_loss": round(sl, 4),
                    "tp1": round(tp1, 4),
                    "tp2": round(tp2, 4),
                    "net_pnl": 0.0,
                    "pnl_pct": 0.0,
                    "confidence_pct": pos.get("confidence", ""),
                    "reason": f"TP1 reached! Size (${sell_allocated:.2f}) < ${min_notional:.1f} minNotional. Stop Loss locked to BREAKEVEN for 100% position!",
                    "outcome": "WIN",
                    "mode": config.get("trading_mode", "paper").upper(),
                }
                log_trade(record, config["paths"]["trade_ledger"])
                closed_signals.append(record)
                continue

            # Real live execution if live mode enabled
            if config.get("trading_mode") == "live":
                try:
                    client.place_spot_order(sym, "sell", sell_units, price=curr_price)
                    logger.info("[LIVE SPOT ORDER] Successfully executed 50%% TP1 sell on %s (%s)", sym, sell_units)
                except Exception as e:
                    logger.error("[LIVE ORDER ERROR] Failed to execute TP1 sell for %s: %s", sym, e)

            gross_pnl = sell_units * (curr_price - entry_price)
            fee = sell_allocated * cost_rate * 2.0
            net_pnl = gross_pnl - fee

            state["balance_usdt"] += (sell_allocated + net_pnl)
            pos["units"] -= sell_units
            pos["allocated_usdt"] -= sell_allocated
            pos["tp1_reached"] = True

            # Count partial TP1 exits as real closed trades
            state["trade_count"] += 1
            state["win_count"] += 1

            # Breakeven / Profit lock: ratchet Stop Loss to secure profit
            if config.get("ai_model", {}).get("breakeven_lock_enabled", True):
                pos["stop_loss"] = entry_price * 1.001

            record = {
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                "symbol": sym,
                "action": "PARTIAL TP1 (50% SOLD)",
                "entry_price": round(entry_price, 4),
                "exit_price": round(curr_price, 4),
                "allocated_usdt": round(sell_allocated, 2),
                "stop_loss": round(sl, 4),
                "tp1": round(tp1, 4),
                "tp2": round(tp2, 4),
                "net_pnl": round(net_pnl, 2),
                "pnl_pct": round((net_pnl / sell_allocated) * 100, 2),
                "confidence_pct": pos.get("confidence", ""),
                "reason": "TP1 Reached (50% profit booked). Stop Loss locked to BREAKEVEN!",
                "outcome": "WIN",
                "mode": config.get("trading_mode", "paper").upper(),
            }
            log_trade(record, config["paths"]["trade_ledger"])
            closed_signals.append(record)
            continue

        # Stage 2: Check Final TP2, Stop Loss, or Max Holding Time Exit
        exit_at_tp2 = config.get("ai_model", {}).get("exit_at_tp2", False)
        hit_tp2 = (curr_price >= tp2) and (exit_at_tp2 or not use_chandelier)
        hit_sl = curr_price <= sl

        if hit_tp2 or hit_sl or is_stale:
            # Real live execution if live mode enabled
            if config.get("trading_mode") == "live":
                try:
                    client.place_spot_order(sym, "sell", units, price=curr_price)
                    logger.info("[LIVE SPOT ORDER] Successfully executed final close on %s (%s)", sym, units)
                except Exception as e:
                    logger.error("[LIVE ORDER ERROR] Failed to execute final close for %s: %s", sym, e)

            if hit_tp2:
                exit_reason = "TP2 FINAL TARGET REACHED"
            elif is_stale and not hit_sl:
                exit_reason = f"MAX HOLDING TIME EXPIRED ({max_hold_hours}h)"
            elif pos.get("tp1_reached"):
                exit_reason = f"CHANDELIER TRAILING EXIT (SL Ratcheted to ${sl:.4f} | Peak: ${highest_p:.4f})"
            else:
                exit_reason = "STOP LOSS HIT"

            is_win = hit_tp2 or (curr_price >= entry_price) or pos.get("tp1_reached", False)
            gross_pnl = units * (curr_price - entry_price)
            fee = allocated * cost_rate * 2.0
            net_pnl = gross_pnl - fee

            state["balance_usdt"] += (allocated + net_pnl)
            state["trade_count"] += 1
            if is_win:
                state["win_count"] += 1
            else:
                state["loss_count"] += 1
                cooldown_hours = config.get("risk_management", {}).get("loss_cooldown_hours", 12)
                if cooldown_hours > 0:
                    cooldown_expiry = (now_utc + timedelta(hours=cooldown_hours)).isoformat()
                    state.setdefault("loss_cooldowns", {})[sym] = cooldown_expiry

            record = {
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                "symbol": sym,
                "action": "FINAL CLOSE",
                "entry_price": round(entry_price, 4),
                "exit_price": round(curr_price, 4),
                "allocated_usdt": round(allocated, 2),
                "stop_loss": round(sl, 4),
                "tp1": round(tp1, 4),
                "tp2": round(tp2, 4),
                "net_pnl": round(net_pnl, 2),
                "pnl_pct": round((net_pnl / allocated) * 100, 2),
                "confidence_pct": pos.get("confidence", ""),
                "reason": exit_reason,
                "outcome": "WIN" if is_win else "LOSS",
                "mode": config.get("trading_mode", "paper").upper(),
            }
            log_trade(record, config["paths"]["trade_ledger"])
            closed_signals.append(record)
            symbols_to_remove.append(sym)

    for s in symbols_to_remove:
        del state["open_positions"][s]

    return closed_signals
