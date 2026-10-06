"""State management and live exchange reconciliation for QuantLab Trading Bot."""

import json
import logging
import os
from datetime import datetime, timezone, timedelta

logger = logging.getLogger("quantlab")


def load_state(state_path="bot_state.json", initial_capital=10000.0):
    """Loads bot state JSON file or initializes a clean state dictionary."""
    if os.path.exists(state_path):
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    st = json.loads(content)
                    if isinstance(st, dict):
                        if "daily_peak_balance" not in st:
                            st["daily_peak_balance"] = st.get("balance_usdt", initial_capital)
                        if "daily_reset_time" not in st:
                            st["daily_reset_time"] = datetime.now(timezone.utc).isoformat()
                        if "circuit_breaker_until" not in st:
                            st["circuit_breaker_until"] = None
                        if "open_positions" not in st or not isinstance(st["open_positions"], dict):
                            st["open_positions"] = {}
                        if "loss_cooldowns" not in st or not isinstance(st["loss_cooldowns"], dict):
                            st["loss_cooldowns"] = {}
                        else:
                            now_utc = datetime.now(timezone.utc)
                            cleaned_cooldowns = {}
                            for k, v in st["loss_cooldowns"].items():
                                try:
                                    exp_dt = datetime.fromisoformat(v)
                                    if exp_dt.tzinfo is None:
                                        exp_dt = exp_dt.replace(tzinfo=timezone.utc)
                                    if exp_dt > now_utc:
                                        cleaned_cooldowns[k] = v
                                except Exception:
                                    pass
                            st["loss_cooldowns"] = cleaned_cooldowns
                        return st
        except Exception as e:
            logger.warning("Corrupted state file '%s' (%s). Rebuilding clean default state.", state_path, e)
    return {
        "balance_usdt": initial_capital,
        "peak_balance": initial_capital,
        "daily_peak_balance": initial_capital,
        "daily_reset_time": datetime.now(timezone.utc).isoformat(),
        "circuit_breaker_until": None,
        "open_positions": {},
        "loss_cooldowns": {},
        "trade_count": 0,
        "win_count": 0,
        "loss_count": 0,
        "last_update": None,
    }


def save_state(state, state_path="bot_state.json"):
    """Atomic write to prevent state corruption on sudden termination."""
    tmp_path = f"{state_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp_path, state_path)


def get_portfolio_equity(state):
    """Calculates total mark-to-market account equity: Cash + Active Position Values."""
    cash = state.get("balance_usdt", 0.0)
    open_pos = state.get("open_positions", {})
    open_val = sum(p.get("allocated_usdt", 0.0) + p.get("unrealized_pnl", 0.0) for p in open_pos.values())
    return cash + open_val


def check_circuit_breaker(state, config):
    """Enforces 24-hour maximum drawdown circuit breaker based on Mark-to-Market total portfolio equity."""
    rm = config.get("risk_management", {})
    max_dd_pct = rm.get("daily_max_drawdown_pct", 4.0)
    cb_hours = rm.get("circuit_breaker_hours", 24)
    now = datetime.now(timezone.utc)
    curr_equity = get_portfolio_equity(state)

    # Check active lock
    cb_until_str = state.get("circuit_breaker_until")
    if cb_until_str:
        cb_until = datetime.fromisoformat(cb_until_str)
        if cb_until.tzinfo is None:
            cb_until = cb_until.replace(tzinfo=timezone.utc)
        if now < cb_until:
            remaining_mins = int((cb_until - now).total_seconds() / 60)
            return True, f"LOCKED ({remaining_mins}m remaining)"
        else:
            state["circuit_breaker_until"] = None
            state["daily_peak_balance"] = curr_equity

    # Daily reset
    last_reset_str = state.get("daily_reset_time")
    if last_reset_str:
        last_reset = datetime.fromisoformat(last_reset_str)
        if last_reset.tzinfo is None:
            last_reset = last_reset.replace(tzinfo=timezone.utc)
        if (now - last_reset).total_seconds() >= 86400:
            state["daily_peak_balance"] = curr_equity
            state["daily_reset_time"] = now.isoformat()

    daily_peak = state.get("daily_peak_balance", curr_equity)
    if curr_equity > daily_peak:
        state["daily_peak_balance"] = curr_equity
        daily_peak = curr_equity

    dd_pct = ((daily_peak - curr_equity) / daily_peak) * 100.0 if daily_peak > 0 else 0.0
    if dd_pct >= max_dd_pct:
        lock_until = now + timedelta(hours=cb_hours)
        state["circuit_breaker_until"] = lock_until.isoformat()
        return True, f"TRIGGERED (-{dd_pct:.1f}% daily DD >= {max_dd_pct}%)"

    return False, f"NORMAL (Daily DD: -{dd_pct:.1f}% / -{max_dd_pct}%)"


def reconcile_positions(client, state, config):
    """
    Reconciles internal bot_state.json with actual exchange balances in live mode.
    - Synchronizes free USDT cash balance.
    - Verifies that open positions in state are still held on Binance.
    - If a position was closed externally (or liquidated), removes it to prevent ghost positions.
    - Updates held units if partially executed.
    """
    try:
        balance_info = client.get_balance()
        if not balance_info or not isinstance(balance_info, dict):
            return

        usdt_free = float(balance_info.get("USDT", {}).get("free", state.get("balance_usdt", 0.0)))
        state["balance_usdt"] = round(usdt_free, 2)

        open_pos = state.get("open_positions", {})
        symbols_to_remove = []

        for sym, pos in open_pos.items():
            base_asset = sym.split("/")[0]
            exchange_qty = float(balance_info.get(base_asset, {}).get("total", 0.0))
            curr_p = pos.get("curr_price", pos.get("entry_price", 1.0))
            est_value = exchange_qty * curr_p

            # If asset balance on exchange is effectively zero (< $1.00 notional)
            if est_value < 1.0:
                logger.warning(
                    "[RECONCILIATION] %s position missing on exchange (holds %.4f %s, value ~$%.2f). Removing ghost position from state.",
                    sym, exchange_qty, base_asset, est_value,
                )
                symbols_to_remove.append(sym)
            else:
                # Update units if changed (e.g. partial manual fill)
                if abs(pos["units"] - exchange_qty) / max(pos["units"], 1e-9) > 0.05:
                    logger.info(
                        "[RECONCILIATION] Updating units for %s from %.6f to exchange actual %.6f",
                        sym, pos["units"], exchange_qty,
                    )
                    pos["units"] = exchange_qty

        for sym in symbols_to_remove:
            del open_pos[sym]

    except Exception as e:
        logger.warning("Live position reconciliation failed: %s", e)
