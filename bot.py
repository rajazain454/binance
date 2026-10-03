"""Binance AI Quantitative Trading Bot.
Loads the trained machine learning model, monitors live Binance candle closes,
predicts trade outcome probabilities, manages dynamic ATR risk/reward brackets,
and executes trades in simulated paper-trading or live Binance spot mode.
"""

import os
import sys
import json
import time
import argparse
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import features
from binance_client import BinanceClient
from dotenv import load_dotenv

load_dotenv()


def load_config(config_path="config.json"):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_state(state_path="bot_state.json", initial_capital=10000.0):
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
                        return st
        except Exception as e:
            print(f"[Warning] Corrupted state file '{state_path}' ({e}). Rebuilding clean default state.")
    return {
        "balance_usdt": initial_capital,
        "peak_balance": initial_capital,
        "daily_peak_balance": initial_capital,
        "daily_reset_time": datetime.now(timezone.utc).isoformat(),
        "circuit_breaker_until": None,
        "open_positions": {},
        "trade_count": 0,
        "win_count": 0,
        "loss_count": 0,
        "last_update": None
    }


def save_state(state, state_path="bot_state.json"):
    """Atomic write to prevent state corruption on sudden termination."""
    tmp_path = f"{state_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp_path, state_path)


def send_discord_alert(webhook_url, title, fields, color=65280, description=""):
    if not webhook_url or not webhook_url.startswith("https://discord.com/api/webhooks/"):
        return False, "Invalid or unconfigured Discord Webhook URL."
    payload = {
        "username": "QuantLab Binance AI",
        "avatar_url": "https://bin.bnbstatic.com/static/images/common/favicon.ico",
        "embeds": [
            {
                "title": title,
                "description": description,
                "color": color,
                "fields": fields,
                "footer": {"text": "Binance QuantLab Execution Engine"},
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        ]
    }
    data_encoded = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        webhook_url,
        data=data_encoded,
        headers={"Content-Type": "application/json", "User-Agent": "QuantLabBot/1.0"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return (resp.status in [200, 204]), "Dispatched to Discord"
    except urllib.error.HTTPError as e:
        try:
            err_data = json.loads(e.read().decode("utf-8"))
            if err_data.get("code") == 10015:
                return False, "Discord Error 10015 (Unknown Webhook): This webhook URL does not exist or was deleted on Discord."
            return False, f"Discord API Error {e.code}: {err_data.get('message', str(e))}"
        except Exception:
            return False, f"HTTP Error {e.code}: {e.reason}"
    except Exception as e:
        return False, str(e)


def log_trade(trade_record, ledger_path="logs/trade_history.csv"):
    os.makedirs(os.path.dirname(ledger_path), exist_ok=True)
    exists = os.path.exists(ledger_path) and os.path.getsize(ledger_path) > 0
    df = pd.DataFrame([trade_record])
    df.to_csv(ledger_path, mode="a", header=not exists, index=False)


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
        if (now - last_reset).total_seconds() >= 86400:
            state["daily_peak_balance"] = curr_equity
            state["daily_reset_time"] = now.isoformat()

    daily_peak = state.get("daily_peak_balance", curr_equity)
    if curr_equity > daily_peak:
        state["daily_peak_balance"] = curr_equity
        daily_peak = curr_equity

    dd_pct = ((daily_peak - curr_equity) / daily_peak) * 100.0 if daily_peak > 0 else 0.0
    if dd_pct >= max_dd_pct:
        from datetime import timedelta
        lock_until = now + timedelta(hours=cb_hours)
        state["circuit_breaker_until"] = lock_until.isoformat()
        return True, f"TRIGGERED (-{dd_pct:.1f}% daily DD >= {max_dd_pct}%)"

    return False, f"NORMAL (Daily DD: -{dd_pct:.1f}% / -{max_dd_pct}%)"


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
        except Exception:
            continue

        entry_price = pos["entry_price"]
        sl = pos["stop_loss"]
        tp1 = pos.get("take_profit_1", pos.get("take_profit", entry_price * 1.02))
        tp2 = pos.get("take_profit_2", entry_price * 1.04)
        units = pos["units"]
        allocated = pos["allocated_usdt"]

        # Update live mark-to-market metrics
        pos["curr_price"] = curr_price
        gross_unrealized = units * (curr_price - entry_price)
        pos["unrealized_pnl"] = round(gross_unrealized, 2)
        pos["unrealized_pnl_pct"] = round(((curr_price / entry_price) - 1.0) * 100.0, 2)

        # Check maximum holding time decay
        is_stale = False
        entry_time_str = pos.get("entry_time")
        if entry_time_str and max_hold_hours > 0:
            try:
                entry_dt = datetime.fromisoformat(entry_time_str.replace(" ", "T")).replace(tzinfo=timezone.utc)
                if (now_utc - entry_dt).total_seconds() >= max_hold_hours * 3600:
                    is_stale = True
            except Exception:
                pass

        # Stage 1: Check TP1 Partial Exit (50%) & Lock Stop to Breakeven
        if not pos.get("tp1_reached", False) and curr_price >= tp1:
            sell_ratio = config.get("ai_model", {}).get("partial_tp_ratio", 0.50)
            sell_units = units * sell_ratio
            sell_allocated = allocated * sell_ratio

            # Small account guard: If 50% split is below Binance minNotional ($5.00),
            # don't split order into rejected dust. Ratchet Stop Loss to Breakeven to make entire trade risk-free!
            if sell_allocated < 5.0:
                pos["stop_loss"] = entry_price * 1.001
                pos["tp1_reached"] = True
                record = {
                    "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                    "symbol": sym,
                    "action": "BREAKEVEN LOCK (SMALL ACCOUNT)",
                    "entry_price": round(entry_price, 4),
                    "exit_price": round(curr_price, 4),
                    "allocated_usdt": round(allocated, 2),
                    "net_pnl": 0.0,
                    "pnl_pct": 0.0,
                    "reason": f"TP1 reached! Size (${sell_allocated:.2f}) < $5 minNotional. Stop Loss locked to BREAKEVEN for 100% position!",
                    "outcome": "WIN"
                }
                log_trade(record, config["paths"]["trade_ledger"])
                closed_signals.append(record)
                continue

            # Real live execution if live mode enabled
            if config.get("trading_mode") == "live":
                try:
                    client.place_spot_order(sym, "sell", sell_units, price=curr_price)
                    print(f"  [LIVE SPOT ORDER] Successfully executed 50% TP1 sell on {sym} ({sell_units})")
                except Exception as e:
                    print(f"  [LIVE ORDER ERROR] Failed to execute TP1 sell for {sym}: {e}")

            gross_pnl = sell_units * (curr_price - entry_price)
            fee = sell_allocated * cost_rate * 2.0
            net_pnl = gross_pnl - fee

            state["balance_usdt"] += (sell_allocated + net_pnl)
            pos["units"] -= sell_units
            pos["allocated_usdt"] -= sell_allocated
            pos["tp1_reached"] = True

            # Breakeven lock: ratchet Stop Loss to Entry Price + buffer
            if config.get("ai_model", {}).get("breakeven_lock_enabled", True):
                pos["stop_loss"] = entry_price * 1.001

            record = {
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                "symbol": sym,
                "action": "PARTIAL TP1 (50% SOLD)",
                "entry_price": round(entry_price, 4),
                "exit_price": round(curr_price, 4),
                "allocated_usdt": round(sell_allocated, 2),
                "net_pnl": round(net_pnl, 2),
                "pnl_pct": round((net_pnl / sell_allocated) * 100, 2),
                "reason": "TP1 Reached (50% profit booked). Stop Loss locked to BREAKEVEN!",
                "outcome": "WIN"
            }
            log_trade(record, config["paths"]["trade_ledger"])
            closed_signals.append(record)
            continue

        # Stage 2: Check Final TP2, Stop Loss, or Max Holding Time Exit
        hit_tp2 = curr_price >= tp2
        hit_sl = curr_price <= sl

        if hit_tp2 or hit_sl or is_stale:
            # Real live execution if live mode enabled
            if config.get("trading_mode") == "live":
                try:
                    client.place_spot_order(sym, "sell", units, price=curr_price)
                    print(f"  [LIVE SPOT ORDER] Successfully executed final close on {sym} ({units})")
                except Exception as e:
                    print(f"  [LIVE ORDER ERROR] Failed to execute final close for {sym}: {e}")

            if hit_tp2:
                exit_reason = "TP2 FINAL TARGET REACHED"
            elif is_stale and not hit_sl:
                exit_reason = f"MAX HOLDING TIME EXPIRED ({max_hold_hours}h)"
            elif pos.get("tp1_reached"):
                exit_reason = "BREAKEVEN EXIT (RISK-FREE RUNNER)"
            else:
                exit_reason = "STOP LOSS HIT"

            is_win = hit_tp2 or (curr_price >= entry_price)
            gross_pnl = units * (curr_price - entry_price)
            fee = allocated * cost_rate * 2.0
            net_pnl = gross_pnl - fee

            state["balance_usdt"] += (allocated + net_pnl)
            state["trade_count"] += 1
            if is_win:
                state["win_count"] += 1
            else:
                state["loss_count"] += 1

            record = {
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                "symbol": sym,
                "action": "FINAL CLOSE",
                "entry_price": round(entry_price, 4),
                "exit_price": round(curr_price, 4),
                "allocated_usdt": round(allocated, 2),
                "net_pnl": round(net_pnl, 2),
                "pnl_pct": round((net_pnl / allocated) * 100, 2),
                "reason": exit_reason,
                "outcome": "WIN" if is_win else "LOSS"
            }
            log_trade(record, config["paths"]["trade_ledger"])
            closed_signals.append(record)
            symbols_to_remove.append(sym)

    for s in symbols_to_remove:
        del state["open_positions"][s]

    return closed_signals


def get_macro_regime_threshold(client, config):
    """
    Computes dynamic confidence threshold based on macro BTC 200 SMA regime.
    - Macro Bull (BTC > 200 SMA): 55% threshold (captures trending breakouts)
    - Macro Bear / Chop (BTC < 200 SMA): 65% threshold (ultra-defensive capital preservation)
    """
    base_thresh = config.get("ai_model", {}).get("confidence_threshold", 0.60)
    try:
        btc_df = client.fetch_ohlcv("BTC/USDT", timeframe="1d", limit=220)
        if len(btc_df) >= 200:
            c = btc_df["close"]
            sma200 = float(c.rolling(200).mean().iloc[-1])
            curr_c = float(c.iloc[-1])
            if curr_c > sma200:
                return round(base_thresh - 0.05, 2), "MACRO BULL (55% Thresh)"
            else:
                return round(base_thresh + 0.05, 2), "MACRO DEFENSIVE (65% Thresh)"
    except Exception:
        pass
    return base_thresh, f"NORMAL ({base_thresh:.0%})"


def send_daily_digest(state, config):
    """Dispatches a comprehensive 24h portfolio performance digest embed to Discord."""
    discord_cfg = config.get("discord", {})
    if not discord_cfg.get("enabled", False):
        return False, "Discord alerts disabled."
    webhook = os.getenv("DISCORD_WEBHOOK_URL", discord_cfg.get("webhook_url", "")).strip()
    if not webhook:
        return False, "Discord webhook URL not configured."

    balance = state.get("balance_usdt", config["risk_management"]["capital_usdt"])
    init_cap = config["risk_management"]["capital_usdt"]
    open_pos = state.get("open_positions", {})

    total_unrealized = sum(p.get("unrealized_pnl", 0.0) for p in open_pos.values())
    total_alloc = sum(p.get("allocated_usdt", 0.0) for p in open_pos.values())
    total_equity = balance + total_alloc + total_unrealized
    total_ret = ((total_equity / init_cap) - 1.0) * 100.0

    total_trades = state.get("trade_count", 0)
    wins = state.get("win_count", 0)
    wr = (wins / total_trades * 100) if total_trades > 0 else 0.0

    fields = [
        {"name": "Total Equity (MtM)", "value": f"**${total_equity:,.2f} USDT** ({total_ret:+.2f}%)", "inline": True},
        {"name": "Free Cash (USDT)", "value": f"${balance:,.2f}", "inline": True},
        {"name": "Open Unrealized PnL", "value": f"**${total_unrealized:+,.2f} USDT**", "inline": True},
        {"name": "Closed Trades", "value": f"{total_trades} (Wins: {wins})", "inline": True},
        {"name": "Win Rate", "value": f"**{wr:.1f}%**", "inline": True},
        {"name": "Active Holdings", "value": f"{len(open_pos)} / {config['risk_management']['max_open_trades']} Slots", "inline": True},
        {"name": "Trading Mode", "value": config.get("trading_mode", "paper").upper(), "inline": True},
        {"name": "Timeframe", "value": config.get("timeframe", "1h"), "inline": True}
    ]
    if open_pos:
        desc = "Current active positions:\n" + "\n".join([
            f"• **{sym}**: Size ${pos['allocated_usdt']:,.2f} | Entry: ${pos['entry_price']:.4f} | "
            f"Now: ${pos.get('curr_price', pos['entry_price']):.4f} | Unr PnL: ${pos.get('unrealized_pnl', 0.0):+,.2f} ({pos.get('unrealized_pnl_pct', 0.0):+.2f}%)"
            for sym, pos in open_pos.items()
        ])
    else:
        desc = "Portfolio in 100% Cash Defense (No active open risk)."

    ok, res = send_discord_alert(webhook, "📊 DAILY PORTFOLIO PERFORMANCE DIGEST", fields, color=3447003, description=desc)
    return ok, res


def scan_and_execute(client, model_bundle, state, config):
    """Evaluates Binance market candles, runs model inference, and executes trades."""
    model = model_bundle["model"]
    feat_names = model_bundle["feature_names"]
    conf_thresh, regime_label = get_macro_regime_threshold(client, config)
    tp1_mult = config["ai_model"].get("tp1_atr_mult", 1.2)
    tp2_mult = config["ai_model"].get("tp2_atr_mult", 2.4)
    sl_mult = config["ai_model"].get("sl_atr_mult", 1.4)
    max_trades = config["risk_management"].get("max_open_trades", 3)
    risk_pct = config["risk_management"].get("risk_per_trade_pct", 2.0) / 100.0
    tf = config["timeframe"]

    cb_locked, cb_status = check_circuit_breaker(state, config)

    analysis_rows = []
    new_signals = []

    for sym in config["symbols"]:
        try:
            df = client.fetch_ohlcv(sym, timeframe=tf, limit=300)
            if len(df) < 50:
                continue

            feat_df = features.extract_features(df)
            atr_series = features.compute_atr(df, 14)

            last_feats = feat_df.iloc[-1:][feat_names]
            prob_win = float(model.predict_proba(last_feats)[0, 1])

            c = float(df["close"].iloc[-1])
            curr_atr = float(atr_series.iloc[-1])
            sl = c - (sl_mult * curr_atr)
            tp1 = c + (tp1_mult * curr_atr)
            tp2 = c + (tp2_mult * curr_atr)

            # MTF Macro Confluence Check (4h Trend Alignment)
            mtf_cfg = config.get("mtf_confluence", {})
            use_mtf = mtf_cfg.get("enabled", True)
            min_rsi = mtf_cfg.get("macro_rsi_min", 48.0)
            is_macro_bullish, mtf_info = features.evaluate_macro_confluence(df, min_rsi=min_rsi)

            # Funding rate sentiment check (avoid entering if market is over-leveraged long)
            fr = client.fetch_funding_rate(sym)
            funding_overheated = (fr > 0.0003)

            is_high_conviction = prob_win >= conf_thresh
            is_open = sym in state["open_positions"]
            confluence_passed = is_macro_bullish if use_mtf else True

            if is_open:
                status = "HOLDING"
            elif cb_locked:
                status = "CIRCUIT PAUSE"
            elif funding_overheated:
                status = "FUNDING HOT"
            elif is_high_conviction and confluence_passed:
                status = "BUY TRIGGER"
            elif is_high_conviction and not confluence_passed:
                status = "4H BLOCKED"
            else:
                status = "SCANNING"

            row = {
                "symbol": sym,
                "price": c,
                "atr": curr_atr,
                "prob_win": prob_win,
                "high_conviction": is_high_conviction,
                "mtf_status": mtf_info.get("status", "N/A"),
                "status": status,
                "stop_loss": sl,
                "tp1": tp1,
                "tp2": tp2
            }
            analysis_rows.append(row)

            # Execution condition: 1h conviction + 4h macro alignment + circuit breaker clear
            if not cb_locked and is_high_conviction and confluence_passed and not is_open and len(state["open_positions"]) < max_trades:
                # Risk Parity position sizing with Dynamic Alpha Kelly Scaling
                available_balance = state["balance_usdt"]
                dynamic_sizing = config.get("risk_management", {}).get("dynamic_alpha_sizing", True)
                if dynamic_sizing:
                    surplus = max(-0.10, min(0.20, prob_win - conf_thresh))
                    alpha_mult = 1.0 + (surplus * 2.0)
                    eff_risk_pct = risk_pct * alpha_mult
                else:
                    eff_risk_pct = risk_pct

                risk_amount = available_balance * eff_risk_pct
                risk_per_unit = c - sl

                if risk_per_unit > 0:
                    units = risk_amount / risk_per_unit
                    slot_cap = (available_balance / (max_trades - len(state["open_positions"]))) * 0.95
                    if dynamic_sizing:
                        max_alloc = slot_cap * min(1.25, max(0.80, alpha_mult))
                    else:
                        max_alloc = slot_cap
                    pos_cost = min(available_balance * 0.98, max_alloc, units * c)
                    units = pos_cost / c

                    if pos_cost >= 5.0:  # Minimum Binance order threshold ($5.00 USDT)
                        # Real live execution if live mode enabled
                        if config.get("trading_mode") == "live":
                            try:
                                order_res = client.place_spot_order(sym, "buy", units, price=c)
                                print(f"  [LIVE SPOT ORDER] Successfully executed buy for {sym}: {order_res.get('id', 'FILLED')}")
                            except Exception as e:
                                print(f"  [LIVE ORDER REJECTED] Binance order failed for {sym}: {e}")
                                continue

                        state["balance_usdt"] -= pos_cost
                        state["open_positions"][sym] = {
                            "entry_price": c,
                            "allocated_usdt": pos_cost,
                            "units": units,
                            "stop_loss": sl,
                            "take_profit_1": tp1,
                            "take_profit_2": tp2,
                            "tp1_reached": False,
                            "entry_time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                            "confidence": round(prob_win * 100, 1),
                            "macro_4h": mtf_info.get("status", "BULLISH")
                        }

                        trade_signal = {
                            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                            "symbol": sym,
                            "action": "BUY (ENTER)",
                            "price": round(c, 4),
                            "size_usdt": round(pos_cost, 2),
                            "stop_loss": round(sl, 4),
                            "tp1": round(tp1, 4),
                            "tp2": round(tp2, 4),
                            "confidence_pct": round(prob_win * 100, 1),
                            "macro_4h": mtf_info.get("status", "BULLISH"),
                            "mode": config.get("trading_mode", "paper").upper()
                        }
                        log_trade(trade_signal, config["paths"]["trade_ledger"])
                        new_signals.append(trade_signal)
        except Exception as e:
            print(f"Error analyzing {sym}: {e}")

    return analysis_rows, new_signals, cb_status


def print_dashboard(analysis_rows, new_signals, closed_signals, state, config, cb_status):
    os.system("cls" if os.name == "nt" else "clear")
    balance = state["balance_usdt"]
    open_pos = state.get("open_positions", {})
    mode = config.get("trading_mode", "paper").upper()
    use_mtf = config.get("mtf_confluence", {}).get("enabled", True)

    total_unrealized = sum(p.get("unrealized_pnl", 0.0) for p in open_pos.values())
    total_alloc = sum(p.get("allocated_usdt", 0.0) for p in open_pos.values())
    total_equity = balance + total_alloc + total_unrealized

    print("=" * 104)
    print(f"  BINANCE QUANTITATIVE AI TRADING BOT  |  MODE: {mode}  |  TF: {config['timeframe']} (4H MTF: {'ON' if use_mtf else 'OFF'})")
    print("=" * 104)
    win_cnt = state.get("win_count", 0)
    total_cnt = state.get("trade_count", 0)
    wr = (win_cnt / total_cnt * 100) if total_cnt > 0 else 0.0

    print(f"  Equity (MtM): ${total_equity:,.2f} USDT  |  Free Cash: ${balance:,.2f}  |  Active Alloc: ${total_alloc:,.2f}  |  Unrealized: ${total_unrealized:+,.2f}")
    print(f"  Positions: {len(open_pos)}/{config['risk_management']['max_open_trades']}  |  Circuit Breaker: {cb_status}  |  Total Trades: {total_cnt}  |  Win Rate: {wr:.1f}%")
    print("-" * 104)

    # Market Radar Table with 4h MTF column
    print(f"{'Symbol':<10} {'Price':>10} {'AI Win Prob':>12} {'Threshold':>10} {'4h Trend':>10} {'Status':<16} {'ATR(14)':>9}")
    print("-" * 104)
    for r in analysis_rows:
        conv_tag = "[BUY TRIGGER]" if r["status"] == "BUY TRIGGER" else (r["status"])
        print(f"{r['symbol']:<10} {r['price']:>10.4f} {r['prob_win']:>11.1%} {config['ai_model']['confidence_threshold']:>9.0%} "
              f"{r['mtf_status']:>10} {conv_tag:<16} {r['atr']:>9.4f}")

    # Active Holdings
    print("\n" + "-" * 104)
    print("  ACTIVE OPEN POSITIONS & LIVE MARK-TO-MARKET PERFORMANCE")
    print("-" * 104)
    if not open_pos:
        print("  No active positions. Scanning Binance market for high-probability setups...")
    else:
        for sym, p in open_pos.items():
            entry_p = p['entry_price']
            curr_p = p.get('curr_price', entry_p)
            unr_pnl = p.get('unrealized_pnl', 0.0)
            unr_pct = p.get('unrealized_pnl_pct', 0.0)
            tp1_val = p.get('take_profit_1') or p.get('take_profit') or (entry_p * 1.015)
            tp2_val = p.get('take_profit_2') or (tp1_val * 1.015)
            tp1_status = "LOCKED [BREAKEVEN]" if p.get("tp1_reached") else f"${tp1_val:.4f}"
            print(f"  [HOLD] {sym:<9} Entry: ${entry_p:<9.4f} | Now: ${curr_p:<9.4f} | Unr: ${unr_pnl:+7.2f} ({unr_pct:+6.2f}%) | "
                  f"Stop: ${p['stop_loss']:<9.4f} | TP1: {tp1_status:<17} | TP2: ${tp2_val:<9.4f}")

    # Closed signals
    if closed_signals:
        print("\n" + "-" * 88)
        print("  RECENT CLOSED TRADES")
        print("-" * 88)
        for cs in closed_signals:
            tag = "[+WIN]" if cs["outcome"] == "WIN" else "[-LOSS]"
            print(f"  {tag} {cs['symbol']} Net PnL: ${cs['net_pnl']:+,.2f} ({cs['pnl_pct']:+.2f}%) | {cs['reason']}")

    # New entries
    if new_signals:
        print("\n" + "-" * 88)
        print("  NEW EXECUTED ORDERS")
        print("-" * 88)
        for ns in new_signals:
            print(f"  [NEW ORDER] {ns['action']} {ns['symbol']} @ ${ns['price']:.4f} | Size: ${ns['size_usdt']} USDT | Conf: {ns['confidence_pct']}%")

    print("=" * 88 + "\n")


def execute_cycle(client, model_bundle, config):
    state_path = config["paths"]["state_file"]
    state = load_state(state_path, initial_capital=config["risk_management"]["capital_usdt"])

    # 1. Update existing positions
    closed_signals = check_and_update_positions(client, state, config)

    # 2. Scan market and execute new orders
    analysis_rows, new_signals, cb_status = scan_and_execute(client, model_bundle, state, config)

    # 3. Print CLI dashboard
    print_dashboard(analysis_rows, new_signals, closed_signals, state, config, cb_status)

    # 4. Discord alerts
    discord_cfg = config.get("discord", {})
    webhook = os.getenv("DISCORD_WEBHOOK_URL", discord_cfg.get("webhook_url", "")).strip()
    if discord_cfg.get("enabled", False) and webhook:
        for s in new_signals:
            entry_p = s['price']
            tp2_p = s.get('tp2', entry_p * 1.02)
            sl_p = s.get('stop_loss', entry_p * 0.98)
            est_win_pct = ((tp2_p / entry_p) - 1.0) * 100
            est_loss_pct = ((sl_p / entry_p) - 1.0) * 100

            fields = [
                {"name": "🪙 Coin", "value": f"**{s['symbol']}**", "inline": True},
                {"name": "💵 Amount Bought", "value": f"**${s['size_usdt']:.2f} USDT**", "inline": True},
                {"name": "📈 Buy Price", "value": f"${entry_p:,.2f}", "inline": True},
                {"name": "🎯 Target Price (Take Profit)", "value": f"**${tp2_p:,.2f}** ({est_win_pct:+.2f}%)", "inline": True},
                {"name": "🛡️ Safety Price (Stop Loss)", "value": f"${sl_p:,.2f} ({est_loss_pct:+.2f}%)", "inline": True},
                {"name": "🤖 AI Confidence", "value": f"**{s['confidence_pct']:.0f}%**", "inline": True},
                {"name": "📌 Next Steps", "value": "Sit back! The bot is watching the market and will exit automatically.", "inline": False}
            ]
            title = f"🟢 BOUGHT {s['symbol']} (${s['size_usdt']:.2f})"
            desc = f"The AI detected a high-probability buying opportunity on **{s['symbol']}**."
            send_discord_alert(webhook, title, fields, color=3066993, description=desc)

        for cs in closed_signals:
            is_win = cs["outcome"] == "WIN"
            color = 5763719 if is_win else 15548997
            title = f"🎉 WON TRADE: {cs['symbol']}" if is_win else f"🛡️ TRADE CLOSED: {cs['symbol']}"
            pnl_val = cs['net_pnl']
            pnl_pct = cs['pnl_pct']

            fields = [
                {"name": "🪙 Coin", "value": f"**{cs['symbol']}**", "inline": True},
                {"name": "💰 Profit / Loss", "value": f"**${pnl_val:+,.2f} USDT** ({pnl_pct:+.2f}%)", "inline": True},
                {"name": "🏁 Sold At", "value": f"${cs['exit_price']:,.2f}", "inline": True},
                {"name": "ℹ️ Reason", "value": cs["reason"], "inline": False}
            ]
            send_discord_alert(webhook, title, fields, color=color)

    # Save state
    state["last_update"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    save_state(state, state_path)


def main():
    parser = argparse.ArgumentParser(description="Binance AI Quantitative Trading Bot")
    parser.add_argument("--config", default="config.json", help="Path to config file")
    parser.add_argument("--model", default="models/binance_ai_model.joblib", help="Path to model bundle")
    parser.add_argument("--scan", action="store_true", help="Run single scan and exit")
    parser.add_argument("--loop", action="store_true", help="Run continuously on candle close loop")
    parser.add_argument("--capital", type=float, default=None, help="Override starting capital")
    parser.add_argument("--test-discord", action="store_true", help="Send a test embed to Discord")
    parser.add_argument("--digest", action="store_true", help="Send 24h portfolio digest embed to Discord")

    args = parser.parse_args()
    config = load_config(args.config)

    if args.digest:
        state = load_state(config["paths"]["state_file"], initial_capital=config["risk_management"]["capital_usdt"])
        ok, res = send_daily_digest(state, config)
        print(f"Daily digest dispatch result: {res}")
        if not args.scan and not args.loop:
            return

    if args.test_discord:
        d_cfg = config.get("discord", {})
        webhook = os.getenv("DISCORD_WEBHOOK_URL", d_cfg.get("webhook_url", "")).strip()
        fields = [
            {"name": "Status", "value": "Online & Operational", "inline": True},
            {"name": "Trading Mode", "value": config.get("trading_mode", "paper").upper(), "inline": True},
            {"name": "Exchange", "value": "Binance Spot", "inline": True},
            {"name": "Timeframe", "value": config.get("timeframe", "1h"), "inline": True}
        ]
        ok, res = send_discord_alert(webhook, "🤖 QuantLab Binance AI Bot Test Alert", fields, color=3447003)
        print(f"Discord test result: {res}")
        if not args.scan and not args.loop:
            return

    if args.capital:
        config["risk_management"]["capital_usdt"] = args.capital

    if not os.path.exists(args.model):
        sys.exit(f"Model file '{args.model}' not found. Please run 'python train.py' first.")

    model_bundle = joblib.load(args.model)
    client = BinanceClient(config)

    execute_cycle(client, model_bundle, config)

    if args.loop:
        tf_sec = 3600 if config["timeframe"] == "1h" else (14400 if config["timeframe"] == "4h" else 86400)
        print(f"[Loop Active] Monitoring Binance on {config['timeframe']} candles...")
        last_digest_date = datetime.now(timezone.utc).date()
        while True:
            now_dt = datetime.now(timezone.utc)
            if now_dt.date() > last_digest_date and now_dt.hour == 0:
                try:
                    st = load_state(config["paths"]["state_file"])
                    send_daily_digest(st, config)
                    last_digest_date = now_dt.date()
                except Exception:
                    pass
            now = time.time()
            sleep_time = tf_sec - (now % tf_sec) + 10
            time.sleep(sleep_time)
            try:
                execute_cycle(client, model_bundle, config)
            except Exception as e:
                print(f"Cycle execution error: {e}")


if __name__ == "__main__":
    main()
