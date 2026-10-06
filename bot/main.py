"""Main execution loop and CLI entrypoint for QuantLab Trading Bot."""

import argparse
from datetime import datetime, timezone
import json
import logging
import os
import sys
import time
import joblib

from binance_client import BinanceClient
from bot.alerts import send_daily_digest, send_discord_alert
from bot.dashboard import print_dashboard
from bot.positions import check_and_update_positions, render_trade_progress
from bot.scanner import scan_and_execute
from bot.state import load_state, reconcile_positions, save_state
from models import check_model_staleness
from utils import load_config, setup_logger

logger = setup_logger("logs/bot.log")


def execute_cycle(client, model_bundle, config):
    """Executes a single market scan, position management, and alert dispatch cycle."""
    state_path = config["paths"]["state_file"]
    state = load_state(state_path, initial_capital=config["risk_management"]["capital_usdt"])

    # Live position and balance reconciliation (if live trading mode and keys configured)
    if config.get("trading_mode") == "live" and client.has_credentials:
        reconcile_positions(client, state, config)

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
            entry_p = s["price"]
            tp2_p = s.get("tp2", entry_p * 1.02)
            sl_p = s.get("stop_loss", entry_p * 0.98)
            est_win_pct = ((tp2_p / entry_p) - 1.0) * 100
            est_loss_pct = ((sl_p / entry_p) - 1.0) * 100

            strat_label = s.get("strategy_setups", "AI Multi-Tool Confluence")
            strat_score_val = s.get("strategy_score", s.get("confidence_pct", 60.0))
            ind_summary = s.get("indicators_summary", "RSI / Bollinger / StochRSI Aligned")

            fields = [
                {"name": "🪙 Coin", "value": f"**{s['symbol']}**", "inline": True},
                {"name": "💵 Amount Bought", "value": f"**${s['size_usdt']:.2f} USDT**", "inline": True},
                {"name": "📈 Buy Price", "value": f"${entry_p:,.2f}", "inline": True},
                {"name": "🎯 Target Price (Take Profit)", "value": f"**${tp2_p:,.2f}** ({est_win_pct:+.2f}%)", "inline": True},
                {"name": "🛡️ Safety Price (Stop Loss)", "value": f"${sl_p:,.2f} ({est_loss_pct:+.2f}%)", "inline": True},
                {"name": "🤖 AI Confidence", "value": f"**{s['confidence_pct']:.0f}%**", "inline": True},
                {"name": "🎯 Strategy Setup", "value": f"**{strat_label}** (Score: {strat_score_val:.0f}/100)", "inline": False},
                {"name": "📊 Technical Indicators", "value": f"`{ind_summary}`", "inline": False},
                {"name": "📌 Next Steps", "value": "Sit back! The bot is watching the market and will exit automatically.", "inline": False},
            ]
            title = f"🟢 BOUGHT {s['symbol']} (${s['size_usdt']:.2f})"
            desc = f"The AI detected a high-probability buying opportunity on **{s['symbol']}** using **{strat_label}**."
            send_discord_alert(webhook, title, fields, color=3066993, description=desc)

        for cs in closed_signals:
            action = cs.get("action", "")
            pnl_val = cs["net_pnl"]
            pnl_pct = cs["pnl_pct"]

            if action == "BREAKEVEN LOCK (SMALL ACCOUNT)":
                title = f"🛡️ STOP LOSS MOVED TO BREAKEVEN: {cs['symbol']}"
                color = 3447003  # Blue
                fields = [
                    {"name": "🪙 Coin", "value": f"**{cs['symbol']}**", "inline": True},
                    {"name": "🔒 Trade Status", "value": "**100% Position Still Open (Risk-Free)**", "inline": True},
                    {"name": "🎯 Trigger Price", "value": f"${cs['exit_price']:,.2f}", "inline": True},
                    {"name": "🛡️ Protected Breakeven SL", "value": f"**${cs['entry_price']:,.2f}**", "inline": True},
                    {"name": "ℹ️ Note", "value": cs["reason"], "inline": False},
                    {"name": "🚀 Next Target", "value": "Position is still open and running risk-free toward TP2!", "inline": False},
                ]
            elif action == "PARTIAL TP1 (50% SOLD)":
                title = f"🎯 TP1 HIT (50% PROFIT SECURED): {cs['symbol']}"
                color = 5763719  # Green
                fields = [
                    {"name": "🪙 Coin", "value": f"**{cs['symbol']}**", "inline": True},
                    {"name": "💰 Booked Profit (50%)", "value": f"**${pnl_val:+,.2f} USDT** ({pnl_pct:+.2f}%)", "inline": True},
                    {"name": "🏁 Sold 50% At", "value": f"${cs['exit_price']:,.2f}", "inline": True},
                    {"name": "🛡️ Remaining 50%", "value": f"Stop Loss locked to Entry (${cs['entry_price']:,.2f}) — Risk-Free!", "inline": False},
                    {"name": "ℹ️ Reason", "value": cs["reason"], "inline": False},
                ]
            else:
                is_win = cs["outcome"] == "WIN"
                color = 5763719 if is_win else 15548997
                title = f"🎉 WON TRADE: {cs['symbol']}" if is_win else f"🛡️ TRADE CLOSED: {cs['symbol']}"
                fields = [
                    {"name": "🪙 Coin", "value": f"**{cs['symbol']}**", "inline": True},
                    {"name": "💰 Net Profit / Loss", "value": f"**${pnl_val:+,.2f} USDT** ({pnl_pct:+.2f}%)", "inline": True},
                    {"name": "🏁 Exit Price", "value": f"${cs['exit_price']:,.2f}", "inline": True},
                    {"name": "📈 Entry Price", "value": f"${cs['entry_price']:,.2f}", "inline": True},
                    {"name": "ℹ️ Reason", "value": cs["reason"], "inline": False},
                ]
            send_discord_alert(webhook, title, fields, color=color)

        # Hourly Radar & Position Status Alert
        hourly_radar_enabled = discord_cfg.get("hourly_radar_alert", False)
        if hourly_radar_enabled and not new_signals and not closed_signals:
            now_utc = datetime.now(timezone.utc)
            last_alert_str = state.get("last_hourly_alert")
            should_send_hourly = True
            if last_alert_str:
                try:
                    last_alert_dt = datetime.fromisoformat(last_alert_str.replace(" ", "T"))
                    if last_alert_dt.tzinfo is None:
                        last_alert_dt = last_alert_dt.replace(tzinfo=timezone.utc)
                    if (now_utc - last_alert_dt).total_seconds() < 3000:
                        should_send_hourly = False
                except Exception:
                    pass

            if should_send_hourly:
                state["last_hourly_alert"] = now_utc.strftime("%Y-%m-%d %H:%M:%S")
                total_equity = state["balance_usdt"]
                active_str = ""
                if state.get("open_positions"):
                    pos_lines = []
                    for psym, p in state["open_positions"].items():
                        c_p = p.get("curr_price", p["entry_price"])
                        u_pnl = p.get("unrealized_pnl", 0.0)
                        u_pct = p.get("unrealized_pnl_pct", 0.0)
                        total_equity += (p["allocated_usdt"] + u_pnl)
                        tp1_val = p.get("take_profit_1", p.get("take_profit", 0.0))
                        sl_val = p.get("stop_loss", 0.0)
                        gauge = render_trade_progress(c_p, p["entry_price"], tp1_val, sl_val)
                        pos_lines.append(
                            f"• **{psym}**: Now `${c_p:,.2f}` | Entry `${p['entry_price']:,.2f}` | PnL: `${u_pnl:+.2f}` ({u_pct:+.2f}%)\n"
                            f"  🎯 Target: {gauge} | SL: `${sl_val:,.2f}`"
                        )
                    active_str = "\n".join(pos_lines)
                else:
                    active_str = "🛡️ 100% Cash Defense (Waiting for high-conviction breakout)"

                top_cands = sorted(analysis_rows, key=lambda x: (x["prob_win"], x["strategy_score"]), reverse=True)[:3]
                cand_lines = []
                for c in top_cands:
                    cand_lines.append(f"• **{c['symbol']}**: AI Conviction `{c['prob_win']*100:.1f}%` | Status: `{c['status']}` | Setup: *{c['active_setups']}*")
                cand_str = "\n".join(cand_lines) if cand_lines else "Scanning..."

                fields = [
                    {"name": "💼 Total Portfolio Equity", "value": f"**${total_equity:,.2f} USDT** (Free Cash: ${state['balance_usdt']:,.2f})", "inline": False},
                    {"name": "📊 Active Positions & Targets", "value": active_str, "inline": False},
                    {"name": "🎯 Top Market Radar Scans", "value": cand_str, "inline": False},
                    {"name": "⚙️ Bot Health", "value": "🟢 Online 24/7 | Checking candle close & volatility every cycle", "inline": False},
                ]
                title = "📡 Hourly Market Radar & Position Update"
                send_discord_alert(webhook, title, fields, color=3447003)

    # Save state
    state["last_update"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    save_state(state, state_path)


def main():
    """CLI argument parsing and continuous or single-cycle execution entrypoint."""
    parser = argparse.ArgumentParser(description="Binance AI Quantitative Trading Bot")
    parser.add_argument("--config", default="config.json", help="Path to config file")
    parser.add_argument("--model", default="models/binance_ai_model.joblib", help="Path to model bundle")
    parser.add_argument("--scan", action="store_true", help="Run single scan and exit")
    parser.add_argument("--loop", action="store_true", help="Run continuously on candle close loop")
    parser.add_argument("--capital", type=float, default=None, help="Override starting capital")
    parser.add_argument("--test-discord", action="store_true", help="Send a test embed to Discord")
    parser.add_argument("--digest", action="store_true", help="Send 24h portfolio digest embed to Discord")
    parser.add_argument("--force", action="store_true", help="Force action (e.g. bypass digest frequency limit)")

    args = parser.parse_args()
    config = load_config(args.config)

    if args.digest:
        state = load_state(config["paths"]["state_file"], initial_capital=config["risk_management"]["capital_usdt"])
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if state.get("last_digest_date") == today_str and not args.force:
            logger.info("[Digest] Daily digest already dispatched for today (%s). Skipping.", today_str)
        else:
            ok, res = send_daily_digest(state, config)
            logger.info("Daily digest dispatch result: %s", res)
        if not args.scan and not args.loop:
            return

    if args.test_discord:
        d_cfg = config.get("discord", {})
        webhook = os.getenv("DISCORD_WEBHOOK_URL", d_cfg.get("webhook_url", "")).strip()
        fields = [
            {"name": "Status", "value": "Online & Operational", "inline": True},
            {"name": "Trading Mode", "value": config.get("trading_mode", "paper").upper(), "inline": True},
            {"name": "Exchange", "value": "Binance Spot", "inline": True},
            {"name": "Timeframe", "value": config.get("timeframe", "1h"), "inline": True},
        ]
        ok, res = send_discord_alert(webhook, "🤖 QuantLab Binance AI Bot Test Alert", fields, color=3447003)
        logger.info("Discord test result: %s", res)
        if not args.scan and not args.loop:
            return

    if args.capital:
        config["risk_management"]["capital_usdt"] = args.capital

    if not os.path.exists(args.model):
        sys.exit(f"Model file '{args.model}' not found. Please run 'python train.py' first.")

    model_bundle = joblib.load(args.model)
    is_stale, age_days, stale_msg = check_model_staleness(model_bundle)
    if is_stale:
        logger.warning("[MODEL STALENESS WARNING] %s", stale_msg)
    else:
        logger.info("[MODEL STATUS] %s", stale_msg)

    client = BinanceClient(config)

    execute_cycle(client, model_bundle, config)

    if args.loop:
        tf_sec = 3600 if config["timeframe"] == "1h" else (14400 if config["timeframe"] == "4h" else 86400)
        logger.info("[Loop Active] Monitoring Binance on %s candles...", config["timeframe"])
        last_digest_date = datetime.now(timezone.utc).date()
        while True:
            now_dt = datetime.now(timezone.utc)
            if now_dt.date() > last_digest_date and now_dt.hour == 0:
                try:
                    st = load_state(config["paths"]["state_file"])
                    send_daily_digest(st, config)
                    last_digest_date = now_dt.date()
                except Exception as e:
                    logger.warning("Daily digest dispatch failed: %s", e)
            now = time.time()
            sleep_time = tf_sec - (now % tf_sec) + 10
            time.sleep(sleep_time)
            try:
                execute_cycle(client, model_bundle, config)
            except Exception as e:
                logger.error("Cycle execution error: %s", e, exc_info=True)


if __name__ == "__main__":
    main()
