"""Main execution loop and CLI entrypoint for Halal Spot Volatility Grid Bot."""

import argparse
from datetime import datetime, timezone
import json
import logging
import os
import sys
import time

from dotenv import load_dotenv

from binance_client import BinanceClient
from bot.alerts import send_daily_digest, send_discord_alert
from bot.dashboard import print_dashboard
from bot.grid import run_grid_cycle
from bot.state import load_state, reconcile_positions, save_state
from utils import load_config, setup_logger

load_dotenv()
logger = logging.getLogger("quantlab")


def execute_cycle(client, config, model_bundle=None):
    """Executes a single market scan, grid cycle, and alert dispatch."""
    state_path = config["paths"]["state_file"]
    state = load_state(state_path, initial_capital=config["risk_management"]["capital_usdt"])

    # Live position and balance reconciliation (if live trading mode and keys configured)
    if config.get("trading_mode") == "live" and client.has_credentials:
        reconcile_positions(client, state, config)

    new_signals = []
    closed_signals = []
    analysis_rows = []

    # Run Halal Spot Volatility Grid cycle
    new_grid_buys, harvested_profits, grid_status = run_grid_cycle(client, state, config)
    cb_status = grid_status.get("status", "ACTIVE_HARVESTING")

    for h in harvested_profits:
        closed_signals.append({
            "symbol": h["symbol"],
            "action": "GRID_HARVEST",
            "entry_price": h["entry_price"],
            "exit_price": h["exit_price"],
            "net_pnl": h["net_pnl"],
            "pnl_pct": h["pnl_pct"],
            "outcome": "WIN",
            "position_id": h["position_id"],
            "reason": h["reason"],
        })

    for b in new_grid_buys:
        new_signals.append({
            "symbol": b["symbol"],
            "action": "GRID_BUY",
            "price": b["entry_price"],
            "tp2": b["take_profit"],
            "stop_loss": b.get("stop_loss", b["entry_price"] * 0.95),
            "size_usdt": b["allocated_usdt"],
            "confidence_pct": 100.0,
            "strategy_setups": "Adaptive Halal Volatility Grid",
            "strategy_score": 95.0,
            "indicators_summary": f"Targeting +{((b['take_profit']/b['entry_price'])-1)*100:.1f}% bounce oscillation",
        })

    basket = config.get("grid_trading", {}).get("halal_basket", ["SOL/USDT", "AVAX/USDT", "NEAR/USDT", "LINK/USDT"])
    basket_prices = grid_status.get("basket_prices", {})
    basket_vol = grid_status.get("basket_vol", {})
    active_symbols = grid_status.get("active_symbols", [grid_status.get("active_symbol")])
    cand_details = grid_status.get("candidate_details", {})

    for sym in basket:
        is_active = (sym in active_symbols)
        sym_p = basket_prices.get(sym, 0.0)
        if sym_p <= 0.0:
            try:
                sym_p = client.get_ticker_price(sym) if hasattr(client, "get_ticker_price") else 0.0
            except Exception:
                sym_p = 0.0

        c_info = cand_details.get(sym, {})
        has_signal = c_info.get("signal_triggered", False)
        status_text = "SNIPING" if is_active else ("SIGNAL CONFIRMED" if has_signal else "MONITORING")
        setup_desc = c_info.get("setup_name", f"Consolidating ({basket_vol.get(sym, 0.0):.1f}% ATR)")
        prob_win = c_info.get("prob_win", 0.85 if is_active else 0.50)

        analysis_rows.append({
            "symbol": sym,
            "price": sym_p,
            "prob_win": prob_win,
            "strategy_score": 92.0 if has_signal else (85.0 if is_active else 50.0),
            "active_setups": [setup_desc],
            "mtf_status": "BULL TREND" if is_active or has_signal else "NORMAL",
            "status": status_text,
        })

    # Also include macro benchmark if available
    macro_bench = config.get("grid_trading", {}).get("macro_benchmark", "BTC/USDT")
    if macro_bench and macro_bench not in basket:
        bench_p = basket_prices.get(macro_bench, 0.0)
        if bench_p <= 0.0:
            try:
                bench_p = client.get_ticker_price(macro_bench) if hasattr(client, "get_ticker_price") else 0.0
            except Exception:
                bench_p = 0.0
        analysis_rows.append({
            "symbol": macro_bench,
            "price": bench_p,
            "prob_win": 0.70,
            "strategy_score": 75.0,
            "active_setups": ["Macro Benchmark"],
            "mtf_status": "BENCHMARK",
            "status": "MONITORING",
        })

    # Print CLI dashboard
    print_dashboard(analysis_rows, new_signals, closed_signals, state, config, cb_status)

    # Discord alerts
    discord_cfg = config.get("discord", {})
    webhook = os.getenv("DISCORD_WEBHOOK_URL", discord_cfg.get("webhook_url", "")).strip()
    if discord_cfg.get("enabled", False) and webhook:
        for s in new_signals:
            entry_p = s["price"]
            tp2_p = s.get("tp2", entry_p * 1.02)
            sl_p = s.get("stop_loss", entry_p * 0.95)
            est_win_pct = ((tp2_p / entry_p) - 1.0) * 100
            est_loss_pct = ((sl_p / entry_p) - 1.0) * 100

            fields = [
                {"name": "🪙 Coin", "value": f"**{s['symbol']}**", "inline": True},
                {"name": "💵 Amount Bought", "value": f"**${s['size_usdt']:.2f} USDT**", "inline": True},
                {"name": "📈 Buy Price", "value": f"${entry_p:,.4f}", "inline": True},
                {"name": "🎯 Take-Profit Target", "value": f"**${tp2_p:,.4f}** ({est_win_pct:+.2f}%)", "inline": True},
                {"name": "🛡️ Emergency Stop", "value": f"${sl_p:,.4f} ({est_loss_pct:+.2f}%)", "inline": True},
                {"name": "🕌 Shariah Status", "value": "100% Spot Asset Ownership (*Qabd*)", "inline": True},
                {"name": "⚙️ Strategy", "value": "**Adaptive Halal Volatility Grid**", "inline": False},
            ]
            title = f"🟢 SPOT GRID BUY: {s['symbol']} (${s['size_usdt']:.2f})"
            desc = f"Dip harvested on **{s['symbol']}**. Automated Spot order placed."
            send_discord_alert(webhook, title, fields, color=3066993, description=desc)

        for cs in closed_signals:
            pnl_val = cs["net_pnl"]
            pnl_pct = cs["pnl_pct"]
            title = f"💰 GRID PROFIT HARVESTED: {cs['symbol']}"
            fields = [
                {"name": "🪙 Coin", "value": f"**{cs['symbol']}**", "inline": True},
                {"name": "💵 Net Profit Booked", "value": f"**+${pnl_val:,.2f} USDT** ({pnl_pct:+.2f}%)", "inline": True},
                {"name": "🏁 Sold At", "value": f"${cs['exit_price']:,.4f}", "inline": True},
                {"name": "📈 Bought At", "value": f"${cs['entry_price']:,.4f}", "inline": True},
                {"name": "📊 Total Grid Profits", "value": f"**${state.get('grid_state', {}).get('total_grid_profit_usdt', 0.0):.2f} USDT**", "inline": True},
                {"name": "🕌 Shariah Status", "value": "100% Halal Commerce (*Bay'*)", "inline": True},
            ]
            send_discord_alert(webhook, title, fields, color=5763719)

        # Hourly Radar Alert
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
                from bot.alerts import send_radar_status_alert
                send_radar_status_alert(state, config)

    # Save state
    state["last_update"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    save_state(state, state_path)


def main():
    """CLI argument parsing and continuous or single-cycle execution entrypoint."""
    parser = argparse.ArgumentParser(description="Halal Spot Volatility Grid Trading Bot")
    parser.add_argument("--config", default="config.json", help="Path to config file")
    parser.add_argument("--scan", action="store_true", help="Run single scan and exit")
    parser.add_argument("--loop", action="store_true", help="Run continuously on candle close loop")
    parser.add_argument("--capital", type=float, default=None, help="Override starting capital")
    parser.add_argument("--test-discord", action="store_true", help="Send a test embed to Discord")
    parser.add_argument("--status", action="store_true", help="Send real-time position status & progress gauge to Discord")
    parser.add_argument("--digest", action="store_true", help="Send 24h portfolio digest embed to Discord")
    parser.add_argument("--force", action="store_true", help="Force action (e.g. bypass digest frequency limit)")

    args = parser.parse_args()
    config = load_config(args.config)
    log_file = os.path.join(config.get("paths", {}).get("logs_dir", "logs"), "bot.log")
    setup_logger(log_file)

    if args.status:
        state = load_state(config["paths"]["state_file"], initial_capital=config["risk_management"]["capital_usdt"])
        from bot.alerts import send_radar_status_alert
        ok, res = send_radar_status_alert(state, config)
        logger.info("Live radar status dispatch result: %s", res)
        if not args.scan and not args.loop:
            return

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
            {"name": "Strategy", "value": "Halal Spot Volatility Grid", "inline": True},
            {"name": "Timeframe", "value": config.get("timeframe", "1h"), "inline": True},
        ]
        ok, res = send_discord_alert(webhook, "🤖 Halal Quantitative Spot Grid Bot Test Alert", fields, color=3447003)
        logger.info("Discord test result: %s", res)
        if not args.scan and not args.loop:
            return

    if args.capital:
        config["risk_management"]["capital_usdt"] = args.capital

    client = BinanceClient(config)

    execute_cycle(client, config)

    if args.loop:
        tf_sec = 3600 if config["timeframe"] == "1h" else 1800
        logger.info("[Loop Active] Monitoring Binance Halal Grid on %s intervals...", config["timeframe"])
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
                execute_cycle(client, config)
            except Exception as e:
                logger.error("Cycle execution error: %s", e, exc_info=True)


if __name__ == "__main__":
    main()
