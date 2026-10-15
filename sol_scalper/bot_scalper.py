"""Main continuous execution loop for SOL/USDC Institutional Scalper."""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from dotenv import load_dotenv

from sol_scalper.alerts import send_discord_scalper_alert
from sol_scalper.dashboard import print_scalper_dashboard
from sol_scalper.market_data import compute_market_snapshot
from sol_scalper.strategy import evaluate_active_position, evaluate_scalp_entry
from sol_scalper.trader import (
    execute_buy_order,
    execute_sell_order,
    load_scalper_state,
    save_scalper_state,
)

load_dotenv()
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [sol_scalper] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join("logs", "sol_scalper.log"), encoding="utf-8"),
    ],
)
logger = logging.getLogger("sol_scalper")


def load_config(config_path: str = "sol_scalper/config.json") -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_scalper_cycle(config: Dict[str, Any], state: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    recent_events = []
    d_cfg = config.get("discord", {})
    webhook = os.getenv("DISCORD_WEBHOOK_URL", d_cfg.get("webhook_url", "")).strip()

    snapshot = compute_market_snapshot(config.get("strategy", {}))
    if not snapshot:
        return None, recent_events

    active_pos = state.get("active_position")
    trading_cfg = config.get("trading", {})

    # 1. Manage Active Position (Take-Profit / Stop-Loss / Trailing)
    if active_pos:
        should_exit, outcome, reason = evaluate_active_position(active_pos, snapshot, trading_cfg)
        if should_exit:
            close_ev = execute_sell_order(snapshot, state, config, outcome, reason)
            recent_events.append(close_ev)
            if d_cfg.get("enabled") and webhook:
                color = 5763719 if outcome == "WIN" else 15548997
                fields = [
                    {"name": "🪙 Asset", "value": "**SOL/USDC**", "inline": True},
                    {"name": "💵 Exit Price", "value": f"${close_ev['exit_price']:.2f}", "inline": True},
                    {"name": "📊 Return", "value": f"**{close_ev['gain_pct']:+.2f}%**", "inline": True},
                    {"name": "💰 Net PnL", "value": f"**{close_ev['pnl_usd']:+.2f} USD**", "inline": True},
                    {"name": "📋 Reason", "value": reason, "inline": False},
                ]
                title = f"🎯 SCALP WIN (+{close_ev['gain_pct']:.2f}%)" if outcome == "WIN" else f"🛡️ RISK GUARD STOP (-{abs(close_ev['gain_pct']):.2f}%)"
                send_discord_scalper_alert(webhook, title, fields, color=color)

    # 2. Look for New Scalp Entry
    else:
        # Check cooldown
        last_trade_t = state.get("last_trade_time", 0)
        cooldown_sec = trading_cfg.get("cooldown_seconds", 60)
        if (time.time() - last_trade_t) >= cooldown_sec:
            should_buy, reason = evaluate_scalp_entry(snapshot, config.get("strategy", {}))
            if should_buy:
                new_pos = execute_buy_order(snapshot, state, config, reason)
                if new_pos:
                    buy_ev = {
                        "action": "SCALP_BOUGHT",
                        "symbol": "SOLUSDC",
                        "price": new_pos["entry_price_usd"],
                        "size_usd": new_pos["allocated_usd"],
                        "reason": reason,
                    }
                    recent_events.append(buy_ev)
                    if d_cfg.get("enabled") and webhook:
                        fields = [
                            {"name": "🪙 Asset", "value": "**SOL/USDC**", "inline": True},
                            {"name": "💵 Entry Price", "value": f"${new_pos['entry_price_usd']:.2f}", "inline": True},
                            {"name": "⚡ Allocated", "value": f"**${new_pos['allocated_usd']:.2f}** ({new_pos['sol_amount']:.4f} SOL)", "inline": True},
                            {"name": "🎯 Target TP (+1.8%)", "value": f"${new_pos['target_tp_price']:.2f}", "inline": True},
                            {"name": "🛡️ Stop Loss (-0.9%)", "value": f"${new_pos['target_sl_price']:.2f}", "inline": True},
                            {"name": "📈 Setup", "value": reason, "inline": False},
                        ]
                        send_discord_scalper_alert(webhook, "🟢 NEW SOL/USDC SCALP OPENED", fields, color=3066993, desc="Pullback entry inside bullish trend.")

    return snapshot, recent_events


def main():
    parser = argparse.ArgumentParser(description="SOL/USDC Quantitative Institutional Scalper")
    parser.add_argument("--config", default="sol_scalper/config.json", help="Path to config file")
    parser.add_argument("--scan", action="store_true", help="Run single market check and exit")
    parser.add_argument("--loop", action="store_true", help="Run continuous 10s scalper loop")
    args = parser.parse_args()

    config = load_config(args.config)
    state_path = config.get("paths", {}).get("state_file", "sol_scalper/state.json")
    state = load_scalper_state(state_path, initial_capital_usd=config.get("trading", {}).get("starting_capital_usd", 50.0))

    if args.scan:
        snap, events = run_scalper_cycle(config, state)
        print_scalper_dashboard(snap, state, config, events)
        state["last_update"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        save_scalper_state(state, state_path)
        return

    logger.info("⚡ SOL/USDC Systematic Scalper active in 10s loop mode...")
    try:
        while True:
            try:
                snap, events = run_scalper_cycle(config, state)
                print_scalper_dashboard(snap, state, config, events)
                state["last_update"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                save_scalper_state(state, state_path)
            except Exception as e:
                logger.error("Scalper loop error: %s", e, exc_info=True)

            time.sleep(10)
    except KeyboardInterrupt:
        save_scalper_state(state, state_path)
        print("\n🛑 SOL/USDC Scalper gracefully paused by user. State saved.")


if __name__ == "__main__":
    main()
