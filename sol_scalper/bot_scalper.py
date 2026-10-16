"""Main continuous execution loop for Multi-Asset (JUP + SOL) Quantitative Scalper."""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

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
from sol_scalper.market_data import compute_basket_snapshots, fetch_btc_macro_regime
from sol_scalper.strategy import evaluate_active_position, evaluate_basket_entry
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


def run_scalper_cycle(config: Dict[str, Any], state: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    recent_events = []
    d_cfg = config.get("discord", {})
    webhook = os.getenv("DISCORD_WEBHOOK_URL", d_cfg.get("webhook_url", "")).strip()

    strat_cfg = config.get("strategy", {})
    btc_regime = fetch_btc_macro_regime(strat_cfg)

    basket = config.get("basket", [])
    snapshots = compute_basket_snapshots(basket, strat_cfg)
    if not snapshots:
        return {}, recent_events, btc_regime

    open_pos = state.setdefault("open_positions", {})
    trading_cfg = config.get("trading", {})
    max_pos = trading_cfg.get("max_concurrent_positions", 1)

    # 1. Manage Active Positions across all held assets
    closed_syms = []
    for sym, pos in list(open_pos.items()):
        if sym in snapshots:
            snap = snapshots[sym]
            should_exit, outcome, reason = evaluate_active_position(pos, snap, trading_cfg)
            if should_exit:
                close_ev = execute_sell_order(sym, snap, state, config, outcome, reason)
                if close_ev:
                    recent_events.append(close_ev)
                    closed_syms.append(sym)
                    if d_cfg.get("enabled") and webhook:
                        color = 5763719 if outcome == "WIN" else 15548997
                        fields = [
                            {"name": "🪙 Asset", "value": f"**{sym}**", "inline": True},
                            {"name": "💵 Exit Price", "value": f"${close_ev['exit_price']:.4f}", "inline": True},
                            {"name": "📊 Return", "value": f"**{close_ev['gain_pct']:+.2f}%**", "inline": True},
                            {"name": "💰 Net PnL", "value": f"**{close_ev['pnl_usd']:+.2f} USD**", "inline": True},
                            {"name": "📋 Reason", "value": reason, "inline": False},
                        ]
                        title = f"🎯 SCALP WIN (+{close_ev['gain_pct']:.2f}%): {sym}" if outcome == "WIN" else f"🛡️ RISK GUARD STOP (-{abs(close_ev['gain_pct']):.2f}%): {sym}"
                        send_discord_scalper_alert(webhook, title, fields, color=color)

    # 2. Look for New Scalp Entries across Basket Assets
    if len(open_pos) < max_pos:
        last_trade_t = state.get("last_trade_time", 0)
        cooldown_sec = trading_cfg.get("cooldown_seconds", 120)

        if (time.time() - last_trade_t) >= cooldown_sec:
            for item in basket:
                sym = item["symbol"]
                if sym in open_pos:
                    continue  # already in position
                if sym not in snapshots:
                    continue

                snap = snapshots[sym]
                should_buy, reason, dynamic_targets = evaluate_basket_entry(snap, strat_cfg, btc_regime=btc_regime)
                if should_buy:
                    new_pos = execute_buy_order(sym, snap, state, config, reason, dynamic_targets=dynamic_targets)
                    if new_pos:
                        buy_ev = {
                            "action": "BOUGHT",
                            "symbol": sym,
                            "price": new_pos["entry_price_usd"],
                            "size_usd": new_pos["allocated_usd"],
                            "reason": reason,
                        }
                        recent_events.append(buy_ev)
                        if d_cfg.get("enabled") and webhook:
                            fields = [
                                {"name": "🪙 Asset", "value": f"**{sym}**", "inline": True},
                                {"name": "💵 Entry Price", "value": f"${new_pos['entry_price_usd']:.4f}", "inline": True},
                                {"name": "⚡ Allocated", "value": f"**${new_pos['allocated_usd']:.2f}** ({new_pos['units']:.2f} units)", "inline": True},
                                {"name": "🎯 Volatility Target TP", "value": f"+{new_pos.get('tp_pct', 5.0):.1f}% (${new_pos['target_tp_price']:.4f})", "inline": True},
                                {"name": "🛡️ Volatility Stop Loss", "value": f"-{new_pos.get('sl_pct', 2.0):.1f}% (${new_pos['target_sl_price']:.4f})", "inline": True},
                                {"name": "📋 Signal Reason", "value": reason, "inline": False},
                            ]
                            send_discord_scalper_alert(webhook, f"🟢 SCALPER ENTRY: {sym}", fields, color=3066993)
                        break  # Max 1 position constraint

    return snapshots, recent_events, btc_regime


def main():
    parser = argparse.ArgumentParser(description="Multi-Asset 15-Minute Quantitative Scalper")
    parser.add_argument("--config", default="sol_scalper/config.json", help="Path to config file")
    parser.add_argument("--scan", action="store_true", help="Run single market check and exit")
    parser.add_argument("--loop", action="store_true", help="Run continuous 10s scalper loop")
    args = parser.parse_args()

    config = load_config(args.config)
    state_path = config.get("paths", {}).get("state_file", "sol_scalper/state.json")
    state = load_scalper_state(state_path, initial_capital_usd=config.get("trading", {}).get("starting_capital_usd", 50.0))

    if args.scan:
        snaps, events, btc_regime = run_scalper_cycle(config, state)
        print_scalper_dashboard(snaps, state, config, events, btc_regime=btc_regime)
        state["last_update"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        save_scalper_state(state, state_path)
        return

    logger.info("⚡ 15M Quantitative Scalper active in 10s loop mode...")
    try:
        while True:
            try:
                snaps, events, btc_regime = run_scalper_cycle(config, state)
                print_scalper_dashboard(snaps, state, config, events, btc_regime=btc_regime)
                state["last_update"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                save_scalper_state(state, state_path)
            except Exception as e:
                logger.error("Scalper loop error: %s", e, exc_info=True)

            time.sleep(10)
    except KeyboardInterrupt:
        save_scalper_state(state, state_path)
        print("\n🛑 Multi-Asset Scalper gracefully paused by user. State saved.")


if __name__ == "__main__":
    main()
