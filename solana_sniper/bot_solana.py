"""Main entrypoint and real-time execution loop for Solana Meme & Pump.fun Sniper."""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple, Optional

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

from solana_sniper.alerts import send_discord_sniper_alert
from solana_sniper.dashboard import print_sniper_dashboard
from solana_sniper.rugcheck import verify_token_security
from solana_sniper.scanner import scan_solana_momentum_tokens
from solana_sniper.trader import (
    execute_snipe,
    load_sniper_state,
    manage_active_positions,
    save_sniper_state,
)

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [solana_sniper] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join("logs", "solana_sniper.log"), encoding="utf-8"),
    ],
)
logger = logging.getLogger("solana_sniper")


def load_config(config_path: str = "solana_sniper/config.json"):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_sniper_cycle(config: Dict[str, Any], state: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    recent_events = []
    d_cfg = config.get("discord", {})
    webhook = os.getenv("DISCORD_WEBHOOK_URL", d_cfg.get("webhook_url", "")).strip()

    # 1. Manage Active Positions (TP1 Moonbags, Trailing Stops, Stop Loss)
    partial_harvests, closed_trades = manage_active_positions(state, config)
    
    for h in partial_harvests:
        recent_events.append(h)
        if d_cfg.get("enabled") and webhook:
            fields = [
                {"name": "🪙 Token", "value": f"**{h['symbol']}**", "inline": True},
                {"name": "🎯 Multiplier", "value": f"**{h['multiplier']:.2f}x** 🚀", "inline": True},
                {"name": "💰 Net Profit", "value": f"**+{h['pnl_sol']:.3f} SOL**", "inline": True},
                {"name": "🛡️ Status", "value": "100% Capital Recovered. Remaining 50% Running as Moonbag!", "inline": False},
            ]
            send_discord_sniper_alert(webhook, f"🎯 2X MOONBAG SECURED: {h['symbol']}", fields, color=5763719, desc=h["reason"])

    for c in closed_trades:
        recent_events.append(c)
        if d_cfg.get("enabled") and webhook:
            color = 5763719 if c["outcome"] == "WIN" else 15548997
            fields = [
                {"name": "🪙 Token", "value": f"**{c['symbol']}**", "inline": True},
                {"name": "🏁 Multiplier", "value": f"**{c['multiplier']:.2f}x**", "inline": True},
                {"name": "💵 Realized PnL", "value": f"**{c['pnl_sol']:+.4f} SOL**", "inline": True},
                {"name": "📋 Exit Reason", "value": c["reason"], "inline": False},
            ]
            send_discord_sniper_alert(webhook, f"🏁 SNIPE CLOSED: {c['symbol']} ({c['multiplier']:.2f}x)", fields, color=color)

    # 2. Scan for Breakout Candidates
    open_pos = state.get("open_positions", {})
    max_pos = config.get("trading", {}).get("max_open_positions", 3)
    candidates = scan_solana_momentum_tokens(config.get("trading", {}))

    if len(open_pos) < max_pos:
        cooldown_mints = state.get("cooldown_mints", {})
        now_ts = time.time()
        for cand in candidates:
            mint = cand["mint"]
            if mint in open_pos:
                continue

            # Prevent re-entering recently stopped-out tokens (60m cooldown)
            if mint in cooldown_mints and (now_ts - cooldown_mints[mint]) < 3600:
                continue

            # 3. RugCheck Security Verification
            is_safe, rug_reason, _ = verify_token_security(mint, config.get("safety", {}))
            if not is_safe:
                logger.warning("🛡️ [RUG BLOCKED] Skipped %s (%s): %s", cand["symbol"], mint[:8], rug_reason)
                continue

            # 4. Execute Snipe
            new_pos = execute_snipe(cand, state, config)
            if new_pos:
                snipe_ev = {
                    "action": "SNIPED",
                    "symbol": cand["symbol"],
                    "mint": mint,
                    "multiplier": 1.0,
                    "pnl_sol": 0.0,
                    "reason": f"Volume Surge ({cand['volume_5m']:,.0f} USD, Ratio {cand['buy_sell_ratio']:.1f}x)",
                }
                recent_events.append(snipe_ev)
                if d_cfg.get("enabled") and webhook:
                    fields = [
                        {"name": "🪙 Token", "value": f"**{cand['symbol']}**", "inline": True},
                        {"name": "💵 Entry Price", "value": f"${cand['price_usd']:.8f}", "inline": True},
                        {"name": "⚡ Allocated Size", "value": f"**{new_pos['allocated_sol']:.2f} SOL**", "inline": True},
                        {"name": "📊 5m Volume", "value": f"${cand['volume_5m']:,.0f}", "inline": True},
                        {"name": "📈 Buy/Sell Ratio", "value": f"{cand['buy_sell_ratio']:.1f}x Buys", "inline": True},
                        {"name": "🛡️ RugCheck", "value": "PASSED (Safe Mint & LP)", "inline": True},
                        {"name": "🔗 DexScreener", "value": f"[View Chart]({cand['url']})", "inline": False},
                    ]
                    send_discord_sniper_alert(webhook, f"🟢 NEW SOLANA SNIPE: {cand['symbol']}", fields, color=3066993, desc="Automated sniper entered on high volume impulse.")
                break

    return candidates, recent_events


def main():
    parser = argparse.ArgumentParser(description="Solana Meme Token & Pump.fun Velocity Sniper")
    parser.add_argument("--config", default="solana_sniper/config.json", help="Path to config file")
    parser.add_argument("--scan", action="store_true", help="Run single scan and exit")
    parser.add_argument("--loop", action="store_true", help="Run continuous 15s sniper hunting loop")
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)
    config = load_config(args.config)
    state_path = config.get("paths", {}).get("state_file", "solana_sniper/state.json")
    state = load_sniper_state(state_path, initial_capital_sol=config.get("trading", {}).get("capital_sol", 1.0))

    if args.scan:
        candidates, recent_events = run_sniper_cycle(config, state)
        print_sniper_dashboard(candidates, state, config, recent_events)
        state["last_update"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        save_sniper_state(state, state_path)
        return

    logger.info("⚡ Solana High-Velocity Meme Sniper active in loop mode...")
    try:
        while True:
            try:
                candidates, recent_events = run_sniper_cycle(config, state)
                print_sniper_dashboard(candidates, state, config, recent_events)
                state["last_update"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                save_sniper_state(state, state_path)
            except Exception as e:
                logger.error("Sniper loop cycle error: %s", e, exc_info=True)

            time.sleep(15)
    except KeyboardInterrupt:
        save_sniper_state(state, state_path)
        print("\n🛑 Solana Sniper gracefully paused by user. State saved.")


if __name__ == "__main__":
    main()

