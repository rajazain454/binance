"""Discord webhook alerts and daily digests for QuantLab Trading Bot."""

import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
import logging

logger = logging.getLogger("quantlab")


def send_discord_alert(webhook_url, title, fields, color=65280, description=""):
    """Dispatches a structured embed message to a Discord webhook."""
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
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ],
    }
    data_encoded = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        webhook_url,
        data=data_encoded,
        headers={"Content-Type": "application/json", "User-Agent": "QuantLabBot/1.0"},
        method="POST",
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
        {"name": "Timeframe", "value": config.get("timeframe", "1h"), "inline": True},
    ]

    from bot.positions import render_trade_progress

    if open_pos:
        pos_bullets = []
        for sym, pos in open_pos.items():
            curr_p = pos.get("curr_price", pos["entry_price"])
            tp1_v = pos.get("take_profit_1", pos.get("take_profit", curr_p * 1.02))
            sl_v = pos.get("stop_loss", curr_p * 0.98)
            g = render_trade_progress(curr_p, pos["entry_price"], tp1_v, sl_v)
            pos_bullets.append(
                f"• **{sym}**: Size ${pos['allocated_usdt']:,.2f} | Entry: ${pos['entry_price']:.2f} | Now: ${curr_p:.2f} (PnL: ${pos.get('unrealized_pnl', 0.0):+,.2f} / {pos.get('unrealized_pnl_pct', 0.0):+.2f}%)\n"
                f"  Target: {g}"
            )
        desc = "Current active positions:\n" + "\n".join(pos_bullets)
    else:
        desc = "Portfolio in 100% Cash Defense (No active open risk)."

    bot_mod = sys.modules.get("bot")
    alert_fn = getattr(bot_mod, "send_discord_alert", send_discord_alert) if bot_mod else send_discord_alert
    ok, res = alert_fn(webhook, "📊 DAILY PORTFOLIO PERFORMANCE DIGEST", fields, color=3447003, description=desc)
    if ok:
        state["last_digest_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        from bot.state import save_state
        save_state(state, config["paths"]["state_file"])
    return ok, res
