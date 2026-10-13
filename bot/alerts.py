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

    ledger_path = config.get("paths", {}).get("trade_ledger", "logs/trade_history.csv")
    if os.path.exists(ledger_path) and os.path.getsize(ledger_path) > 0:
        try:
            from bot.stats import compute_ledger_stats
            l_stats = compute_ledger_stats(ledger_path)
            if l_stats and l_stats.get("closed_positions", 0) > 0:
                pf_display = f"{l_stats['profit_factor']:.2f}" if isinstance(l_stats['profit_factor'], (int, float)) else str(l_stats['profit_factor'])
                fields.append({"name": "Trade Expectancy", "value": f"**${l_stats['expectancy']:+,.2f} USDT**", "inline": True})
                fields.append({"name": "Profit Factor", "value": f"**{pf_display}**", "inline": True})
                fields.append({"name": "BE Win Rate", "value": f"**{l_stats['break_even_win_rate_pct']:.1f}%**", "inline": True})
        except Exception:
            pass

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


def send_radar_status_alert(state, config):
    """Dispatches a real-time status embed of current active positions and progress to Discord."""
    discord_cfg = config.get("discord", {})
    if not discord_cfg.get("enabled", False):
        return False, "Discord alerts disabled."
    webhook = os.getenv("DISCORD_WEBHOOK_URL", discord_cfg.get("webhook_url", "")).strip()
    if not webhook:
        return False, "No webhook configured."

    from bot.positions import render_trade_progress

    total_equity = state["balance_usdt"]
    active_str = ""
    open_pos = state.get("open_positions", {})
    if open_pos:
        pos_lines = []
        sym_counters = {}
        for pos_id, p in open_pos.items():
            sym = p.get("symbol", "ASSET")
            sym_counters[sym] = sym_counters.get(sym, 0) + 1
            lvl_idx = sym_counters[sym]
            c_p = p.get("curr_price", p["entry_price"])
            u_pnl = p.get("unrealized_pnl", 0.0)
            u_pct = p.get("unrealized_pnl_pct", 0.0)
            total_equity += (p["allocated_usdt"] + u_pnl)
            tp_target = p.get("take_profit", p["entry_price"] * 1.015)
            sl_val = p.get("stop_loss", p["entry_price"] * 0.95)
            gauge = render_trade_progress(c_p, p["entry_price"], tp_target, sl_val)
            trailing_str = f" | 🚀 *Trailing Peak: ${p.get('peak_price', c_p):,.4f}*" if p.get("trailing_active") else ""
            pos_lines.append(
                f"• **{sym} (Level {lvl_idx})**: Live `${c_p:,.4f}` (Entry `${p['entry_price']:,.4f}`) | PnL: `${u_pnl:+.2f}` ({u_pct:+.2f}%){trailing_str}\n"
                f"  🎯 Target: {gauge}"
            )
        active_str = "\n".join(pos_lines)
    else:
        active_str = "🛡️ 100% Cash Defense (Waiting for next value dip to buy)"

    grid_st = state.get("grid_state", {})
    active_syms = grid_st.get("active_symbols") or [grid_st.get("active_symbol", "NEAR/USDT")]
    syms_display = ", ".join(active_syms) if isinstance(active_syms, list) else str(active_syms)

    total_trades = state.get("trade_count", 0)
    wins = state.get("win_count", 0)
    wr = (wins / total_trades * 100) if total_trades > 0 else 0.0

    bnb_discount = config.get("risk_management", {}).get("bnb_fee_discount", True)
    fee_tag = "0.075% (BNB -25% applied)" if bnb_discount else "0.100%"

    fields = [
        {"name": "💼 Total Portfolio Equity", "value": f"**${total_equity:,.2f} USDT** (Free Cash: ${state['balance_usdt']:,.2f})", "inline": False},
        {"name": "🎯 Active Grid Pair(s)", "value": f"**{syms_display}**", "inline": True},
        {"name": "💰 Booked Grid Profit", "value": f"**+${grid_st.get('total_grid_profit_usdt', 0.0):.2f} USDT** ({wins} Wins | {wr:.0f}% WR)", "inline": True},
        {"name": "📊 Active Holdings & Real-Time Target Gauge", "value": active_str, "inline": False},
        {"name": "💎 Capital Efficiency", "value": f"VIP0 Spot Fee: **{fee_tag}** | 100% Halal Spot Asset Ownership (*Qabd*)", "inline": False},
        {"name": "⚙️ Bot Health", "value": "🟢 Online 24/7 | Checking order book every cycle", "inline": False},
    ]
    title = "📡 Live Halal Grid Status & Target Progress"
    return send_discord_alert(webhook, title, fields, color=3447003)

