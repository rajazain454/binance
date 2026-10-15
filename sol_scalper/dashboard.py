"""Real-time Interactive CLI Terminal Dashboard for SOL/USDC Institutional Scalper."""

import os
import sys
from typing import Dict, Any, List

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


def print_scalper_dashboard(
    snapshot: Dict[str, Any], 
    state: Dict[str, Any], 
    config: Dict[str, Any], 
    recent_events: List[Dict[str, Any]] = None
):
    os.system("cls" if os.name == "nt" else "clear")
    W = 88
    mode = config.get("mode", "paper").upper()
    cash = state.get("cash_usd", 50.0)
    pos = state.get("active_position")

    curr_price = snapshot["price"] if snapshot else 0.0
    pos_val = (pos["allocated_usd"] * (curr_price / pos["entry_price_usd"])) if pos and curr_price > 0 else 0.0
    equity = cash + pos_val

    trades = state.get("trade_count", 0)
    wins = state.get("win_count", 0)
    losses = state.get("loss_count", 0)
    wr = (wins / trades * 100.0) if trades > 0 else 0.0
    pnl = state.get("realized_pnl_usd", 0.0)

    print("=" * W)
    print(f"  ⚡ SOL/USDC QUANTITATIVE SYSTEMATIC SCALPER  |  MODE: {mode}  |  STATUS: ACTIVE")
    print("=" * W)
    print(
        f"  Total Equity: ${equity:,.2f}  |  Liquid Cash: ${cash:,.2f}  |  In Position: ${pos_val:,.2f}\n"
        f"  Realized PnL: {pnl:+,.2f} USD  |  Win Rate: {wr:.1f}% ({wins}W / {losses}L)  |  Trades: {trades}"
    )
    print("-" * W)

    # 1. Market Technicals
    if snapshot:
        trend_badge = "🟢 " + snapshot["trend_state"] if snapshot["is_bullish_trend"] else ("🔴 " + snapshot["trend_state"] if snapshot["is_bearish_trend"] else "🟡 " + snapshot["trend_state"])
        print("  MARKET INTELLIGENCE (1-MINUTE REAL-TIME FEED)")
        print("-" * W)
        print(
            f"  SOL Price: ${snapshot['price']:,.2f}   |   "
            f"RSI (14): {snapshot['rsi']:.1f}   |   "
            f"ATR: ${snapshot['atr']:.2f}\n"
            f"  EMA 9: ${snapshot['ema_fast']:.2f}   |   "
            f"EMA 21: ${snapshot['ema_slow']:.2f}   |   "
            f"EMA 50: ${snapshot['ema_trend']:.2f}\n"
            f"  Regime: {trend_badge}"
        )
    print("-" * W)

    # 2. Position Status
    print("  CURRENT POSITION")
    print("-" * W)
    if not pos:
        print("  🛡️ 100% Cash in USDC. Scanning 1m/5m chart for oversold pullback reversal...")
    else:
        gain_pct = (curr_price - pos["entry_price_usd"]) / pos["entry_price_usd"] * 100.0
        pnl_usd = pos_val - pos["allocated_usd"]
        status_color = "🟢" if gain_pct >= 0 else "🔴"
        print(
            f"  {status_color} Asset: SOL/USDC  |  Size: {pos['sol_amount']:.4f} SOL (${pos['allocated_usd']:.2f})\n"
            f"  Entry: ${pos['entry_price_usd']:.2f}  |  Now: ${curr_price:.2f}  |  Gain: {gain_pct:+.2f}% ({pnl_usd:+.2f} USD)\n"
            f"  🎯 Target TP (+1.8%): ${pos['target_tp_price']:.2f}  |  🛡️ Stop Loss (-0.9%): ${pos['target_sl_price']:.2f}\n"
            f"  Peak Price: ${pos.get('peak_price_usd', pos['entry_price_usd']):.2f}"
        )
    print("-" * W)

    # 3. Recent Scalp Events
    if recent_events:
        print("  RECENT SCALP EXECUTIONS")
        print("-" * W)
        for ev in recent_events[-3:]:
            pnl_str = f"{ev.get('pnl_usd', 0.0):+.2f} USD" if "pnl_usd" in ev else ""
            print(f"  [{ev.get('action')}] {ev.get('outcome', '')} {pnl_str} | {ev.get('reason')}")
        print("-" * W)

    print("=" * W + "\n")


if __name__ == "__main__":
    import json
    from sol_scalper.market_data import compute_market_snapshot

    cfg_path = "sol_scalper/config.json"
    st_path = "sol_scalper/state.json"
    
    cfg = {}
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            
    st = {}
    if os.path.exists(st_path):
        with open(st_path, "r", encoding="utf-8") as f:
            st = json.load(f)
            
    snap = compute_market_snapshot(cfg.get("strategy", {}))
    print_scalper_dashboard(snap, st, cfg)
