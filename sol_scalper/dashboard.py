"""Real-time Multi-Asset Interactive CLI Terminal Dashboard for JUP + SOL Quantitative Scalper."""

import os
import sys
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


def print_scalper_dashboard(
    snapshots: Dict[str, Dict[str, Any]], 
    state: Dict[str, Any], 
    config: Dict[str, Any], 
    recent_events: List[Dict[str, Any]] = None,
    btc_regime: Optional[Dict[str, Any]] = None
):
    if sys.stdout and hasattr(sys.stdout, "isatty") and sys.stdout.isatty():
        os.system("cls" if os.name == "nt" else "clear")
    W = 92
    mode = config.get("mode", "paper").upper()
    cash = state.get("cash_usd", 50.0)
    open_pos = state.get("open_positions", {})

    # Calculate active position equity
    pos_val = 0.0
    for sym, p in open_pos.items():
        curr_p = snapshots.get(sym, {}).get("price", p["entry_price_usd"])
        pos_val += p["allocated_usd"] * (curr_p / p["entry_price_usd"])

    equity = cash + pos_val
    trades = state.get("trade_count", 0)
    wins = state.get("win_count", 0)
    losses = state.get("loss_count", 0)
    wr = (wins / trades * 100.0) if trades > 0 else 0.0
    pnl = state.get("realized_pnl_usd", 0.0)

    strat_cfg = config.get("strategy", {})
    tf = strat_cfg.get("timeframe", "15m").upper()
    tokens_str = " + ".join([s.replace("USDC", "") for s in snapshots.keys()]) or "MULTI-ASSET"

    print("=" * W)
    print(f"  ⚡ QUANTITATIVE HALAL SCALPER ({tokens_str})  |  TIMEFRAME: {tf}  |  MODE: {mode}")
    print("=" * W)
    print(
        f"  Total Equity: ${equity:,.2f}  |  Liquid Cash: ${cash:,.2f}  |  In Positions: ${pos_val:,.2f}\n"
        f"  Realized PnL: {pnl:+,.2f} USD  |  Win Rate: {wr:.1f}% ({wins}W / {losses}L)  |  Trades: {trades}"
    )

    # BTC Macro Gate Banner
    if btc_regime:
        gate_status = "🟢 ACTIVE / PERMITTED" if btc_regime.get("is_safe") else "🔴 HALTED / BLOCKED"
        btc_p = btc_regime.get("price", 0.0)
        btc_rsi = btc_regime.get("rsi", 50.0)
        print(f"  BTC Macro Gate : {gate_status} (BTC: ${btc_p:,.0f} | 1H RSI: {btc_rsi:.1f} | {btc_regime.get('reason', '')})")
    print("-" * W)

    # 1. Multi-Asset Market Radar
    print(f"  {tf} INSTITUTIONAL RADAR (TREND + VOLUME + ATR VOLATILITY)")
    print("-" * W)
    print(f"  {'Asset':<10} {'Price':>11} {'RSI':>7} {'Vol Ratio':>11} {'ATR (14)':>10} {'Trend':<14}")
    print("  " + "-" * (W - 4))
    for sym, snap in snapshots.items():
        regime = "🟢 UPTREND" if snap["is_bullish"] else ("🔴 DOWNTREND" if snap["is_bearish"] else "🟡 CHOP")
        vol_r = snap.get("volume_ratio", 1.0)
        vol_str = f"{vol_r:.2f}x"
        atr_pct = snap.get("atr_pct", 1.5)
        atr_str = f"±{atr_pct:.2f}%"
        print(
            f"  {sym:<10} ${snap['price']:>10.4f} {snap['rsi']:>7.1f} "
            f"{vol_str:>11} {atr_str:>10} {regime:<14}"
        )
    print("-" * W)

    # 2. Open Scalp Positions
    print("  ACTIVE POSITIONS")
    print("-" * W)
    if not open_pos:
        print(f"  🛡️ 100% Cash in USDC. Scanning {tokens_str} for 15m pullback bounce...")
    else:
        for sym, p in open_pos.items():
            curr_p = snapshots.get(sym, {}).get("price", p["entry_price_usd"])
            gain_pct = (curr_p - p["entry_price_usd"]) / p["entry_price_usd"] * 100.0
            cur_val = p["allocated_usd"] * (curr_p / p["entry_price_usd"])
            net_d = cur_val - p["allocated_usd"]
            color = "🟢" if gain_pct >= 0 else "🔴"
            tp_pct = config.get("trading", {}).get("take_profit_pct", 5.0)
            sl_pct = config.get("trading", {}).get("stop_loss_pct", 2.0)
            print(
                f"  {color} [{sym}] Size: ${p['allocated_usd']:.2f} ({p.get('units', 0.0):.2f} units) | "
                f"Entry: ${p['entry_price_usd']:.4f} | Now: ${curr_p:.4f} | Gain: {gain_pct:+.2f}% ({net_d:+.2f} USD)\n"
                f"     🎯 Target TP (+{tp_pct:.1f}%): ${p['target_tp_price']:.4f} | 🛡️ Stop Loss (-{sl_pct:.1f}%): ${p['target_sl_price']:.4f}"
            )
    print("-" * W)

    # 3. Recent Scalp Events
    if recent_events:
        print("  RECENT SCALP EXECUTIONS")
        print("-" * W)
        for ev in recent_events[-3:]:
            pnl_s = f"{ev.get('pnl_usd', 0.0):+.2f} USD" if "pnl_usd" in ev else ""
            print(f"  [{ev.get('action')}] {ev.get('symbol')} {ev.get('outcome', '')} {pnl_s} | {ev.get('reason')}")
        print("-" * W)

    print("=" * W + "\n")


if __name__ == "__main__":
    import json
    from sol_scalper.market_data import compute_basket_snapshots

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
            
    snaps = compute_basket_snapshots(cfg.get("basket", []), cfg.get("strategy", {}))
    print_scalper_dashboard(snaps, st, cfg)
