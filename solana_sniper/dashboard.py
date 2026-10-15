"""Real-time Interactive CLI Terminal Dashboard for Solana Meme Sniper."""

import os
from typing import List, Dict, Any


def fmt_p(p: float) -> str:
    if p <= 0:
        return "$0.00"
    elif p >= 1.0:
        return f"${p:,.4f}"
    elif p >= 0.0001:
        return f"${p:.6f}"
    else:
        return f"${p:.8f}"


def print_sniper_dashboard(
    candidates: List[Dict[str, Any]], 
    state: Dict[str, Any], 
    config: Dict[str, Any],
    recent_events: List[Dict[str, Any]] = None
):
    os.system("cls" if os.name == "nt" else "clear")
    W = 95
    mode = config.get("mode", "paper").upper()
    bal_sol = state.get("balance_sol", 1.0)
    open_pos = state.get("open_positions", {})
    alloc_sol = sum(p.get("allocated_sol", 0.0) for p in open_pos.values())
    unrealized_sol = sum(p.get("allocated_sol", 0.0) * (p.get("multiplier", 1.0) - 1.0) for p in open_pos.values())
    equity_sol = bal_sol + alloc_sol + unrealized_sol

    # Approximate SOL price in USD ($150 baseline for equity display)
    sol_usd_rate = 150.0

    trade_cnt = state.get("trade_count", 0)
    win_cnt = state.get("win_count", 0)
    loss_cnt = state.get("loss_count", 0)
    wr = (win_cnt / trade_cnt * 100.0) if trade_cnt > 0 else 0.0
    moonbags = state.get("moonbags_harvested", 0)
    realized_sol = state.get("realized_pnl_sol", 0.0)

    print("=" * W)
    print(f"  ⚡ SOLANA PUMP.FUN & RAYDIUM MEME SNIPER  |  MODE: {mode}  |  ENGINE: ACTIVE HUNTING")
    print("=" * W)
    print(
        f"  Balance: {bal_sol:.3f} SOL (~${bal_sol * sol_usd_rate:,.2f})  |  "
        f"Alloc: {alloc_sol:.3f} SOL  |  Unrealized: {unrealized_sol:+.3f} SOL\n"
        f"  Total Equity: {equity_sol:.3f} SOL (~${equity_sol * sol_usd_rate:,.2f})  |  "
        f"Realized PnL: {realized_sol:+.3f} SOL  |  Moonbags (2x+): {moonbags}  |  Win Rate: {wr:.1f}% ({win_cnt}/{trade_cnt})"
    )
    print("-" * W)

    # 1. Active Open Snipes
    print("  ACTIVE SNIPER POSITIONS & REAL-TIME MULTIPLIERS")
    print("-" * W)
    if not open_pos:
        print("  🛡️ 100% Free Capital. Scanner hunting explosive Solana / Pump.fun volume surges...")
    else:
        for mint, p in open_pos.items():
            mult = p.get("multiplier", 1.0)
            status_tag = "🚀 [MOONBAG RUNNER]" if p.get("tp1_taken") else ("[2X TARGET NEAR]" if mult >= 1.5 else "[HUNTING 2X]")
            pnl_sol = p.get("allocated_sol", 0.1) * (mult - 1.0)
            print(
                f"  [{p['symbol']:<8}] Entry: {fmt_p(p['entry_price_usd']):>12} | "
                f"Now: {fmt_p(p.get('current_price_usd', p['entry_price_usd'])):>12} | "
                f"Mult: {mult:>5.2f}x | PnL: {pnl_sol:>+7.3f} SOL | {status_tag}"
            )
    print("-" * W)

    # 2. Live Scanned Radar
    print("  LIVE BREAKOUT RADAR (DEXSCREENER & PUMP.FUN MIGRATIONS)")
    print("-" * W)
    print(f"  {'Symbol':<10} {'Price':>12} {'Liquidity':>12} {'5m Volume':>12} {'Buys/Sells':>12} {'Type':<12}")
    print("  " + "-" * (W - 4))
    
    shown_cand = candidates[:7]
    if not shown_cand:
        print("  Scanning memepools for liquidity + volume spikes...")
    else:
        for c in shown_cand:
            type_str = "🔥 PUMP.FUN" if c.get("is_pump_fun") else "⚡ RAYDIUM"
            ratio_str = f"{c['buys_5m']}/{c['sells_5m']} ({c['buy_sell_ratio']:.1f}x)"
            print(
                f"  {c['symbol']:<10} {fmt_p(c['price_usd']):>12} "
                f"${c['liquidity_usd']:>10,.0f} ${c['volume_5m']:>10,.0f} "
                f"{ratio_str:>12} {type_str:<12}"
            )

    if recent_events:
        print("-" * W)
        print("  RECENT HARVEST & SNIPE EVENTS")
        print("-" * W)
        for ev in recent_events[-3:]:
            print(f"  [{ev.get('action')}] {ev.get('symbol')} | Mult: {ev.get('multiplier', 1.0):.2f}x | PnL: {ev.get('pnl_sol', 0.0):+.3f} SOL | {ev.get('reason')}")

    print("=" * W + "\n")
