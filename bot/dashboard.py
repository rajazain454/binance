"""Terminal CLI Dashboard renderer for QuantLab Trading Bot."""

import os


def print_dashboard(analysis_rows, new_signals, closed_signals, state, config, cb_status):
    """Renders real-time interactive terminal dashboard displaying metrics, scans, and positions."""
    os.system("cls" if os.name == "nt" else "clear")
    balance = state["balance_usdt"]
    open_pos = state.get("open_positions", {})
    mode = config.get("trading_mode", "paper").upper()
    use_mtf = config.get("mtf_confluence", {}).get("enabled", True)

    total_unrealized = sum(p.get("unrealized_pnl", 0.0) for p in open_pos.values())
    total_alloc = sum(p.get("allocated_usdt", 0.0) for p in open_pos.values())
    total_equity = balance + total_alloc + total_unrealized

    print("=" * 108)
    print(f"  BINANCE QUANTITATIVE AI TRADING BOT  |  MODE: {mode}  |  TF: {config['timeframe']} (4H MTF: {'ON' if use_mtf else 'OFF'})")
    print("=" * 108)
    win_cnt = state.get("win_count", 0)
    total_cnt = state.get("trade_count", 0)
    wr = (win_cnt / total_cnt * 100) if total_cnt > 0 else 0.0

    print(f"  Equity (MtM): ${total_equity:,.2f} USDT  |  Free Cash: ${balance:,.2f}  |  Active Alloc: ${total_alloc:,.2f}  |  Unrealized: ${total_unrealized:+,.2f}")
    print(f"  Positions: {len(open_pos)}/{config['risk_management']['max_open_trades']}  |  Circuit Breaker: {cb_status}  |  Total Trades: {total_cnt}  |  Win Rate: {wr:.1f}%")
    print("-" * 108)

    # Market Radar Table with Strategy Setups
    print(f"{'Symbol':<10} {'Price':>10} {'AI Win Prob':>12} {'Strategy Setup':<34} {'4h Trend':>10} {'Status':<15}")
    print("-" * 108)
    for r in analysis_rows:
        conv_tag = "[BUY TRIGGER]" if r["status"] == "BUY TRIGGER" else (r["status"])
        setups_str = ", ".join(r.get("active_setups", [])) if r.get("active_setups") else "Consolidating"
        if len(setups_str) > 32:
            setups_str = setups_str[:29] + "..."
        print(f"{r['symbol']:<10} {r['price']:>10.4f} {r['prob_win']:>11.1%} {setups_str:<34} "
              f"{r['mtf_status']:>10} {conv_tag:<15}")

    # Active Holdings
    print("\n" + "-" * 108)
    print("  ACTIVE OPEN POSITIONS & LIVE MARK-TO-MARKET PERFORMANCE")
    print("-" * 108)
    if not open_pos:
        print("  No active positions. Scanning Binance market for high-probability setups...")
    else:
        for sym, p in open_pos.items():
            entry_p = p["entry_price"]
            curr_p = p.get("curr_price", entry_p)
            unr_pnl = p.get("unrealized_pnl", 0.0)
            unr_pct = p.get("unrealized_pnl_pct", 0.0)
            tp1_val = p.get("take_profit_1") or p.get("take_profit") or (entry_p * 1.015)
            tp2_val = p.get("take_profit_2") or (tp1_val * 1.015)
            tp1_status = "LOCKED [BREAKEVEN]" if p.get("tp1_reached") else f"${tp1_val:.4f}"
            print(f"  [HOLD] {sym:<9} Entry: ${entry_p:<9.4f} | Now: ${curr_p:<9.4f} | Unr: ${unr_pnl:+7.2f} ({unr_pct:+6.2f}%) | "
                  f"Stop: ${p['stop_loss']:<9.4f} | TP1: {tp1_status:<17} | TP2: ${tp2_val:<9.4f}")

    # Closed signals
    if closed_signals:
        print("\n" + "-" * 88)
        print("  RECENT CLOSED TRADES")
        print("-" * 88)
        for cs in closed_signals:
            action = cs.get("action", "")
            if action == "BREAKEVEN LOCK (SMALL ACCOUNT)":
                tag = "[🛡️ BE-LOCK]"
            elif cs["outcome"] == "WIN":
                tag = "[+WIN]"
            else:
                tag = "[-LOSS]"
            print(f"  {tag} {cs['symbol']} Net PnL: ${cs['net_pnl']:+,.2f} ({cs['pnl_pct']:+.2f}%) | {cs['reason']}")

    # New entries
    if new_signals:
        print("\n" + "-" * 88)
        print("  NEW EXECUTED ORDERS")
        print("-" * 88)
        for ns in new_signals:
            strat_info = ns.get("strategy_setups", "Quant Confluence")
            print(f"  [NEW ORDER] {ns['action']} {ns['symbol']} @ ${ns['price']:.4f} | Size: ${ns['size_usdt']} USDT | Conf: {ns['confidence_pct']}% | Strategy: {strat_info}")

    print("=" * 88 + "\n")
