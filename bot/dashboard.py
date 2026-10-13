"""Terminal CLI Dashboard renderer for QuantLab Trading Bot."""

import os


def print_dashboard(analysis_rows, new_signals, closed_signals, state, config, cb_status):
    """Renders real-time interactive terminal dashboard displaying metrics, scans, and positions."""
    os.system("cls" if os.name == "nt" else "clear")
    balance = state["balance_usdt"]
    open_pos = state.get("open_positions", {})
    mode = config.get("trading_mode", "paper").upper()

    total_unrealized = sum(p.get("unrealized_pnl", 0.0) for p in open_pos.values())
    total_alloc = sum(p.get("allocated_usdt", 0.0) for p in open_pos.values())
    total_equity = balance + total_alloc + total_unrealized

    W = 90
    print("=" * W)
    print(f"  BINANCE 350-COIN MOMENTUM SURFER  |  24/7 DYNAMIC SCAN  |  MODE: {mode}  |  TF: {config['timeframe']}")
    print("=" * W)
    win_cnt = state.get("win_count", 0)
    total_cnt = state.get("trade_count", 0)
    wr = (win_cnt / total_cnt * 100) if total_cnt > 0 else 0.0

    print(f"  Equity (MtM): ${total_equity:,.2f} USDT  |  Free: ${balance:,.2f}  |  Alloc: ${total_alloc:,.2f}  |  Unr: ${total_unrealized:+,.2f}")
    print(f"  Positions: {len(open_pos)}/{config['risk_management']['max_open_trades']}  |  Engine State: {cb_status}  |  Win Rate: {wr:.1f}% ({win_cnt}/{total_cnt})")

    # BNB Fee Discount & Shariah Status
    bnb_discount = config.get("risk_management", {}).get("bnb_fee_discount", True)
    fee_tag = "0.075% (BNB -25% Active)" if bnb_discount else f"{config.get('risk_management', {}).get('fee_rate', 0.001)*100:.3f}%"
    print(f"  Shariah: 100% Spot Asset Ownership (*Qabd*)  |  Fee Tier: VIP0 ({fee_tag})")

    macro_regime = state.get("macro_regime")
    if macro_regime:
        print(f"  Regime: {macro_regime}")
    grid_st = state.get("grid_state")
    if grid_st:
        active_syms = grid_st.get("active_symbols") or [grid_st.get("active_symbol", "SOL/USDT")]
        syms_display = ", ".join(active_syms) if (isinstance(active_syms, list) and active_syms) else "Scanning Basket"
        print(f"  Active Scalp(s): {syms_display}  |  Booked Profit: +${grid_st.get('total_grid_profit_usdt', 0.0):.2f} USDT (Cycles: {grid_st.get('cycles_completed', 0)})")
    print("-" * W)

    # Market Radar Table with Strategy Setups
    print(f"  {'Symbol':<10} {'Price':>11} {'AI Win%':>9} {'Volatility / Setup':<32} {'Status':<16}")
    print("  " + "-" * (W - 4))
    for r in analysis_rows:
        conv_tag = "[SIGNAL TRIGGER]" if "CONFIRMED" in r["status"] else f"[{r['status']}]"
        setups_str = ", ".join(r.get("active_setups", [])) if r.get("active_setups") else "Consolidating"
        if len(setups_str) > 30:
            setups_str = setups_str[:27] + "..."
        price_str = f"${r['price']:,.4f}" if r['price'] > 0 else "Scanning..."
        print(f"  {r['symbol']:<10} {price_str:>11} {r['prob_win']:>8.1%} {setups_str:<32} {conv_tag:<16}")

    # Active Holdings
    print("\n" + "-" * W)
    print("  ACTIVE SPOT HOLDINGS & REAL-TIME MARK-TO-MARKET")
    print("-" * W)
    if not open_pos:
        print("  🛡️ 100% Cash Defense. Scanning for confirmed oversold reversal bounce...")
    else:
        sym_counts = {}
        for pos_id, p in open_pos.items():
            sym = p.get("symbol", "ASSET")
            sym_counts[sym] = sym_counts.get(sym, 0) + 1
            lvl_num = sym_counts[sym]
            entry_p = p["entry_price"]
            curr_p = p.get("curr_price", entry_p)
            unr_pnl = p.get("unrealized_pnl", 0.0)
            unr_pct = p.get("unrealized_pnl_pct", 0.0)
            tp_val = p.get("take_profit", entry_p * 1.015)
            
            is_trailing = p.get("trailing_active", False)
            if is_trailing:
                peak = p.get("peak_price", curr_p)
                status_badge = f"🚀 TRAILING (Peak: ${peak:,.2f})"
            else:
                status_badge = f"TP: ${tp_val:,.4f}"

            label = f"{sym} (L{lvl_num})"
            print(f"  [HOLD] {label:<14} Entry: ${entry_p:<8.4f} | Now: ${curr_p:<8.4f} | "
                  f"PnL: ${unr_pnl:+6.2f} ({unr_pct:+5.2f}%) | {status_badge}")

    # Closed signals & events
    if closed_signals:
        print("\n" + "-" * W)
        print("  RECENT CLOSED TRADES & HARVEST EVENTS")
        print("-" * W)
        for cs in closed_signals:
            tag = "[+WIN]" if cs.get("outcome") == "WIN" else "[-LOSS]"
            pid = cs.get("position_id", "")
            pid_str = f" [{pid[:6]}]" if pid else ""
            print(f"  {tag}{pid_str} {cs['symbol']} Booked PnL: ${cs['net_pnl']:+,.2f} ({cs['pnl_pct']:+.2f}%) | {cs['reason']}")

    # New entries
    if new_signals:
        print("\n" + "-" * W)
        print("  NEW EXECUTED ORDERS")
        print("-" * W)
        for ns in new_signals:
            strat_info = ns.get("strategy_setups", "Halal Grid")
            print(f"  [BUY] {ns['symbol']} @ ${ns['price']:,.4f} | Size: ${ns['size_usdt']} USDT | TP: ${ns.get('tp2', 0.0):,.4f} | {strat_info}")

    print("=" * W + "\n")
