"""Terminal CLI Dashboard renderer for QuantLab Trading Bot."""

import os


def fmt_price(p: float) -> str:
    """Formats prices with appropriate precision across micro-cap memecoins and macro assets."""
    if p <= 0:
        return "Scanning..."
    if p >= 100:
        return f"${p:,.2f}"
    elif p >= 1:
        return f"${p:,.4f}"
    elif p >= 0.001:
        return f"${p:.6f}"
    else:
        return f"${p:.8f}"


def print_dashboard(analysis_rows, new_signals, closed_signals, state, config, cb_status):
    """Renders real-time interactive terminal dashboard displaying metrics, scans, and positions."""
    os.system("cls" if os.name == "nt" else "clear")
    balance = state["balance_usdt"]
    open_pos = state.get("open_positions", {})
    mode = config.get("trading_mode", "paper").upper()

    total_unrealized = sum(p.get("unrealized_pnl", 0.0) for p in open_pos.values())
    total_alloc = sum(p.get("allocated_usdt", 0.0) for p in open_pos.values())
    total_equity = balance + total_alloc + total_unrealized

    W = 92
    strat_name = config.get("grid_trading", {}).get("strategy_name", "BINANCE COMPOUNDING SCALPER").upper()
    print("=" * W)
    print(f"  {strat_name}  |  MODE: {mode}  |  TF: {config['timeframe']}")
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
        active_syms = grid_st.get("active_symbols") or [grid_st.get("active_symbol", "PEPE/USDT")]
        syms_display = ", ".join(active_syms) if (isinstance(active_syms, list) and active_syms) else "Scanning Basket"
        print(f"  Active Scalp(s): {syms_display}  |  Booked Profit: +${grid_st.get('total_grid_profit_usdt', 0.0):.2f} USDT (Cycles: {grid_st.get('cycles_completed', 0)})")
    print("-" * W)

    # Market Radar Table with Strategy Setups
    print(f"  {'Symbol':<12} {'Price':>13} {'AI Win%':>9} {'Volatility / Setup':<30} {'Status':<16}")
    print("  " + "-" * (W - 4))
    for r in analysis_rows:
        conv_tag = "[SIGNAL TRIGGER]" if "CONFIRMED" in r["status"] else f"[{r['status']}]"
        setups_str = ", ".join(r.get("active_setups", [])) if r.get("active_setups") else "Consolidating"
        if len(setups_str) > 28:
            setups_str = setups_str[:25] + "..."
        price_str = fmt_price(r['price'])
        print(f"  {r['symbol']:<12} {price_str:>13} {r['prob_win']:>8.1%} {setups_str:<30} {conv_tag:<16}")

    # Active Holdings
    print("\n" + "-" * W)
    print("  ACTIVE SPOT HOLDINGS & REAL-TIME MARK-TO-MARKET")
    print("-" * W)
    if not open_pos:
        print("  🛡️ 100% Cash Defense. Scanning for high-velocity memecoin breakout...")
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
            tp_val = p.get("take_profit", entry_p * 1.045)
            
            is_trailing = p.get("trailing_active", False)
            is_partial_taken = p.get("partial_tp_taken", False)
            if is_trailing:
                peak = p.get("peak_price", curr_p)
                status_badge = f"🚀 TRAILING RUNNER (50% Locked | Peak: {fmt_price(peak)})" if is_partial_taken else f"🚀 TRAILING (Peak: {fmt_price(peak)})"
            else:
                tp1_val = p.get("tp1_price")
                tp1_str = f"TP1: {fmt_price(tp1_val)} | " if tp1_val else ""
                status_badge = f"{tp1_str}TP2: {fmt_price(tp_val)}"

            label = f"{sym} (L{lvl_num})"
            print(f"  [HOLD] {label:<14} Entry: {fmt_price(entry_p):<11} | Now: {fmt_price(curr_p):<11} | "
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
            strat_info = ns.get("strategy_setups", "Memecoin Scalp")
            print(f"  [BUY] {ns['symbol']} @ {fmt_price(ns['price'])} | Size: ${ns['size_usdt']:.2f} USDT | TP: {fmt_price(ns.get('tp2', 0.0))} | {strat_info}")

    print("=" * W + "\n")
