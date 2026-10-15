import sys
import os
import requests
import time
import numpy as np
from datetime import datetime, timezone

sys.path.insert(0, "f:/Random/WORK/quantlab")
from sol_scalper.backtest import fetch_historical_candles, calc_ema, calc_rsi_series

def run_unified_portfolio_backtest():
    print("=" * 85)
    print("  🔬 UNIFIED PORTFOLIO BACKTEST: MAX 1 CONCURRENT POSITION (ZERO CORRELATION RISK)")
    print("  Timeframe: 15M  |  Basket: PYTH + RAY + SOL  |  Capital: $50.00 USD")
    print("  Targets: Take Profit: +5.00%  |  Stop Loss: -2.00%  |  DEX Fee/Slippage: 0.20% Round-Trip")
    print("=" * 85)

    symbols = ["PYTHUSDT", "RAYUSDT", "SOLUSDT"]
    candle_map = {}
    
    print("  Fetching 3,000 historical 15m candles (~31 days) for all basket assets...")
    for sym in symbols:
        c = fetch_historical_candles(sym, interval="15m", total_candles=3000)
        if not c or len(c) < 100:
            print(f"  ❌ Failed to fetch candles for {sym}")
            return
        candle_map[sym] = c
        print(f"  ✅ {sym:<10}: {len(c):,} candles fetched.")

    # Align candles by timestamp
    # Find common timestamps
    ts_sets = [set(c["timestamp"] for c in candle_map[s]) for s in symbols]
    common_ts = sorted(list(set.intersection(*ts_sets)))
    print(f"  Synced {len(common_ts):,} chronological bars across all 3 assets.")
    print("-" * 85)

    # Precalculate indicators per symbol on aligned timeframes
    indicator_data = {}
    for sym in symbols:
        c_dict = {c["timestamp"]: c for c in candle_map[sym]}
        aligned_candles = [c_dict[ts] for ts in common_ts]
        closes = np.array([c["close"] for c in aligned_candles], dtype=float)
        opens = np.array([c["open"] for c in aligned_candles], dtype=float)
        highs = np.array([c["high"] for c in aligned_candles], dtype=float)
        lows = np.array([c["low"] for c in aligned_candles], dtype=float)

        ema9 = calc_ema(closes, 9)
        ema21 = calc_ema(closes, 21)
        ema50 = calc_ema(closes, 50)
        rsi = calc_rsi_series(closes, 14)

        indicator_data[sym] = {
            "closes": closes,
            "opens": opens,
            "highs": highs,
            "lows": lows,
            "ema9": ema9,
            "ema21": ema21,
            "ema50": ema50,
            "rsi": rsi,
        }

    # Simulation State
    cash = 50.0
    starting_usd = 50.0
    active_pos = None  # Exactly MAX 1 POSITION AT ALL TIMES
    trades = []
    equity_curve = [50.0]

    tp_pct = 5.0
    sl_pct = 2.0
    fee_pct = 0.002  # 0.20% round-trip fee + slippage
    bet_pct = 0.70   # 70% of cash allocated to the single position

    # Loop bar-by-bar chronologically
    for i in range(50, len(common_ts)):
        ts = common_ts[i]
        dt_str = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")

        # 1. Manage Active Position (if one is open)
        if active_pos:
            sym = active_pos["symbol"]
            d = indicator_data[sym]
            curr_h = d["highs"][i]
            curr_l = d["lows"][i]
            curr_c = d["closes"][i]
            entry_p = active_pos["entry_price"]
            allocated = active_pos["allocated_usd"]

            # Update peak price for trailing stop
            peak_p = max(active_pos.get("peak_price", entry_p), curr_h)
            active_pos["peak_price"] = peak_p

            gain_at_high = (curr_h - entry_p) / entry_p * 100.0
            loss_at_low = (curr_l - entry_p) / entry_p * 100.0
            peak_gain = (peak_p - entry_p) / entry_p * 100.0

            should_exit = False
            exit_price = curr_c
            outcome = "HOLD"
            reason = ""

            # Rule A: Take Profit Hit (+5.0%)
            if gain_at_high >= tp_pct:
                should_exit = True
                exit_price = entry_p * (1.0 + tp_pct / 100.0)
                outcome = "WIN"
                reason = "🎯 TP +5.00% Hit"

            # Rule B: Trailing Stop Lock (Trigger at +2.5%, trail by 1.0%)
            elif peak_gain >= 2.50 and curr_l <= (peak_p * 0.990):
                should_exit = True
                exit_price = peak_p * 0.990
                outcome = "WIN" if exit_price > entry_p else "LOSS"
                reason = f"🚀 Trailing Lock (+{(exit_price - entry_p)/entry_p*100:.2f}%)"

            # Rule C: Stop Loss Cut (-2.0%)
            elif loss_at_low <= -sl_pct:
                should_exit = True
                exit_price = entry_p * (1.0 - sl_pct / 100.0)
                outcome = "LOSS"
                reason = "🛡️ Hard Stop -2.00%"

            # Rule D: Stale Cut (24 bars = 6 hours)
            elif (i - active_pos["entry_idx"]) >= 24:
                should_exit = True
                exit_price = curr_c
                outcome = "WIN" if exit_price > entry_p else "LOSS"
                reason = f"⏳ Timeout (6h) ({ (exit_price - entry_p)/entry_p*100:+.2f}%)"

            if should_exit:
                gross_proceeds = allocated * (exit_price / entry_p)
                fee = gross_proceeds * fee_pct
                net_pnl = gross_proceeds - allocated - fee
                cash += (allocated + net_pnl)
                trades.append({
                    "date": dt_str,
                    "symbol": sym,
                    "entry": entry_p,
                    "exit": exit_price,
                    "outcome": outcome,
                    "return_pct": (exit_price - entry_p) / entry_p * 100.0,
                    "pnl_usd": net_pnl,
                    "balance": cash,
                    "reason": reason
                })
                active_pos = None

        # 2. Look for New Entry across Basket (ONLY IF 0 POSITIONS OPEN)
        if active_pos is None and cash >= 15.0:
            for sym in symbols:
                d = indicator_data[sym]
                c_now = d["closes"][i]
                o_now = d["opens"][i]
                c_prev = d["closes"][i-1]
                o_prev = d["opens"][i-1]
                l_prev = d["lows"][i-1]
                rsi_now = d["rsi"][i]
                rsi_prev = d["rsi"][i-1]
                ema9_now = d["ema9"][i]
                ema21_now = d["ema21"][i]
                ema50_now = d["ema50"][i]

                # Rule 1: Macro Trend (EMA 9 > 21 > 50 and Price > 50)
                is_uptrend = (ema9_now > ema21_now > ema50_now) and (c_now > ema50_now)

                # Rule 2: Support Touch / RSI Reset on Bar i-1
                dip_to_support = (l_prev <= ema21_now * 1.002) or (rsi_prev <= 48.0)

                # Rule 3: Confirmed Green Reversal Candle on Bar i
                green_reversal = (c_now > o_now) and (c_now > c_prev) and (c_now >= ema9_now * 0.999)

                # Rule 4: Not Overextended (RSI between 38 and 58)
                rsi_valid = (rsi_now <= 58.0) and (rsi_now >= 38.0)

                if is_uptrend and dip_to_support and green_reversal and rsi_valid:
                    pos_size = round(cash * bet_pct, 2)
                    cash -= pos_size
                    active_pos = {
                        "symbol": sym,
                        "entry_price": c_now,
                        "peak_price": c_now,
                        "allocated_usd": pos_size,
                        "entry_idx": i,
                        "entry_time": dt_str
                    }
                    # Break out so only 1 asset is entered!
                    break

        equity_curve.append(cash + (active_pos["allocated_usd"] if active_pos else 0.0))

    # Compile Final Report
    wins = [t for t in trades if t["pnl_usd"] > 0]
    losses = [t for t in trades if t["pnl_usd"] <= 0]
    win_rate = (len(wins) / len(trades) * 100.0) if trades else 0.0
    total_net_pnl = cash - starting_usd
    total_return_pct = (cash - starting_usd) / starting_usd * 100.0
    gross_profit = sum(t["pnl_usd"] for t in wins)
    gross_loss = abs(sum(t["pnl_usd"] for t in losses))
    profit_factor = (gross_profit / max(0.01, gross_loss)) if losses else 99.0

    print("=" * 85)
    print("  🏆 UNIFIED SINGLE-POSITION PORTFOLIO BACKTEST SUMMARY")
    print("=" * 85)
    print(f"  Starting Balance   : ${starting_usd:.2f} USD")
    print(f"  Ending Balance     : ${cash:.2f} USD")
    print(f"  Total Net Return   : {total_net_pnl:+.2f} USD ({total_return_pct:+.2f}%)")
    print(f"  Total Closed Trades: {len(trades)} trades (~1 trade per day)")
    print(f"  Win / Loss Record  : {len(wins)} Wins / {len(losses)} Losses")
    print(f"  Win Rate           : {win_rate:.1f}%")
    print(f"  Profit Factor      : {profit_factor:.2f}")
    print("-" * 85)
    print("  LAST 10 TRADES LEDGER (CHRONOLOGICAL):")
    print(f"  {'Date':<17} {'Symbol':<10} {'Entry':>10} {'Exit':>10} {'Return':>9} {'PnL (USD)':>10} {'Outcome':<8} {'Reason'}")
    print("  " + "-" * 81)
    for t in trades[-10:]:
        print(
            f"  {t['date']:<17} {t['symbol']:<10} ${t['entry']:>9.4f} ${t['exit']:>9.4f} "
            f"{t['return_pct']:>+8.2f}% {t['pnl_usd']:>+9.2f}  {t['outcome']:<8} {t['reason']}"
        )
    print("=" * 85)

if __name__ == "__main__":
    run_unified_portfolio_backtest()
