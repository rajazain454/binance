import sys
import os
import requests
import time
import numpy as np
from datetime import datetime, timezone

sys.path.insert(0, "f:/Random/WORK/quantlab")
from sol_scalper.backtest import fetch_historical_candles, calc_ema, calc_rsi_series
from sol_scalper.market_data import calculate_atr

def run_unified_portfolio_backtest():
    print("=" * 88)
    print("  🔬 UNIFIED PORTFOLIO BACKTEST: IMPROVEMENTS 1, 2, 3 APPLIED")
    print("  1. BTC Macro Gate (BTC 1H EMA 50 & RSI > 40)")
    print("  2. Adaptive Volatility Targets (ATR-Based TP & SL)")
    print("  3. Volume Confirmation (Reversal Candle Vol >= 1.1x 20-period avg)")
    print("  Basket: PYTH + RAY + SOL  |  Timeframe: 15M  |  Capital: $50.00 USD  |  Max 1 Position")
    print("=" * 88)

    symbols = ["PYTHUSDT", "RAYUSDT", "SOLUSDT"]
    candle_map = {}
    
    print("  Fetching 3,000 historical 15m candles (~31 days) for all basket assets + BTC...")
    for sym in symbols:
        c = fetch_historical_candles(sym, interval="15m", total_candles=3000)
        if not c or len(c) < 100:
            print(f"  ❌ Failed to fetch candles for {sym}")
            return
        candle_map[sym] = c
        print(f"  ✅ {sym:<10}: {len(c):,} candles fetched.")

    btc_candles = fetch_historical_candles("BTCUSDT", interval="15m", total_candles=3000)
    print(f"  ✅ BTCUSDT   : {len(btc_candles):,} candles fetched.")

    # Find common timestamps
    ts_sets = [set(c["timestamp"] for c in candle_map[s]) for s in symbols]
    ts_sets.append(set(c["timestamp"] for c in btc_candles))
    common_ts = sorted(list(set.intersection(*ts_sets)))
    print(f"  Synced {len(common_ts):,} chronological bars across all assets.")
    print("-" * 88)

    # Precalculate BTC Macro Filter
    btc_dict = {c["timestamp"]: c for c in btc_candles}
    btc_aligned = [btc_dict[ts] for ts in common_ts]
    btc_closes = np.array([c["close"] for c in btc_aligned], dtype=float)
    # 1H EMA 50 on 15m bars is approx period 200 (4 * 50 = 200)
    btc_ema200 = calc_ema(btc_closes, 200)
    btc_rsi = calc_rsi_series(btc_closes, 56)

    # Precalculate indicators per symbol on aligned timeframes
    indicator_data = {}
    for sym in symbols:
        c_dict = {c["timestamp"]: c for c in candle_map[sym]}
        aligned_candles = [c_dict[ts] for ts in common_ts]
        closes = np.array([c["close"] for c in aligned_candles], dtype=float)
        opens = np.array([c["open"] for c in aligned_candles], dtype=float)
        highs = np.array([c["high"] for c in aligned_candles], dtype=float)
        lows = np.array([c["low"] for c in aligned_candles], dtype=float)
        volumes = np.array([c["volume"] for c in aligned_candles], dtype=float)

        ema9 = calc_ema(closes, 9)
        ema21 = calc_ema(closes, 21)
        ema50 = calc_ema(closes, 50)
        rsi = calc_rsi_series(closes, 14)
        atr = calculate_atr(highs, lows, closes, period=14)

        indicator_data[sym] = {
            "closes": closes,
            "opens": opens,
            "highs": highs,
            "lows": lows,
            "volumes": volumes,
            "ema9": ema9,
            "ema21": ema21,
            "ema50": ema50,
            "rsi": rsi,
            "atr": atr,
        }

    # Simulation State
    cash = 50.0
    starting_usd = 50.0
    active_pos = None  # Exactly MAX 1 POSITION AT ALL TIMES
    trades = []
    equity_curve = [50.0]

    fee_pct = 0.002  # 0.20% round-trip DEX swap fee + slippage
    bet_pct = 0.70   # 70% of cash allocated to the single position

    # Loop bar-by-bar chronologically
    for i in range(200, len(common_ts)):
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

            # Dynamic tailored targets
            tp_pct = active_pos["tp_pct"]
            sl_pct = active_pos["sl_pct"]
            trailing_act = active_pos["trailing_act"]
            trailing_cb = active_pos["trailing_cb"]

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

            # Rule A: Take Profit Hit
            if gain_at_high >= tp_pct:
                should_exit = True
                exit_price = entry_p * (1.0 + tp_pct / 100.0)
                outcome = "WIN"
                reason = f"🎯 TP +{tp_pct:.1f}% Hit"

            # Rule B: Trailing Stop Lock
            elif peak_gain >= trailing_act and curr_l <= (peak_p * (1.0 - trailing_cb / 100.0)):
                should_exit = True
                exit_price = peak_p * (1.0 - trailing_cb / 100.0)
                outcome = "WIN" if exit_price > entry_p else "LOSS"
                reason = f"🚀 Trailing Lock (+{(exit_price - entry_p)/entry_p*100:.2f}%)"

            # Rule C: Stop Loss Cut
            elif loss_at_low <= -sl_pct:
                should_exit = True
                exit_price = entry_p * (1.0 - sl_pct / 100.0)
                outcome = "LOSS"
                reason = f"🛡️ Stop -{sl_pct:.1f}%"

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
            # Check Improvement 1: Bitcoin Macro Gate
            btc_safe = (btc_closes[i] >= btc_ema200[i] * 0.995) and (btc_rsi[i] >= 40.0)
            if not btc_safe:
                continue  # Halt entries while BTC is dumping

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
                v_now = d["volumes"][i]
                avg_vol = np.mean(d["volumes"][i-20:i])

                # Trend Alignment
                is_uptrend = (ema9_now > ema21_now > ema50_now) and (c_now > ema50_now)

                # Pullback to Support on Bar i-1
                dip_to_support = (l_prev <= ema21_now * 1.002) or (rsi_prev <= 48.0)

                # Confirmed Green Reversal Candle on Bar i
                green_reversal = (c_now > o_now) and (c_now > c_prev) and (c_now >= ema9_now * 0.999)

                # RSI Not Overextended
                rsi_valid = (rsi_now <= 58.0) and (rsi_now >= 38.0)

                # Check Improvement 3: Volume Confirmation (Vol >= 1.1x 20-period avg)
                volume_confirmed = v_now >= (avg_vol * 1.10)

                if is_uptrend and dip_to_support and green_reversal and rsi_valid and volume_confirmed:
                    # Check Improvement 2: Dynamic ATR Targets
                    atr_val = d["atr"][i]
                    atr_pct = (atr_val / c_now) * 100.0
                    sl_pct = max(2.0, min(3.5, round(atr_pct * 1.5, 2)))
                    tp_pct = max(4.5, min(8.0, round(atr_pct * 3.5, 2)))

                    pos_size = round(cash * bet_pct, 2)
                    cash -= pos_size
                    active_pos = {
                        "symbol": sym,
                        "entry_price": c_now,
                        "peak_price": c_now,
                        "allocated_usd": pos_size,
                        "tp_pct": tp_pct,
                        "sl_pct": sl_pct,
                        "trailing_act": round(sl_pct * 1.25, 2),
                        "trailing_cb": round(sl_pct * 0.50, 2),
                        "entry_idx": i,
                        "entry_time": dt_str
                    }
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

    print("=" * 88)
    print("  🏆 UNIFIED BACKTEST WITH IMPROVEMENTS 1, 2, 3 (BTC GATE + ATR + VOLUME)")
    print("=" * 88)
    print(f"  Starting Balance   : ${starting_usd:.2f} USD")
    print(f"  Ending Balance     : ${cash:.2f} USD")
    print(f"  Total Net Return   : {total_net_pnl:+.2f} USD ({total_return_pct:+.2f}%)")
    print(f"  Total Closed Trades: {len(trades)} trades")
    print(f"  Win / Loss Record  : {len(wins)} Wins / {len(losses)} Losses")
    print(f"  Win Rate           : {win_rate:.1f}%")
    print(f"  Profit Factor      : {profit_factor:.2f}")
    print("-" * 88)
    print("  SAMPLE CHRONOLOGICAL TRADE LEDGER (LAST 10 TRADES):")
    print(f"  {'Date':<17} {'Symbol':<10} {'Entry':>10} {'Exit':>10} {'Return':>9} {'PnL (USD)':>10} {'Outcome':<8} {'Reason'}")
    print("  " + "-" * 84)
    for t in trades[-10:]:
        print(
            f"  {t['date']:<17} {t['symbol']:<10} ${t['entry']:>9.4f} ${t['exit']:>9.4f} "
            f"{t['return_pct']:>+8.2f}% {t['pnl_usd']:>+9.2f}  {t['outcome']:<8} {t['reason']}"
        )
    print("=" * 88)

if __name__ == "__main__":
    run_unified_portfolio_backtest()
