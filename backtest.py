"""Walk-Forward Backtester and Strategy Validator.
Simulates trading execution bar-by-bar using the trained AI model,
enforcing ATR stop loss, take profit targets, capital allocation, and transaction fees.
Supports individual asset backtesting as well as portfolio-wide universe backtesting.
"""

import os
import sys
import json
import argparse
import joblib
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import features
from binance_client import BinanceClient
from utils import load_config


def run_backtest(df, model_bundle, config, symbol="BTC/USDT"):
    model = model_bundle["model"]
    feature_names = model_bundle["feature_names"]
    conf_thresh = config.get("ai_model", {}).get("confidence_threshold", 0.60)
    ai_cfg = config.get("ai_model", {})
    tp1_mult = ai_cfg.get("tp1_atr_mult", 1.2)
    tp2_mult = ai_cfg.get("tp2_atr_mult", ai_cfg.get("tp_atr_mult", 2.4))
    sl_mult = ai_cfg.get("sl_atr_mult", 1.4)
    partial_ratio = ai_cfg.get("partial_tp_ratio", 0.50)
    breakeven_lock = ai_cfg.get("breakeven_lock_enabled", True)
    risk_pct = config["risk_management"]["risk_per_trade_pct"] / 100.0
    cost_rate = config["risk_management"]["fee_rate"] + config["risk_management"]["slippage_rate"]
    initial_capital = config["risk_management"]["capital_usdt"]

    # Generate features and ATR
    feat_df = features.extract_features(df)
    atr = features.compute_atr(df, 14)

    common_idx = feat_df.index.intersection(atr.index)
    feat_df = feat_df.loc[common_idx]
    atr = atr.loc[common_idx]
    df_aligned = df.loc[common_idx]

    if len(df_aligned) < 20:
        return {
            "symbol": symbol,
            "error": f"Insufficient data for {symbol} ({len(df_aligned)} bars)"
        }

    # Predict probabilities
    X = feat_df[feature_names]
    probs = model.predict_proba(X)[:, 1]

    # Simulation state
    capital = initial_capital
    peak_capital = capital
    trades = []
    in_pos = False
    tp1_reached = False
    entry_price = 0.0
    entry_time = None
    entry_bar_idx = 0
    stop_loss = 0.0
    take_profit_1 = 0.0
    take_profit_2 = 0.0
    pos_units = 0.0
    allocated_cash = 0.0
    highest_price = 0.0
    pos_atr = 0.0

    # Live-aligned parameters
    max_hold_bars = config.get("risk_management", {}).get("max_holding_hours", 48)
    min_reserve_pct = config.get("risk_management", {}).get("min_cash_reserve_pct", 0.15)
    trail_cfg = config.get("trailing_stop", {})
    use_chandelier = trail_cfg.get("enabled", False)
    trail_mult = trail_cfg.get("atr_mult", 2.2)
    cooldown_bars = config.get("risk_management", {}).get("loss_cooldown_hours", 12)
    req_bull_st = ai_cfg.get("require_bull_supertrend", True)
    min_cmf = ai_cfg.get("min_cmf", -0.05)

    # Circuit breaker and BTC cascade parameters (matches live bot)
    cb_cfg = config.get("risk_management", {})
    max_dd_pct = cb_cfg.get("daily_max_drawdown_pct", 8.0)
    cb_hours = cb_cfg.get("circuit_breaker_hours", 24)
    cb_until_bar = -1

    btc_cascade_series = None
    if symbol != "BTC/USDT":
        clean_btc = "binance_BTC_USDT_1h.csv"
        btc_path = os.path.join(config.get("paths", {}).get("data_dir", "data"), clean_btc)
        if os.path.exists(btc_path):
            try:
                btc_df = pd.read_csv(btc_path, index_col=0, parse_dates=True)
                btc_ret_1h = btc_df["close"].pct_change()
                btc_cascade_series = btc_ret_1h.reindex(df_aligned.index, method="ffill")
            except Exception:
                pass

    # Cooldown tracking
    cooldown_until_bar = -1

    # Pre-compute SuperTrend and CMF for filter alignment with live bot
    try:
        st_series, _ = features.compute_supertrend(df_aligned, 10, 3.0)
        cmf_series = features.compute_cmf(df_aligned, 20)
    except Exception:
        st_series = pd.Series(1.0, index=df_aligned.index)
        cmf_series = pd.Series(0.0, index=df_aligned.index)

    equity_curve = []

    for i in range(len(df_aligned)):
        t = df_aligned.index[i]
        c = float(df_aligned["close"].iloc[i])
        h = float(df_aligned["high"].iloc[i])
        l = float(df_aligned["low"].iloc[i])
        curr_atr = float(atr.iloc[i])
        prob = float(probs[i])

        if in_pos:
            # Track highest price for chandelier trailing
            highest_price = max(highest_price, h)

            # Stage 1: Partial Scale-Out (50%) at TP1 & Breakeven Lock
            if not tp1_reached and h >= take_profit_1:
                sell_units = pos_units * partial_ratio
                sell_alloc = allocated_cash * partial_ratio
                gross_ret = (take_profit_1 / entry_price) - 1.0
                fee_cost = (sell_alloc * cost_rate) * 2.0
                net_pnl = (sell_units * (take_profit_1 - entry_price)) - fee_cost

                capital += net_pnl
                pos_units -= sell_units
                allocated_cash -= sell_alloc
                tp1_reached = True

                if breakeven_lock:
                    stop_loss = entry_price * 1.001

                trades.append({
                    "symbol": symbol,
                    "entry_time": str(entry_time)[:19],
                    "exit_time": str(t)[:19],
                    "entry_price": round(entry_price, 4),
                    "exit_price": round(take_profit_1, 4),
                    "exit_reason": "TP1 Partial Exit (50% Sold)",
                    "net_pnl": round(net_pnl, 2),
                    "pnl_pct": round(gross_ret * 100, 2),
                    "win": True
                })

            # Chandelier Dynamic Trailing Stop (after TP1 banked)
            if use_chandelier and tp1_reached and pos_atr > 0:
                chandelier_stop = highest_price - (trail_mult * pos_atr)
                be_level = entry_price * 1.001
                stop_loss = max(stop_loss, chandelier_stop, be_level)

            # Max holding time exit
            bars_held = i - entry_bar_idx
            is_stale = (max_hold_bars > 0) and (bars_held >= max_hold_bars)

            # Stage 2: Final TP2, Stop Loss, Chandelier Exit, or Max Hold Exit
            hit_tp2 = h >= take_profit_2
            hit_sl = l <= stop_loss

            if hit_tp2 or hit_sl or is_stale:
                if hit_tp2 and hit_sl:
                    exit_price = stop_loss
                    exit_reason = "Stop Loss (High Volatility)"
                elif hit_tp2:
                    exit_price = take_profit_2
                    exit_reason = "TP2 Final Target"
                elif is_stale and not hit_sl:
                    exit_price = c
                    exit_reason = f"Max Holding Time ({max_hold_bars}h)"
                elif tp1_reached:
                    exit_price = stop_loss
                    exit_reason = "Chandelier Trailing Exit"
                else:
                    exit_price = stop_loss
                    exit_reason = "Stop Loss"

                gross_ret = (exit_price / entry_price) - 1.0
                fee_cost = (allocated_cash * cost_rate) * 2.0
                net_pnl = (pos_units * (exit_price - entry_price)) - fee_cost
                capital += net_pnl
                # Match live bot: tp1_reached counts as win
                win = hit_tp2 or (exit_price >= entry_price) or tp1_reached

                trades.append({
                    "symbol": symbol,
                    "entry_time": str(entry_time)[:19],
                    "exit_time": str(t)[:19],
                    "entry_price": round(entry_price, 4),
                    "exit_price": round(exit_price, 4),
                    "exit_reason": exit_reason,
                    "net_pnl": round(net_pnl, 2),
                    "pnl_pct": round(gross_ret * 100, 2),
                    "win": win
                })
                in_pos = False

                # Set loss cooldown (matches live bot)
                if not win and cooldown_bars > 0:
                    cooldown_until_bar = i + cooldown_bars

                tp1_reached = False

        # Entry logic: confidence threshold + MTF 4h + SuperTrend + CMF filters
        use_mtf = config.get("mtf_confluence", {}).get("enabled", True)
        is_mtf_bullish = bool(feat_df["mtf_4h_bullish"].iloc[i] > 0.5) if ("mtf_4h_bullish" in feat_df and use_mtf) else True

        # SuperTrend and CMF filter gates (aligned with live bot)
        is_supertrend_bull = bool(st_series.iloc[i] > 0.5) if i < len(st_series) else True
        cmf_val = float(cmf_series.iloc[i]) if i < len(cmf_series) else 0.0
        supertrend_ok = (not req_bull_st) or is_supertrend_bull
        cmf_ok = cmf_val >= min_cmf

        # Loss cooldown check (aligned with live bot)
        is_on_cooldown = (i < cooldown_until_bar)

        # 24h rolling circuit breaker check (aligned with live bot)
        rolling_24h_peak = max(equity_curve[-24:]) if len(equity_curve) >= 24 else peak_capital
        if rolling_24h_peak > 0:
            rolling_dd = ((capital / rolling_24h_peak) - 1.0) * 100.0
            if rolling_dd <= -max_dd_pct:
                cb_until_bar = max(cb_until_bar, i + cb_hours)
        cb_locked = (i < cb_until_bar)

        # BTC cascade halt check (aligned with live bot)
        btc_halt_thresh = config.get("risk_management", {}).get("btc_cascade_halt_pct", -0.025)
        is_btc_halted = False
        if btc_cascade_series is not None and i < len(btc_cascade_series):
            ret_val = float(btc_cascade_series.iloc[i])
            if not np.isnan(ret_val) and ret_val <= btc_halt_thresh:
                is_btc_halted = True

        if (not in_pos and prob >= conf_thresh and is_mtf_bullish
                and supertrend_ok and cmf_ok and not is_on_cooldown
                and not cb_locked and not is_btc_halted
                and not np.isnan(curr_atr) and curr_atr > 0):
            entry_price = c
            entry_time = t
            entry_bar_idx = i
            stop_loss = entry_price - (sl_mult * curr_atr)
            take_profit_1 = entry_price + (tp1_mult * curr_atr)
            take_profit_2 = entry_price + (tp2_mult * curr_atr)
            tp1_reached = False
            highest_price = c
            pos_atr = curr_atr

            # Dynamic alpha sizing (aligned with live bot)
            dynamic_sizing = config.get("risk_management", {}).get("dynamic_alpha_sizing", True)
            if dynamic_sizing:
                surplus = max(-0.10, min(0.20, prob - conf_thresh))
                alpha_mult = 1.0 + (surplus * 2.0)
                eff_risk_pct = risk_pct * alpha_mult
            else:
                eff_risk_pct = risk_pct

            risk_amount = capital * eff_risk_pct
            risk_per_unit = entry_price - stop_loss
            if risk_per_unit > 0:
                units = risk_amount / risk_per_unit
                # Cash reserve & portfolio cap enforcement (aligned with live bot)
                min_reserve = capital * min_reserve_pct
                spendable = max(0.0, capital - min_reserve)
                max_trade_pct = config.get("risk_management", {}).get("max_trade_allocation_pct", 0.32)
                portfolio_cap = capital * max_trade_pct
                allocated_cash = min(spendable * 0.95, portfolio_cap, units * entry_price)
                min_notional = float(config.get("risk_management", {}).get("min_notional_usdt", 5.0))
                if allocated_cash >= min_notional:  # Dynamic Binance minNotional
                    pos_units = allocated_cash / entry_price
                    in_pos = True

        equity_curve.append(capital)
        peak_capital = max(peak_capital, capital)

    # If position is still open at the very last bar, close it out at last close
    if in_pos:
        last_t = df_aligned.index[-1]
        last_c = float(df_aligned["close"].iloc[-1])
        gross_ret = (last_c / entry_price) - 1.0
        fee_cost = (allocated_cash * cost_rate) * 2.0
        net_pnl = (pos_units * (last_c - entry_price)) - fee_cost
        capital += net_pnl
        win = net_pnl > 0 or (tp1_reached and last_c >= entry_price)

        trades.append({
            "symbol": symbol,
            "entry_time": str(entry_time)[:19],
            "exit_time": str(last_t)[:19],
            "entry_price": round(entry_price, 4),
            "exit_price": round(last_c, 4),
            "exit_reason": "End of Backtest",
            "net_pnl": round(net_pnl, 2),
            "pnl_pct": round(gross_ret * 100, 2),
            "win": win
        })
        equity_curve[-1] = capital

    # Compute detailed statistics
    eq_series = pd.Series(equity_curve, index=df_aligned.index)
    peak_series = eq_series.cummax()
    drawdown_series = (eq_series - peak_series) / peak_series
    max_dd = float(drawdown_series.min())

    # Daily returns for Sharpe computation
    if isinstance(eq_series.index, pd.DatetimeIndex):
        daily_rets = eq_series.resample("1D").last().pct_change().dropna()
    else:
        daily_rets = eq_series.iloc[::24].pct_change().dropna()
    sharpe = float((daily_rets.mean() / (daily_rets.std() + 1e-9)) * np.sqrt(365)) if len(daily_rets) > 1 else 0.0

    n_trades = len(trades)
    if n_trades > 0:
        wins = [tr for tr in trades if tr["win"]]
        losses = [tr for tr in trades if not tr["win"]]
        win_rate = len(wins) / n_trades
        gross_profit = sum(tr["net_pnl"] for tr in wins)
        gross_loss = abs(sum(tr["net_pnl"] for tr in losses)) if len(losses) else 1e-9
        profit_factor = gross_profit / gross_loss
        avg_trade = (capital - initial_capital) / n_trades
        avg_win = (gross_profit / len(wins)) if wins else 0.0
        avg_loss = (gross_loss / len(losses)) if losses else 0.0
        risk_reward_ratio = (avg_win / avg_loss) if avg_loss > 0 else 0.0
    else:
        win_rate = 0.0
        profit_factor = 0.0
        avg_trade = 0.0
        avg_win = 0.0
        avg_loss = 0.0
        risk_reward_ratio = 0.0

    total_return_pct = ((capital / initial_capital) - 1.0) * 100.0

    report = {
        "symbol": symbol,
        "timeframe": config.get("timeframe", "1h"),
        "initial_capital": initial_capital,
        "final_capital": round(capital, 2),
        "total_return_pct": round(total_return_pct, 2),
        "max_drawdown_pct": round(max_dd * 100.0, 2),
        "sharpe_ratio": round(sharpe, 2),
        "total_trades": n_trades,
        "win_rate": round(win_rate * 100.0, 1),
        "profit_factor": round(profit_factor, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "risk_reward_ratio": round(risk_reward_ratio, 2),
        "avg_trade_pnl": round(avg_trade, 2),
        "trades": trades
    }
    return report


def load_data_for_symbol(symbol, config, client, offline=False):
    clean_sym = symbol.replace("/", "_")
    cache_path = os.path.join(config["paths"]["data_dir"], f"binance_{clean_sym}_{config['timeframe']}.csv")

    if offline and os.path.exists(cache_path):
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    try:
        df = client.fetch_historical_ohlcv(symbol, timeframe=config["timeframe"], days=120)
        if not df.empty:
            df.to_csv(cache_path)
            return df
    except Exception as e:
        print(f"Notice: Live fetch failed for {symbol}: {e}")

    if os.path.exists(cache_path):
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    return pd.DataFrame()


def print_single_report(res):
    print("\n" + "=" * 70)
    print(f"  BINANCE QUANTITATIVE BACKTEST REPORT: {res['symbol']}")
    print("=" * 70)
    print(f"  Timeframe:             {res['timeframe']}")
    print(f"  Initial Bankroll:      ${res['initial_capital']:,.2f} USDT")
    print(f"  Final Bankroll:        ${res['final_capital']:,.2f} USDT")
    print(f"  Net Return:            {res['total_return_pct']:+.2f}%")
    print(f"  Max Drawdown:          {res['max_drawdown_pct']:.2f}%")
    print(f"  Sharpe Ratio:          {res['sharpe_ratio']:.2f}")
    print(f"  Total Closed Trades:   {res['total_trades']}")
    print(f"  Strategy Win Rate:     {res['win_rate']:.1f}%")
    print(f"  Profit Factor:         {res['profit_factor']:.2f}")
    print(f"  Avg Win vs Avg Loss:   ${res['avg_win']:,.2f} / ${res['avg_loss']:,.2f} (R:R {res['risk_reward_ratio']}:1)")
    print(f"  Expected PnL / Trade:  ${res['avg_trade_pnl']:+,.2f}")
    print("=" * 70)

    if res.get("trades"):
        print("\nLast 5 Closed Trades:")
        for tr in res["trades"][-5:]:
            win_tag = "[WIN]" if tr["win"] else "[LOSS]"
            print(f"  {win_tag:<6} {tr['entry_time']} -> {tr['exit_time']} | "
                  f"Entry: ${tr['entry_price']:.2f} | Exit: ${tr['exit_price']:.2f} ({tr['exit_reason']}) | "
                  f"Net: ${tr['net_pnl']:+,.2f} ({tr['pnl_pct']:+.1f}%)")
    print()


def main():
    parser = argparse.ArgumentParser(description="Backtest Binance Quantitative AI Model")
    parser.add_argument("--config", default="config.json", help="Path to config file")
    parser.add_argument("--model", default="models/binance_ai_model.joblib", help="Path to trained model")
    parser.add_argument("--symbol", default="BTC/USDT", help="Symbol to backtest (or 'ALL' for universe)")
    parser.add_argument("--all", action="store_true", help="Backtest all symbols in config.json")
    parser.add_argument("--offline", action="store_true", help="Use cached data")

    args = parser.parse_args()
    config = load_config(args.config)

    if not os.path.exists(args.model):
        sys.exit(f"Error: Model file '{args.model}' not found. Please run 'python train.py' first.")

    bundle = joblib.load(args.model)
    client = BinanceClient(config)

    symbols_to_test = config["symbols"] if (args.all or args.symbol.upper() == "ALL") else [args.symbol]

    all_reports = []
    for sym in symbols_to_test:
        df = load_data_for_symbol(sym, config, client, offline=args.offline)
        if df.empty:
            print(f"[Warning] No data found for {sym}. Skipping...")
            continue
        report = run_backtest(df, bundle, config, symbol=sym)
        all_reports.append(report)
        if len(symbols_to_test) == 1:
            print_single_report(report)

    if len(symbols_to_test) > 1 and all_reports:
        print("\n" + "=" * 90)
        print("  PORTFOLIO UNIVERSE BACKTEST SUMMARY")
        print("=" * 90)
        print(f"{'Symbol':<12} {'Return %':>10} {'Win Rate':>10} {'Profit Factor':>15} {'Max DD':>10} {'Trades':>8} {'Sharpe':>8}")
        print("-" * 90)
        for r in all_reports:
            if "error" in r:
                continue
            print(f"{r['symbol']:<12} {r['total_return_pct']:>+9.2f}% {r['win_rate']:>9.1f}% {r['profit_factor']:>14.2f} "
                  f"{r['max_drawdown_pct']:>9.2f}% {r['total_trades']:>8d} {r['sharpe_ratio']:>8.2f}")
        print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
