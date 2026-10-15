import argparse
import logging
import os
import sys
from typing import Dict, Any, List, Tuple
import requests
import numpy as np

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("backtester")


def fetch_historical_candles(symbol: str, interval: str = "5m", total_candles: int = 1000) -> List[Dict[str, float]]:
    """Fetches up to 1000 historical institutional candles from Binance public API."""
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={min(total_candles, 1000)}"
    r = requests.get(url, headers={"User-Agent": "QuantBacktest/1.0"}, timeout=10)
    if r.status_code != 200:
        return []
    
    candles = []
    for k in r.json():
        candles.append({
            "timestamp": int(k[0]),
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
        })
    return candles


def calc_ema(prices: np.ndarray, period: int) -> np.ndarray:
    ema = np.zeros_like(prices)
    alpha = 2.0 / (period + 1.0)
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1.0 - alpha) * ema[i - 1]
    return ema


def calc_rsi_series(prices: np.ndarray, period: int = 14) -> np.ndarray:
    deltas = np.diff(prices)
    rsi = np.full(len(prices), 50.0)
    if len(prices) <= period + 1:
        return rsi

    up = deltas.copy()
    down = deltas.copy()
    up[up < 0] = 0.0
    down[down > 0] = 0.0
    down = np.abs(down)

    roll_up = np.zeros(len(deltas))
    roll_down = np.zeros(len(deltas))
    roll_up[period - 1] = np.mean(up[:period])
    roll_down[period - 1] = np.mean(down[:period])

    for i in range(period, len(deltas)):
        roll_up[i] = (roll_up[i - 1] * (period - 1) + up[i]) / period
        roll_down[i] = (roll_down[i - 1] * (period - 1) + down[i]) / period

    for i in range(period, len(deltas)):
        if roll_down[i] == 0:
            rsi[i + 1] = 100.0
        else:
            rs = roll_up[i] / roll_down[i]
            rsi[i + 1] = 100.0 - 100.0 / (1.0 + rs)

    return rsi


def run_backtest_on_symbol(
    symbol: str, 
    candles: List[Dict[str, float]], 
    tp_pct: float = 1.2, 
    sl_pct: float = 0.6, 
    starting_usd: float = 50.0,
    bet_pct: float = 0.40
) -> Dict[str, Any]:
    """Simulates trading execution over historical candles with fee and slippage modeling."""
    closes = np.array([c["close"] for c in candles], dtype=float)
    highs = np.array([c["high"] for c in candles], dtype=float)
    lows = np.array([c["low"] for c in candles], dtype=float)
    volumes = np.array([c["volume"] for c in candles], dtype=float)

    ema9 = calc_ema(closes, 9)
    ema21 = calc_ema(closes, 21)
    ema50 = calc_ema(closes, 50)
    rsi = calc_rsi_series(closes, 14)
    vol_ma = calc_ema(volumes, 20)

    cash = starting_usd
    position = None
    trades = []
    
    fee_pct = 0.0005  # 0.05% Jupiter swap fee

    for i in range(50, len(candles) - 1):
        price = closes[i]
        curr_high = highs[i]
        curr_low = lows[i]

        # 1. Manage Active Position
        if position is not None:
            entry_p = position["entry_price"]
            allocated = position["allocated_usd"]

            # Track peak price
            peak_p = max(position.get("peak_p", entry_p), curr_high)
            position["peak_p"] = peak_p
            peak_gain = (peak_p - entry_p) / entry_p * 100.0

            gain_at_high = (curr_high - entry_p) / entry_p * 100.0
            loss_at_low = (curr_low - entry_p) / entry_p * 100.0
            hit_tp = gain_at_high >= tp_pct
            hit_sl = loss_at_low <= -sl_pct

            # Trailing Stop: If price gained >= 0.5%, trail by 0.25% from peak
            trailing_exit = False
            if peak_gain >= 0.50 and curr_low <= (peak_p * 0.9975):
                trailing_exit = True

            should_exit = False
            exit_price = price
            outcome = "HOLD"

            if hit_tp:
                should_exit = True
                exit_price = entry_p * (1.0 + tp_pct / 100.0)
                outcome = "WIN"
            elif trailing_exit:
                should_exit = True
                exit_price = peak_p * 0.9975
                outcome = "WIN" if exit_price > entry_p else "LOSS"
            elif hit_sl:
                should_exit = True
                exit_price = entry_p * (1.0 - sl_pct / 100.0)
                outcome = "LOSS"
            elif (i - position["entry_idx"]) >= 12:  # Stale cut after 12 bars
                should_exit = True
                exit_price = price
                outcome = "WIN" if price > entry_p else "LOSS"

            if should_exit:
                gross = allocated * (exit_price / entry_p)
                fee = gross * fee_pct
                net_pnl = gross - allocated - fee
                cash += (allocated + net_pnl)
                trades.append({
                    "symbol": symbol,
                    "entry": entry_p,
                    "exit": exit_price,
                    "outcome": outcome,
                    "pnl": net_pnl,
                    "return_pct": (exit_price - entry_p) / entry_p * 100.0
                })
                position = None

        # 2. Look for Buy Signal (Bollinger Band Mean Reversion)
        elif position is None and cash >= 10.0:
            if i >= 20:
                sma20 = np.mean(closes[i-19:i+1])
                std20 = np.std(closes[i-19:i+1])
                lower_band = sma20 - 2.0 * std20
                
                # 2-Bar Trend Pullback Sequence:
                # Bar i-1 dipped into EMA 21 or RSI pullback
                dip_bar = (lows[i - 1] <= ema21[i - 1] * 1.002) or (rsi[i - 1] <= 48.0)
                # Bar i breaks out bullishly
                reversal_bar = (closes[i] > closes[i - 1]) and (closes[i] > ema9[i])
                is_uptrend = (ema9[i] > ema21[i]) and (ema21[i] > ema50[i]) and (price > ema50[i])

                if is_uptrend and dip_bar and reversal_bar:
                    pos_size = cash * bet_pct
                    cash -= pos_size
                    position = {
                        "entry_price": price,
                        "allocated_usd": pos_size,
                        "entry_idx": i
                    }

    # Finalize if still in position
    if position:
        entry_p = position["entry_price"]
        allocated = position["allocated_usd"]
        exit_p = closes[-1]
        net_pnl = (allocated * (exit_p / entry_p)) - allocated
        cash += (allocated + net_pnl)
        trades.append({
            "symbol": symbol,
            "entry": entry_p,
            "exit": exit_p,
            "outcome": "WIN" if net_pnl > 0 else "LOSS",
            "pnl": net_pnl,
            "return_pct": (exit_p - entry_p) / entry_p * 100.0
        })

    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    win_rate = (len(wins) / len(trades) * 100.0) if trades else 0.0
    total_pnl = cash - starting_usd
    profit_factor = (sum(t["pnl"] for t in wins) / max(0.001, abs(sum(t["pnl"] for t in losses)))) if losses else 99.0

    return {
        "symbol": symbol,
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": win_rate,
        "ending_cash": cash,
        "total_pnl": total_pnl,
        "profit_factor": profit_factor,
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-Asset Quantitative Scalper Backtester")
    parser.add_argument("--tp", type=float, default=1.0, help="Take-profit percentage (e.g. 1.0)")
    parser.add_argument("--sl", type=float, default=0.5, help="Stop-loss percentage (e.g. 0.5)")
    args = parser.parse_args()

    symbols = ["SOLUSDT", "JUPUSDT", "RAYUSDT", "WIFUSDT"]
    print("=" * 80)
    print("  🔬 QUANTITATIVE MULTI-ASSET HISTORICAL BACKTEST ENGINE (1,000 CANDLES)")
    print(f"  Targets: Take Profit: +{args.tp:.2f}% | Stop Loss: -{args.sl:.2f}% (2:1 Ratio)")
    print("=" * 80)
    print(f"  {'Symbol':<10} {'Trades':>8} {'Wins':>6} {'Losses':>8} {'Win Rate':>10} {'Profit Factor':>15} {'Net PnL':>12}")
    print("  " + "-" * 76)

    total_net_pnl = 0.0
    total_trades = 0
    total_wins = 0

    for s in symbols:
        candles = fetch_historical_candles(s, interval="5m", total_candles=1000)
        res = run_backtest_on_symbol(s, candles, tp_pct=args.tp, sl_pct=args.sl, starting_usd=50.0, bet_pct=0.40)
        total_net_pnl += res["total_pnl"]
        total_trades += res["trades"]
        total_wins += res["wins"]
        print(
            f"  {res['symbol']:<10} {res['trades']:>8} {res['wins']:>6} {res['losses']:>8} "
            f"{res['win_rate']:>9.1f}% {res['profit_factor']:>15.2f} {res['total_pnl']:>+11.2f} USD"
        )

    overall_wr = (total_wins / total_trades * 100.0) if total_trades else 0.0
    print("  " + "-" * 76)
    print(f"  COMBINED BASKET: {total_trades} Trades | Win Rate: {overall_wr:.1f}% | Total Net PnL: {total_net_pnl:+.2f} USD")
    print("=" * 80)


if __name__ == "__main__":
    main()
