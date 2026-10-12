"""Per-position performance statistics and ledger analytics for QuantLab Trading Bot."""

import argparse
import csv
import os
import sys

# Ensure root dir in path if invoked directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def compute_ledger_stats(ledger_path="logs/trade_history.csv") -> dict:
    """Reads the trade history ledger and computes per-position analytics.

    Groups multiple ledger records (BUY entry, partial TP1, breakeven-locks, final closes)
    by `position_id` to evaluate each trade position as a single coherent unit.

    Returns:
        dict containing:
            - total_records: int
            - closed_positions: int
            - win_count: int
            - loss_count: int
            - win_rate_pct: float
            - loss_rate_pct: float
            - total_net_pnl: float
            - gross_profit: float
            - gross_loss: float
            - avg_win: float
            - avg_loss: float
            - win_loss_ratio: float
            - profit_factor: float
            - expectancy: float (expected net PnL per closed position)
            - break_even_win_rate_pct: float (minimum win rate required to break even)
    """
    default_stats = {
        "ledger_path": ledger_path,
        "total_records": 0,
        "closed_positions": 0,
        "win_count": 0,
        "loss_count": 0,
        "win_rate_pct": 0.0,
        "loss_rate_pct": 0.0,
        "total_net_pnl": 0.0,
        "gross_profit": 0.0,
        "gross_loss": 0.0,
        "avg_win": 0.0,
        "avg_loss": 0.0,
        "win_loss_ratio": 0.0,
        "profit_factor": 0.0,
        "expectancy": 0.0,
        "break_even_win_rate_pct": 50.0,
    }

    if not os.path.exists(ledger_path) or os.path.getsize(ledger_path) == 0:
        return default_stats

    records = []
    try:
        with open(ledger_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(row)
    except Exception:
        return default_stats

    if not records:
        return default_stats

    # Group records by position_id
    positions_map = {}
    for idx, r in enumerate(records):
        pid = r.get("position_id", "").strip()
        if not pid:
            # Fallback for legacy rows without position_id: synthesize unique key per closing row
            pid = f"legacy_{r.get('symbol', 'UNKNOWN')}_{idx}"
        if pid not in positions_map:
            positions_map[pid] = []
        positions_map[pid].append(r)

    closed_pnls = []

    for pid, p_rows in positions_map.items():
        # Determine if this position has been closed
        is_closed = any(
            (r.get("action", "").startswith("FINAL CLOSE") or r.get("outcome") in ("WIN", "LOSS"))
            for r in p_rows
        )
        if not is_closed:
            continue

        # Aggregate total net PnL across all tranches (e.g. TP1 partial booked PnL + runner PnL)
        pos_total_pnl = 0.0
        for r in p_rows:
            pnl_val = r.get("net_pnl", "")
            if pnl_val not in ("", None, "None"):
                try:
                    pos_total_pnl += float(pnl_val)
                except (ValueError, TypeError):
                    pass

        closed_pnls.append(round(pos_total_pnl, 4))

    closed_count = len(closed_pnls)
    if closed_count == 0:
        default_stats["total_records"] = len(records)
        return default_stats

    wins = [p for p in closed_pnls if p > 0.0]
    losses = [p for p in closed_pnls if p <= 0.0]

    win_count = len(wins)
    loss_count = len(losses)
    win_rate_pct = (win_count / closed_count) * 100.0
    loss_rate_pct = (loss_count / closed_count) * 100.0

    gross_profit = sum(wins)
    gross_loss = abs(sum(p for p in closed_pnls if p < 0.0))
    total_net_pnl = sum(closed_pnls)

    avg_win = (gross_profit / win_count) if win_count > 0 else 0.0
    avg_loss = (gross_loss / len([p for p in closed_pnls if p < 0.0])) if any(p < 0.0 for p in closed_pnls) else 0.0

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    if avg_loss > 0:
        win_loss_ratio = avg_win / avg_loss
    elif avg_win > 0:
        win_loss_ratio = float("inf")
    else:
        win_loss_ratio = 0.0

    # Mathematical expectancy: Expected net dollar return per trade position
    expectancy = total_net_pnl / closed_count

    # Break-even win rate: Required win rate where expected value is 0 (q*L = p*W)
    if (avg_win + avg_loss) > 0:
        be_win_rate_pct = (avg_loss / (avg_win + avg_loss)) * 100.0
    else:
        be_win_rate_pct = 50.0

    return {
        "ledger_path": ledger_path,
        "total_records": len(records),
        "closed_positions": closed_count,
        "win_count": win_count,
        "loss_count": loss_count,
        "win_rate_pct": round(win_rate_pct, 2),
        "loss_rate_pct": round(loss_rate_pct, 2),
        "total_net_pnl": round(total_net_pnl, 2),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "avg_win": round(avg_win, 4),
        "avg_loss": round(avg_loss, 4),
        "win_loss_ratio": round(win_loss_ratio, 2) if win_loss_ratio != float("inf") else "Inf",
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else "Inf",
        "expectancy": round(expectancy, 4),
        "break_even_win_rate_pct": round(be_win_rate_pct, 2),
    }


def print_stats_report(ledger_path="logs/trade_history.csv", stats=None):
    """Prints a structured, readable performance report from the ledger."""
    if stats is None:
        stats = compute_ledger_stats(ledger_path)

    pf_str = f"{stats['profit_factor']:.2f}" if isinstance(stats['profit_factor'], (int, float)) else str(stats['profit_factor'])
    wl_str = f"{stats['win_loss_ratio']:.2f}" if isinstance(stats['win_loss_ratio'], (int, float)) else str(stats['win_loss_ratio'])

    print("=" * 86)
    print("  QUANTLAB TRADE LEDGER PERFORMANCE ANALYTICS (PER-POSITION)")
    print("=" * 86)
    print(f"  Ledger Source          : {stats['ledger_path']}")
    print(f"  Total Ledger Records   : {stats['total_records']}")
    print(f"  Closed Positions       : {stats['closed_positions']}")
    print(f"  Total Net PnL          : ${stats['total_net_pnl']:+,.2f} USDT")
    print("-" * 86)
    print(f"  Wins / Losses          : {stats['win_count']} Wins / {stats['loss_count']} Losses")
    print(f"  Actual Win Rate        : {stats['win_rate_pct']:.1f}%")
    print(f"  Break-Even Win Rate    : {stats['break_even_win_rate_pct']:.1f}% (Minimum required)")
    print(f"  Average Win            : +${stats['avg_win']:.2f} USDT")
    print(f"  Average Loss           : -${stats['avg_loss']:.2f} USDT")
    print(f"  Win / Loss Ratio (R)   : {wl_str}")
    print(f"  Profit Factor          : {pf_str}")
    print(f"  Expectancy Per Trade   : ${stats['expectancy']:+,.2f} USDT")
    print("=" * 86)


def main():
    parser = argparse.ArgumentParser(description="QuantLab Trade Ledger Performance Analytics")
    parser.add_argument("--ledger", default="logs/trade_history.csv", help="Path to trade ledger CSV")
    args = parser.parse_args()
    print_stats_report(args.ledger)


if __name__ == "__main__":
    main()
