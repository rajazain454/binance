"""QuantLab Binance Quantitative AI Trading Bot.

Entrypoint wrapper providing 100% backward-compatible access to the modular
subpackages in `bot/`:
- `bot.state`: State loading, saving, circuit breaker, and live exchange reconciliation
- `bot.positions`: Position management, dynamic TP/SL, and chandelier trailing
- `bot.scanner`: Market scanning, macro regime detection, and correlation filter
- `bot.alerts`: Discord alerts and daily performance digests
- `bot.dashboard`: Terminal CLI table dashboard rendering
- `bot.ledger`: Normalized trade logging
- `bot.main`: Execution orchestration and CLI runner
"""

import sys

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Re-export all core components for backward compatibility
from utils import TRADE_CSV_COLUMNS, load_config, setup_logger
from bot.ledger import log_trade
from bot.alerts import send_discord_alert, send_daily_digest
from bot.dashboard import print_dashboard
from bot.positions import check_and_update_positions, render_trade_progress
from bot.state import (
    load_state,
    save_state,
    get_portfolio_equity,
    check_circuit_breaker,
    reconcile_positions,
)
from bot.scanner import (
    compute_asset_correlation,
    get_macro_regime_threshold,
    scan_and_execute,
)
from bot.main import execute_cycle, main
from bot.stats import compute_ledger_stats, print_stats_report

__all__ = [
    "TRADE_CSV_COLUMNS",
    "load_config",
    "setup_logger",
    "log_trade",
    "compute_ledger_stats",
    "print_stats_report",
    "send_discord_alert",
    "send_daily_digest",
    "print_dashboard",
    "check_and_update_positions",
    "render_trade_progress",
    "load_state",
    "save_state",
    "get_portfolio_equity",
    "check_circuit_breaker",
    "reconcile_positions",
    "compute_asset_correlation",
    "get_macro_regime_threshold",
    "scan_and_execute",
    "execute_cycle",
    "main",
]

if __name__ == "__main__":
    main()
