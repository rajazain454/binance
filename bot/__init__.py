"""QuantLab Binance AI Trading Bot Package.

Modular subpackages:
- bot.state: Portfolio state loading, saving, and exchange reconciliation.
- bot.positions: Real-time position tracking, TP/SL, and chandelier trailing stops.
- bot.scanner: Market analysis, macro regime detection, and candidate filtering.
- bot.alerts: Discord webhook alerts and daily performance digests.
- bot.dashboard: Interactive CLI table dashboard rendering.
- bot.ledger: Standardized trade logging.
- bot.main: Execution loop, single-cycle orchestration, and CLI entrypoint.
"""

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

__all__ = [
    "TRADE_CSV_COLUMNS",
    "load_config",
    "setup_logger",
    "log_trade",
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
