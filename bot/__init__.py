"""QuantLab Binance Halal Spot Volatility Grid Bot.

Modular architecture:
- bot.grid: Adaptive Halal Spot Volatility Grid engine (SOL, AVAX, NEAR, LINK).
- bot.state: Portfolio state loading, saving, circuit breaker, and exchange reconciliation.
- bot.positions: Real-time position tracking, TP/SL, and progress gauges.
- bot.alerts: Discord webhook alerts and daily performance digests.
- bot.dashboard: Interactive CLI table dashboard rendering.
- bot.ledger: Standardized trade logging.
- bot.stats: Ledger expectancy and performance analytics.
- bot.main: Execution loop, single-cycle orchestration, and CLI entrypoint.
"""

from utils import TRADE_CSV_COLUMNS, load_config, setup_logger
from bot.ledger import log_trade
from bot.alerts import send_discord_alert, send_daily_digest
from bot.dashboard import print_dashboard
from bot.grid import calculate_grid_levels, run_grid_cycle, select_best_halal_asset
from bot.positions import check_and_update_positions, render_trade_progress
from bot.state import (
    load_state,
    save_state,
    get_portfolio_equity,
    check_circuit_breaker,
    reconcile_positions,
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
    "calculate_grid_levels",
    "run_grid_cycle",
    "select_best_halal_asset",
    "check_and_update_positions",
    "render_trade_progress",
    "load_state",
    "save_state",
    "get_portfolio_equity",
    "check_circuit_breaker",
    "reconcile_positions",
    "execute_cycle",
    "main",
]
