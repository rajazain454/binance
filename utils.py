"""Shared utility functions and constants used across QuantLab modules."""

import json
import logging
import os

logger = logging.getLogger("quantlab")

TRADE_CSV_COLUMNS = [
    "timestamp",
    "position_id",
    "symbol",
    "action",
    "entry_price",
    "exit_price",
    "allocated_usdt",
    "stop_loss",
    "tp1",
    "tp2",
    "net_pnl",
    "pnl_pct",
    "confidence_pct",
    "reason",
    "outcome",
    "mode",
]


def load_config(config_path="config.json"):
    """Load and return the JSON configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def close_logger_handlers(logger_name="quantlab"):
    """Closes and removes all FileHandlers from the specified logger."""
    logger_inst = logging.getLogger(logger_name)
    for h in list(logger_inst.handlers):
        if isinstance(h, logging.FileHandler):
            try:
                h.close()
            except Exception:
                pass
            logger_inst.removeHandler(h)


def setup_logger(log_file="logs/quantlab.log", level=logging.INFO):
    """Configures root/quantlab logger with console and file handlers.

    Accepts a log_file path (or None). If a FileHandler was previously attached,
    it is closed and replaced so tests or runtime components can redirect output
    to a temporary or custom log file without leaking handlers.
    """
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger_inst = logging.getLogger("quantlab")
    logger_inst.setLevel(level)

    # Ensure a console StreamHandler is attached
    has_stream = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger_inst.handlers
    )
    if not has_stream:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger_inst.addHandler(ch)

    # Close and remove existing FileHandlers to switch log files cleanly
    close_logger_handlers("quantlab")

    if log_file:
        dir_name = os.path.dirname(log_file)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        try:
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(formatter)
            logger_inst.addHandler(fh)
        except Exception as e:
            logger_inst.warning("Could not set up file log handler: %s", e)

    return logger_inst
