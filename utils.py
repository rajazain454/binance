"""Shared utility functions and constants used across QuantLab modules."""

import json
import logging
import os

logger = logging.getLogger("quantlab")

TRADE_CSV_COLUMNS = [
    "timestamp",
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


def setup_logger(log_file="logs/quantlab.log", level=logging.INFO):
    """Configures root/quantlab logger with both console and rotating file handlers."""
    os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else ".", exist_ok=True)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger_inst = logging.getLogger("quantlab")
    logger_inst.setLevel(level)

    if not logger_inst.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger_inst.addHandler(ch)

        try:
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(formatter)
            logger_inst.addHandler(fh)
        except Exception as e:
            logger_inst.warning("Could not set up file log handler: %s", e)

    return logger_inst
