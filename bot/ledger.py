"""Standardized trade ledger logging for QuantLab Trading Bot."""

import csv
from datetime import datetime, timezone
import os
import pandas as pd
from utils import TRADE_CSV_COLUMNS


def log_trade(trade_record, ledger_path="logs/trade_history.csv"):
    """Appends a trade record to CSV using a fixed, consistent column schema.

    Writes the header if the file is missing or empty. Validates that an existing
    file's header equals TRADE_CSV_COLUMNS; if not, renames the old file to
    trade_history_legacy_<timestamp>.csv and starts fresh with the canonical schema.
    """
    if isinstance(ledger_path, dict):
        ledger_path = ledger_path.get("paths", {}).get("trade_ledger", "logs/trade_history.csv")

    os.makedirs(os.path.dirname(ledger_path) if os.path.dirname(ledger_path) else ".", exist_ok=True)

    write_header = True
    if os.path.exists(ledger_path) and os.path.getsize(ledger_path) > 0:
        existing_header = None
        try:
            with open(ledger_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                existing_header = next(reader, None)
        except Exception:
            existing_header = None

        if existing_header == TRADE_CSV_COLUMNS:
            write_header = False
        else:
            # Header does not match canonical TRADE_CSV_COLUMNS -> rotate to trade_history_legacy_<timestamp>.csv
            dir_name = os.path.dirname(ledger_path) or "."
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            legacy_path = os.path.join(dir_name, f"trade_history_legacy_{ts}.csv")
            if os.path.exists(legacy_path):
                ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
                legacy_path = os.path.join(dir_name, f"trade_history_legacy_{ts}.csv")
            try:
                os.rename(ledger_path, legacy_path)
            except Exception:
                pass
            write_header = True

    # Ensure every row has every canonical column (fill missing with empty string)
    normalised = {col: trade_record.get(col, "") for col in TRADE_CSV_COLUMNS}
    df = pd.DataFrame([normalised], columns=TRADE_CSV_COLUMNS)
    df.to_csv(ledger_path, mode="a", header=write_header, index=False)
