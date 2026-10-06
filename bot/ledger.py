"""Standardized trade ledger logging for QuantLab Trading Bot."""

import os
import pandas as pd
from utils import TRADE_CSV_COLUMNS


def log_trade(trade_record, ledger_path="logs/trade_history.csv"):
    """Appends a trade record to CSV using a fixed, consistent column schema."""
    os.makedirs(os.path.dirname(ledger_path) if os.path.dirname(ledger_path) else ".", exist_ok=True)
    exists = os.path.exists(ledger_path) and os.path.getsize(ledger_path) > 0
    # Ensure every row has every canonical column (fill missing with empty string)
    normalised = {col: trade_record.get(col, "") for col in TRADE_CSV_COLUMNS}
    df = pd.DataFrame([normalised], columns=TRADE_CSV_COLUMNS)
    df.to_csv(ledger_path, mode="a", header=not exists, index=False)
