"""Archive and reset trade history and bot state for clean performance tracking."""

import os
import shutil
from datetime import datetime, timezone


def archive_trade_history(data_dir="logs", state_file="bot_state.json", initial_capital=30.0):
    """
    Archives legacy mixed-capital trade history and resets state for a clean small-account run.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    archive_dir = os.path.join(data_dir, "archive")
    os.makedirs(archive_dir, exist_ok=True)

    csv_path = os.path.join(data_dir, "trade_history.csv")
    if os.path.exists(csv_path):
        archived_csv = os.path.join(archive_dir, f"trade_history_{timestamp}.csv")
        shutil.copy2(csv_path, archived_csv)
        print(f"[OK] Archived trade ledger to: {archived_csv}")

    if os.path.exists(state_file):
        archived_state = os.path.join(archive_dir, f"bot_state_{timestamp}.json")
        shutil.copy2(state_file, archived_state)
        print(f"[OK] Archived bot state to: {archived_state}")

    print("\nTo reset your bot with clean small-account capital, run:")
    print("  python bot.py --capital 30.0 --scan")


if __name__ == "__main__":
    archive_trade_history()
