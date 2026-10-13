"""Unit tests for bot/stats.py and ledger expectancy metrics."""

import os
import sys
import unittest
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bot.stats import compute_ledger_stats
from tests.base import BaseTestCase


class TestLedgerStats(BaseTestCase):

    def test_empty_ledger_stats(self):
        stats = compute_ledger_stats(self.trade_ledger)
        self.assertEqual(stats["closed_positions"], 0)
        self.assertEqual(stats["win_rate_pct"], 0.0)
        self.assertEqual(stats["profit_factor"], 0.0)

    def test_populated_ledger_stats(self):
        df = pd.DataFrame([
            {"timestamp": "2026-01-01", "position_id": "p1", "symbol": "NEAR/USDT", "action": "GRID_HARVEST", "net_pnl": 0.50, "outcome": "WIN"},
            {"timestamp": "2026-01-02", "position_id": "p2", "symbol": "SOL/USDT", "action": "GRID_HARVEST", "net_pnl": 0.40, "outcome": "WIN"},
            {"timestamp": "2026-01-03", "position_id": "p3", "symbol": "AVAX/USDT", "action": "GRID_HARVEST", "net_pnl": -0.20, "outcome": "LOSS"},
        ])
        df.to_csv(self.trade_ledger, index=False)

        stats = compute_ledger_stats(self.trade_ledger)
        self.assertEqual(stats["closed_positions"], 3)
        self.assertEqual(stats["win_count"], 2)
        self.assertEqual(stats["loss_count"], 1)
        self.assertAlmostEqual(stats["win_rate_pct"], 66.67, places=1)
        self.assertAlmostEqual(stats["total_net_pnl"], 0.70, places=2)
        self.assertGreater(stats["profit_factor"], 4.0)


if __name__ == "__main__":
    unittest.main()
