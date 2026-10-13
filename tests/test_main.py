"""End-to-end unit tests for bot/main.py execute_cycle."""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bot.main import execute_cycle
from tests.base import BaseTestCase


class TestMainCycle(BaseTestCase):

    def test_execute_cycle_end_to_end(self):
        config = {
            "trading_mode": "paper",
            "timeframe": "1h",
            "paths": self.get_test_paths(),
            "risk_management": {
                "capital_usdt": 100.0,
                "fee_rate": 0.00075,
            },
            "grid_trading": {
                "enabled": True,
                "halal_basket": ["NEAR/USDT"],
                "macro_benchmark": "BTC/USDT",
                "order_size_usdt": 12.0,
                "max_grid_levels": 5,
                "min_step_pct": 0.012,
                "min_cash_reserve_usdt": 15.0,
            },
            "discord": {"enabled": False},
        }

        client = MagicMock()
        mock_klines = [
            {"open": 5.0, "high": 5.2, "low": 4.9, "close": 5.1 + (i * 0.01), "volume": 50000.0}
            for i in range(100)
        ]
        client.get_klines.return_value = mock_klines
        client.get_ticker_price.return_value = 5.25

        with patch("bot.main.print_dashboard") as mock_dash:
            execute_cycle(client, config)
            self.assertTrue(mock_dash.called)

        # Confirm state file was written
        self.assertTrue(os.path.exists(self.state_file))


if __name__ == "__main__":
    unittest.main()
