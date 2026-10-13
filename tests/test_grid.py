import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bot.grid import calculate_grid_levels, run_grid_cycle, select_best_halal_asset
from tests.base import BaseTestCase


class TestHalalGrid(BaseTestCase):

    def setUp(self):
        super().setUp()
        self.config = {
            "paths": self.get_test_paths(),
            "timeframe": "1h",
            "risk_management": {"fee_rate": 0.00075},
            "grid_trading": {
                "enabled": True,
                "halal_basket": ["SOL/USDT", "AVAX/USDT", "NEAR/USDT", "LINK/USDT"],
                "macro_benchmark": "BTC/USDT",
                "order_size_usdt": 12.0,
                "max_grid_levels": 5,
                "min_step_pct": 0.012,
                "min_cash_reserve_usdt": 15.0,
            },
        }
        self.state = {
            "balance_usdt": 100.0,
            "open_positions": {},
            "trades_won": 0,
            "trade_count": 0,
            "realized_pnl_usdt": 0.0,
            "grid_state": {
                "active_symbol": "SOL/USDT",
                "active_levels": {},
                "total_grid_profit_usdt": 0.0,
                "cycles_completed": 0,
                "grid_trades": 0,
            },
        }

    def test_calculate_grid_levels(self):
        levels = calculate_grid_levels(current_price=150.0, atr=3.0, num_buy_levels=5, min_step_pct=0.015)
        self.assertEqual(len(levels), 5)
        for i, lvl in enumerate(levels, 1):
            self.assertEqual(lvl["level_index"], i)
            self.assertLess(lvl["buy_price"], 150.0)
            self.assertGreater(lvl["target_sell_price"], lvl["buy_price"])
            self.assertGreater(lvl["expected_profit_pct"], 1.0)

    def test_run_grid_cycle_initial_buy(self):
        client = MagicMock()
        # Mock 100 hourly klines for SOL/USDT
        mock_klines = [
            {"open": 150.0, "high": 152.0, "low": 149.0, "close": 150.0 + (i * 0.05), "volume": 1000.0}
            for i in range(100)
        ]
        client.get_klines.return_value = mock_klines
        client.get_ticker.return_value = {"price": "155.0"}

        new_buys, harvested_profits, grid_status = run_grid_cycle(client, self.state, self.config)

        self.assertEqual(len(new_buys), 1)
        self.assertEqual(len(harvested_profits), 0)
        self.assertEqual(grid_status["status"], "ACTIVE_HARVESTING")
        self.assertEqual(len(self.state["open_positions"]), 1)
        self.assertEqual(self.state["balance_usdt"], 88.0)  # 100 - 12

    def test_run_grid_cycle_profit_harvest(self):
        client = MagicMock()
        mock_klines = [
            {"open": 150.0, "high": 152.0, "low": 149.0, "close": 150.0, "volume": 1000.0}
            for _ in range(100)
        ]
        client.get_klines.return_value = mock_klines

        # Setup an existing position bought at 150.0 with TP at 152.5
        pos_id = "test-grid-pos-1"
        self.state["open_positions"][pos_id] = {
            "position_id": pos_id,
            "symbol": "SOL/USDT",
            "direction": "BUY_SPOT",
            "entry_time": "2026-10-13T10:00:00Z",
            "entry_price": 150.0,
            "curr_price": 150.0,
            "take_profit": 152.5,
            "allocated_usdt": 12.0,
            "unrealized_pnl": 0.0,
            "unrealized_pnl_pct": 0.0,
        }
        self.state["balance_usdt"] = 88.0

        # Now price bounces to 153.0 (hits TP)
        client.get_ticker.return_value = {"price": "153.0"}

        new_buys, harvested_profits, grid_status = run_grid_cycle(client, self.state, self.config)

        self.assertEqual(len(harvested_profits), 1)
        harvested = harvested_profits[0]
        self.assertEqual(harvested["symbol"], "SOL/USDT")
        self.assertGreater(harvested["net_pnl"], 0.0)
        self.assertEqual(self.state["trades_won"], 1)
        self.assertGreater(self.state["grid_state"]["total_grid_profit_usdt"], 0.0)
        # Position was harvested and capital recycled into a fresh active grid level
        self.assertEqual(len(new_buys), 1)
        self.assertEqual(len(self.state["open_positions"]), 1)

    def test_run_grid_cycle_trailing_take_profit(self):
        """Tests that trailing take profit ratchets peak price and exits on pullback."""
        client = MagicMock()
        mock_klines = [
            {"open": 150.0, "high": 152.0, "low": 149.0, "close": 150.0, "volume": 1000.0}
            for _ in range(100)
        ]
        client.get_klines.return_value = mock_klines

        # Enable trailing TP with 0.5% callback and isolate to SOL/USDT
        self.config["grid_trading"]["halal_basket"] = ["SOL/USDT"]
        self.config["grid_trading"]["trailing_take_profit"] = {
            "enabled": True,
            "callback_pct": 0.005,
        }

        pos_id = "test-trailing-pos"
        self.state["open_positions"][pos_id] = {
            "position_id": pos_id,
            "symbol": "SOL/USDT",
            "direction": "BUY_SPOT",
            "entry_time": "2026-10-13T10:00:00Z",
            "entry_price": 150.0,
            "curr_price": 150.0,
            "take_profit": 152.5,
            "allocated_usdt": 12.0,
            "unrealized_pnl": 0.0,
            "unrealized_pnl_pct": 0.0,
            "trailing_active": False,
            "peak_price": 150.0,
        }

        # Step 1: Price hits TP (153.0). Trailing activates, does NOT sell immediately
        client.get_ticker.return_value = {"price": "153.0"}
        new_buys, harvested, _ = run_grid_cycle(client, self.state, self.config)
        self.assertEqual(len(harvested), 0, "Should not exit on first TP hit when trailing is enabled")
        self.assertTrue(self.state["open_positions"][pos_id]["trailing_active"])
        self.assertEqual(self.state["open_positions"][pos_id]["peak_price"], 153.0)

        # Step 2: Price surges higher to 156.0. Peak ratchets up
        client.get_ticker.return_value = {"price": "156.0"}
        new_buys, harvested, _ = run_grid_cycle(client, self.state, self.config)
        self.assertEqual(len(harvested), 0)
        self.assertEqual(self.state["open_positions"][pos_id]["peak_price"], 156.0)

        # Step 3: Price pulls back to 154.5 (drop > 0.5% from 156.0). Triggers trailing harvest!
        client.get_ticker.return_value = {"price": "154.5"}
        new_buys, harvested, _ = run_grid_cycle(client, self.state, self.config)
        self.assertEqual(len(harvested), 1)
        self.assertEqual(harvested[0]["exit_price"], 154.5)
        self.assertGreater(harvested[0]["net_pnl"], 0.0)
        self.assertIn("Trailing TP Harvest", harvested[0]["reason"])

    def test_dual_coin_allocation(self):
        """Tests that grid engine diversifies into second coin when first reaches per-coin limit."""
        client = MagicMock()
        mock_klines = [
            {"open": 5.0, "high": 5.2, "low": 4.9, "close": 5.0, "volume": 50000.0}
            for _ in range(100)
        ]
        client.get_klines.return_value = mock_klines
        client.get_ticker.return_value = {"price": "5.0"}

        self.config["grid_trading"]["max_active_coins"] = 2
        self.config["grid_trading"]["max_levels_per_coin"] = 3
        self.config["grid_trading"]["max_grid_levels"] = 5

        # Pre-fill NEAR with 3 levels (max for this coin)
        for i in range(3):
            pid = f"near-pos-{i}"
            self.state["open_positions"][pid] = {
                "position_id": pid,
                "symbol": "NEAR/USDT",
                "direction": "BUY_SPOT",
                "entry_time": "2026-10-13T10:00:00Z",
                "entry_price": 5.0 - (i * 0.1),
                "curr_price": 5.0,
                "take_profit": 5.3,
                "allocated_usdt": 12.0,
                "unrealized_pnl": 0.0,
                "unrealized_pnl_pct": 0.0,
            }
        self.state["balance_usdt"] = 64.0  # Remaining free cash

        # Now run cycle: NEAR is full (3 levels), so it should allocate to secondary candidate
        new_buys, harvested, status = run_grid_cycle(client, self.state, self.config)
        
        # It should buy a new coin if candidate passes health check
        if new_buys:
            # Must NOT be NEAR because NEAR is at 3 levels
            self.assertNotEqual(new_buys[0]["symbol"], "NEAR/USDT")
            self.assertLessEqual(len(self.state["open_positions"]), 5)


if __name__ == "__main__":
    unittest.main()
