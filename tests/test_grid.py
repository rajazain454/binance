import os
import sys
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bot.grid import calculate_grid_levels, discover_top_momentum_symbols, run_grid_cycle, select_best_halal_asset
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

    def test_volume_surge_setup_detection(self):
        client = MagicMock()
        # 40 klines with flat 1000 vol, then last candle has 4000 vol (4x spike) and green close
        mock_klines = [
            {"open": 10.0, "high": 10.2, "low": 9.9, "close": 10.0, "volume": 1000.0}
            for _ in range(40)
        ]
        mock_klines.append({"open": 10.0, "high": 10.6, "low": 10.0, "close": 10.5, "volume": 4000.0})
        client.get_klines.return_value = mock_klines
        client.get_ticker.return_value = {"price": "10.5"}

        cfg = {
            "grid_trading": {
                "scan_all_pairs": False,
                "vol_spike_multiplier": 2.0,
                "mtf_guard_enabled": False,
            }
        }
        res = select_best_halal_asset(client, ["SOL/USDT"], config=cfg)
        self.assertIsNotNone(res)
        self.assertTrue(res["signal_triggered"])
        self.assertIn("Volume Surge", res["setup_name"])

    def test_mtf_guard_bear_suppression(self):
        client = MagicMock()
        # 15m klines show a bounce
        mock_klines_15m = [
            {"open": 10.0, "high": 10.2, "low": 9.9, "close": 10.0, "volume": 1000.0}
            for _ in range(40)
        ]
        mock_klines_15m.append({"open": 10.0, "high": 10.5, "low": 10.0, "close": 10.5, "volume": 3500.0})
        # 1h klines show a heavy downtrend (dropping from 25 to 10)
        mock_klines_1h = [
            {"open": 25.0 - (i * 0.4), "high": 25.0 - (i * 0.35), "low": 24.5 - (i * 0.4), "close": 24.8 - (i * 0.4), "volume": 2000.0}
            for i in range(35)
        ]

        def klines_side_effect(symbol, *args, **kwargs):
            tf = kwargs.get("timeframe") or kwargs.get("interval") or (args[0] if args else "15m")
            if tf == "1h":
                return mock_klines_1h
            return mock_klines_15m

        client.get_klines.side_effect = klines_side_effect
        client.get_ticker.return_value = {"price": "10.5"}

        cfg = {
            "grid_trading": {
                "scan_all_pairs": False,
                "vol_spike_multiplier": 2.0,
                "mtf_guard_enabled": True,
            }
        }
        res = select_best_halal_asset(client, ["SOL/USDT"], config=cfg)
        self.assertIsNotNone(res)
        self.assertFalse(res["signal_triggered"])  # Suppressed by MTF Guard
        self.assertIn("MTF Bear Block", res["setup_name"])

    def test_partial_profit_scaling_execution(self):
        client = MagicMock()
        mock_klines = [
            {"open": 150.0, "high": 152.0, "low": 149.0, "close": 150.0, "volume": 1000.0}
            for _ in range(100)
        ]
        client.get_klines.return_value = mock_klines

        self.config["grid_trading"]["trailing_take_profit"] = {"enabled": True, "callback_pct": 0.005}
        self.config["grid_trading"]["partial_take_profit"] = {"enabled": True, "tp1_pct": 0.015, "ratio": 0.5}

        pos_id = "partial-test-pos"
        self.state["open_positions"][pos_id] = {
            "position_id": pos_id,
            "symbol": "SOL/USDT",
            "direction": "BUY_SPOT",
            "entry_time": "2026-10-13T10:00:00Z",
            "entry_price": 100.0,
            "curr_price": 100.0,
            "stop_loss": 98.0,
            "take_profit": 104.0,
            "tp1_price": 101.8,
            "partial_tp_taken": False,
            "allocated_usdt": 20.0,
            "unrealized_pnl": 0.0,
            "unrealized_pnl_pct": 0.0,
        }
        self.state["balance_usdt"] = 80.0

        # Price reaches 102.0 (above tp1_price of 101.8, below final 104.0)
        client.get_ticker.return_value = {"price": "102.0"}
        new_buys, harvested, status = run_grid_cycle(client, self.state, self.config)

        self.assertEqual(len(harvested), 1)
        self.assertIn("Partial TP1 Harvest", harvested[0]["reason"])
        # Half of allocated (10.0) should remain open
        self.assertIn(pos_id, self.state["open_positions"])
        self.assertEqual(self.state["open_positions"][pos_id]["allocated_usdt"], 10.0)
        self.assertTrue(self.state["open_positions"][pos_id]["partial_tp_taken"])
        # Stop loss should be ratcheted to guaranteed win
        self.assertGreaterEqual(self.state["open_positions"][pos_id]["stop_loss"], 100.3)

    def test_discover_top_momentum_symbols(self):
        client = MagicMock()
        client.fetch_all_tickers.return_value = {
            "BTC/USDT": {"quoteVolume": 50000000.0, "percentage": 2.5, "last": 65000.0},
            "PEPE/USDT": {"quoteVolume": 15000000.0, "percentage": 14.8, "last": 0.00001},
            "LOWVOL/USDT": {"quoteVolume": 50000.0, "percentage": 25.0, "last": 1.0},
            "BTCDOWN/USDT": {"quoteVolume": 5000000.0, "percentage": 30.0, "last": 2.0},
            "SOL/USDT": {"quoteVolume": 20000000.0, "percentage": 8.2, "last": 150.0},
        }
        runners = discover_top_momentum_symbols(client, min_volume_usdt=2000000.0, max_symbols=3)
        self.assertIn("PEPE/USDT", runners)
        self.assertIn("SOL/USDT", runners)
        self.assertNotIn("LOWVOL/USDT", runners)
        self.assertNotIn("BTCDOWN/USDT", runners)
        self.assertEqual(runners[0], "PEPE/USDT")


    def test_dynamic_atr_tp_sl_scaling(self):
        client = MagicMock()
        # High volatility klines with ATR ~ 5%
        mock_klines = [
            {"open": 100.0, "high": 105.0, "low": 95.0, "close": 100.0 + (i * 0.1), "volume": 1000.0}
            for i in range(100)
        ]
        client.get_klines.return_value = mock_klines
        client.get_ticker.return_value = {"price": "110.0"}

        cfg = {
            "paths": self.get_test_paths(),
            "timeframe": "15m",
            "risk_management": {"fee_rate": 0.00075},
            "grid_trading": {
                "enabled": True,
                "halal_basket": ["SOL/USDT"],
                "macro_benchmark": "BTC/USDT",
                "order_size_usdt": 20.0,
                "max_grid_levels": 2,
                "target_tp_pct": 0.02,
                "stop_loss_pct": 0.03,
                "atr_tp_multiplier": 1.8,
                "atr_sl_multiplier": 1.2,
                "require_confluence_signal": False,
            },
        }
        state = {
            "balance_usdt": 100.0,
            "open_positions": {},
            "grid_state": {"active_symbol": "SOL/USDT", "total_grid_profit_usdt": 0.0, "cycles_completed": 0, "grid_trades": 0},
        }
        new_buys, _, _ = run_grid_cycle(client, state, cfg)
        self.assertEqual(len(new_buys), 1)
        buy = new_buys[0]
        # Volatility is high so TP must be expanded beyond baseline 2%
        expected_min_tp = buy["entry_price"] * 1.03
        self.assertGreater(buy["take_profit"], expected_min_tp)
        self.assertIn("tp1_price", buy)
        self.assertFalse(buy["partial_tp_taken"])

    def test_upper_wick_rejection_filter(self):
        client = MagicMock()
        # Create klines where last candle is green (close > open) but has a massive 60% upper wick (shooting star)
        # Open 10.0, Low 10.0, Close 10.4, High 11.0 (Range 1.0, Upper wick = 11.0 - 10.4 = 0.60 -> 60%)
        mock_klines = [
            {"open": 10.0, "high": 10.2, "low": 9.9, "close": 10.0, "volume": 1000.0}
            for _ in range(40)
        ]
        mock_klines.append({"open": 10.0, "high": 11.0, "low": 10.0, "close": 10.4, "volume": 3500.0})
        client.get_klines.return_value = mock_klines
        client.get_ticker.return_value = {"price": "10.4"}

        cfg = {
            "grid_trading": {
                "scan_all_pairs": False,
                "vol_spike_multiplier": 2.0,
                "mtf_guard_enabled": False,
                "anti_fakeout": {"enabled": True, "max_wick_ratio": 0.35},
            }
        }
        res = select_best_halal_asset(client, ["SOL/USDT"], config=cfg)
        self.assertIsNotNone(res)
        # Signal should NOT trigger because the upper wick is 60% > 35%
        self.assertFalse(res["signal_triggered"])
        self.assertIn("Topping Wick", res["setup_name"])

    def test_time_decay_stale_exit(self):
        client = MagicMock()
        mock_klines = [
            {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}
            for _ in range(100)
        ]
        client.get_klines.return_value = mock_klines

        # Set position entered 6 hours ago
        six_hours_ago = (datetime.now(timezone.utc) - timedelta(hours=6)).isoformat()
        pos_id = "stale-pos"
        self.state["open_positions"][pos_id] = {
            "position_id": pos_id,
            "symbol": "SOL/USDT",
            "direction": "BUY_SPOT",
            "entry_time": six_hours_ago,
            "entry_price": 100.0,
            "curr_price": 100.2,
            "stop_loss": 98.0,
            "take_profit": 103.0,
            "allocated_usdt": 20.0,
            "unrealized_pnl": 0.04,
            "unrealized_pnl_pct": 0.2,
            "trailing_active": False,
        }
        self.config["grid_trading"]["stale_exit"] = {"enabled": True, "hours": 4.0, "max_pnl_threshold_pct": 1.0}
        client.get_ticker.return_value = {"price": "100.2"}

        new_buys, harvested, status = run_grid_cycle(client, self.state, self.config)
        self.assertEqual(len(harvested), 1)
        self.assertIn("Stale Capital Exit", harvested[0]["reason"])
        self.assertNotIn(pos_id, self.state["open_positions"])

    def test_spread_guard_rejection(self):
        client = MagicMock()
        mock_klines = [
            {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}
            for _ in range(100)
        ]
        client.get_klines.return_value = mock_klines
        # Return ticker with 0.5% spread (bid 100, ask 100.5)
        client.get_ticker.return_value = {"price": "100.0", "bid": 100.0, "ask": 100.5}

        self.config["grid_trading"]["spread_guard"] = {"enabled": True, "max_spread_pct": 0.0015}
        self.config["grid_trading"]["require_confluence_signal"] = False

        new_buys, _, _ = run_grid_cycle(client, self.state, self.config)
        # Should be rejected because 0.5% > 0.15% max spread
        self.assertEqual(len(new_buys), 0)

    def test_pullback_retrace_entry_calibration(self):
        client = MagicMock()
        mock_klines = [
            {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}
            for _ in range(100)
        ]
        client.get_klines.return_value = mock_klines
        client.get_ticker.return_value = {"price": "100.0"}

        cfg = {
            "paths": self.get_test_paths(),
            "timeframe": "15m",
            "risk_management": {"fee_rate": 0.00075},
            "grid_trading": {
                "enabled": True,
                "halal_basket": ["SOL/USDT"],
                "macro_benchmark": "BTC/USDT",
                "order_size_usdt": 20.0,
                "max_grid_levels": 2,
                "pullback_retrace": {"enabled": True, "retrace_pct": 0.01}, # 1% retrace
                "require_confluence_signal": False,
            },
        }
        state = {
            "balance_usdt": 100.0,
            "open_positions": {},
            "grid_state": {"active_symbol": "SOL/USDT", "total_grid_profit_usdt": 0.0, "cycles_completed": 0, "grid_trades": 0},
        }
        new_buys, _, _ = run_grid_cycle(client, state, cfg)
        self.assertEqual(len(new_buys), 1)
        # Entry price should be discounted by retrace percentage rather than buying 100.0 peak
        self.assertLess(new_buys[0]["entry_price"], 100.0)

    def test_24h_high_proximity_filter_exhaustion(self):
        client = MagicMock()
        client.fetch_all_tickers.return_value = {
            # Coin A: Up 25% but dumped 10% from high (high 1.10, last 0.99) -> dump is 10%
            "DUMP/USDT": {"quoteVolume": 5000000.0, "percentage": 25.0, "last": 0.99, "high": 1.10},
            # Coin B: Up 15% and near high (high 1.02, last 1.00) -> dump is 2%
            "FRESH/USDT": {"quoteVolume": 5000000.0, "percentage": 15.0, "last": 1.00, "high": 1.02},
            # Coin C: Overextended parabolic (+40% gain)
            "PARABOLIC/USDT": {"quoteVolume": 5000000.0, "percentage": 40.0, "last": 2.0, "high": 2.0},
        }
        runners = discover_top_momentum_symbols(client, min_volume_usdt=2000000.0, max_symbols=5, max_dump_from_high_pct=7.0, max_daily_gain_pct=35.0)
        self.assertIn("FRESH/USDT", runners)
        self.assertNotIn("DUMP/USDT", runners)        # Rejected due to 10% dump off 24h high
        self.assertNotIn("PARABOLIC/USDT", runners)   # Rejected due to >35% gain

    def test_volatility_sizing_risk_parity(self):
        client = MagicMock()
        mock_klines = [
            {"open": 100.0, "high": 102.0, "low": 98.0, "close": 100.0, "volume": 1000.0}
            for _ in range(100)
        ]
        client.get_klines.return_value = mock_klines
        client.get_ticker.return_value = {"price": "100.0"}

        cfg = {
            "paths": self.get_test_paths(),
            "timeframe": "15m",
            "risk_management": {"fee_rate": 0.00075},
            "grid_trading": {
                "enabled": True,
                "halal_basket": ["SOL/USDT"],
                "macro_benchmark": "BTC/USDT",
                "order_size_usdt": 20.0,
                "max_grid_levels": 2,
                "stop_loss_pct": 0.02, # 2% stop loss
                "volatility_sizing": {"enabled": True, "target_risk_usdt": 0.60, "min_order_usdt": 18.0, "max_order_usdt": 35.0},
                "require_confluence_signal": False,
            },
        }
        state = {
            "balance_usdt": 100.0,
            "open_positions": {},
            "grid_state": {"active_symbol": "SOL/USDT", "total_grid_profit_usdt": 0.0, "cycles_completed": 0, "grid_trades": 0},
        }
        new_buys, _, _ = run_grid_cycle(client, state, cfg)
        self.assertEqual(len(new_buys), 1)
        # Sizing with 0.60 risk / 0.02 stop = $30.00 allocated
        self.assertAlmostEqual(new_buys[0]["allocated_usdt"], 30.0, delta=1.0)

    def test_early_momentum_invalidation_guard(self):
        client = MagicMock()
        # 10 klines ending with 2 red candles below EMA9
        mock_klines = [
            {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}
            for _ in range(40)
        ]
        mock_klines.append({"open": 100.0, "high": 100.2, "low": 99.0, "close": 99.2, "volume": 800.0}) # Red candle 1
        mock_klines.append({"open": 99.2, "high": 99.3, "low": 98.5, "close": 98.8, "volume": 700.0})   # Red candle 2
        client.get_klines.return_value = mock_klines
        client.get_ticker.return_value = {"price": "98.8"}

        pos_id = "inv-pos"
        self.state["open_positions"][pos_id] = {
            "position_id": pos_id,
            "symbol": "SOL/USDT",
            "direction": "BUY_SPOT",
            "entry_time": "2026-10-13T10:00:00Z",
            "entry_price": 100.0,
            "curr_price": 98.8,
            "stop_loss": 97.5, # Regular stop loss is far below (at 97.5)
            "take_profit": 103.0,
            "allocated_usdt": 20.0,
            "unrealized_pnl": -0.24,
            "unrealized_pnl_pct": -1.2,
            "trailing_active": False,
        }
        self.config["grid_trading"]["early_momentum_invalidation"] = {"enabled": True, "loss_threshold_pct": 0.5}

        new_buys, harvested, status = run_grid_cycle(client, self.state, self.config)
        self.assertEqual(len(harvested), 1)
        self.assertIn("Early Momentum Invalidation", harvested[0]["reason"])
        self.assertNotIn(pos_id, self.state["open_positions"])


if __name__ == "__main__":
    unittest.main()
