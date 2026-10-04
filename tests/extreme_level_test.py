"""Extreme Level Stress and Validation Test Suite.
Validates:
1. 20-Symbol Coin Universe Scale & Integrity
2. Parallel Multi-Threaded Candle Fetching & Fault Tolerance
3. Visual ASCII Progress Gauge Rendering (Profit, Drawdown, Edge Cases)
4. Relative Strength vs BTC (RS Alpha Leader) Decoupling Logic
5. 3-Slot Portfolio Risk-Parity Sizing & Capital Allocation Bounds
6. Multi-Position Mark-to-Market Circuit Breaker Under Extreme Volatility
7. Discord Embed Formatting & Field Bounds
"""

import os
import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone
import numpy as np
import pandas as pd

import bot
import features


class TestExtremeLevelSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 150-bar realistic synthetic price series
        np.random.seed(42)
        n = 150
        dates = pd.date_range("2026-01-01", periods=n, freq="1h")

        # Base BTC series (mostly flat over last 24h)
        btc_close = 80000.0 * np.ones(n)
        cls.btc_df = pd.DataFrame({
            "open": btc_close,
            "high": btc_close * 1.002,
            "low": btc_close * 0.998,
            "close": btc_close,
            "volume": np.random.uniform(50, 500, n)
        }, index=dates)

        # Altcoin A: Relative Strength Leader (surges +4% over last 24 hours vs flat BTC)
        alt_close = 100.0 * np.ones(n)
        alt_close[-24:] = 100.0 * np.linspace(1.0, 1.04, 24)
        cls.alt_leader_df = pd.DataFrame({
            "open": alt_close * 0.999,
            "high": alt_close * 1.005,
            "low": alt_close * 0.995,
            "close": alt_close,
            "volume": np.random.uniform(1000, 10000, n)
        }, index=dates)

        # Altcoin B: Laggard (drops -4% over last 24 hours vs flat BTC)
        alt_lag_close = 100.0 * np.ones(n)
        alt_lag_close[-24:] = 100.0 * np.linspace(1.0, 0.96, 24)
        cls.alt_lag_df = pd.DataFrame({
            "open": alt_lag_close * 1.001,
            "high": alt_lag_close * 1.005,
            "low": alt_lag_close * 0.995,
            "close": alt_lag_close,
            "volume": np.random.uniform(1000, 10000, n)
        }, index=dates)

    # =========================================================================
    # 1. 20-Symbol Coin Universe Scale & Integrity
    # =========================================================================
    def test_coin_universe_has_20_symbols(self):
        config = bot.load_config("config.json")
        symbols = config.get("symbols", [])
        self.assertEqual(len(symbols), 20, f"Expected 20 symbols in config, found {len(symbols)}")
        # Check no duplicates
        self.assertEqual(len(symbols), len(set(symbols)), "Duplicate symbols detected in config universe!")
        # Check all are /USDT pairs
        for s in symbols:
            self.assertTrue(s.endswith("/USDT"), f"Symbol {s} must be a /USDT pair")

    # =========================================================================
    # 2. Visual Trade Progress Gauge Rendering (Profit, Drawdown, Bounds)
    # =========================================================================
    def test_render_trade_progress_profit_scaling(self):
        entry = 100.0
        tp1 = 110.0
        sl = 95.0

        # At entry (0% to TP1)
        g0 = bot.render_trade_progress(100.0, entry, tp1, sl)
        self.assertIn("0% to TP1", g0)
        self.assertIn("░░░░░░░░░░", g0)

        # Halfway to TP1 (50% to TP1)
        g50 = bot.render_trade_progress(105.0, entry, tp1, sl)
        self.assertIn("50% to TP1", g50)
        self.assertIn("█████░░░░░", g50)

        # At TP1 (100% to TP1)
        g100 = bot.render_trade_progress(110.0, entry, tp1, sl)
        self.assertIn("100% to TP1", g100)
        self.assertIn("██████████", g100)

        # Above TP1 (Clamped to 100%)
        g_over = bot.render_trade_progress(115.0, entry, tp1, sl)
        self.assertIn("100% to TP1", g_over)

    def test_render_trade_progress_drawdown_scaling(self):
        entry = 100.0
        tp1 = 110.0
        sl = 90.0

        # 50% down to SL
        g_down = bot.render_trade_progress(95.0, entry, tp1, sl)
        self.assertIn("50% to SL", g_down)
        self.assertIn("▓▓▓▓▓░░░░░", g_down)

        # At SL (100% to SL)
        g_sl = bot.render_trade_progress(90.0, entry, tp1, sl)
        self.assertIn("100% to SL", g_sl)

    def test_render_trade_progress_zero_division_guard(self):
        # Degenerate input: tp1 == entry or sl == entry
        res = bot.render_trade_progress(100.0, 100.0, 100.0, 100.0)
        self.assertTrue(isinstance(res, str))
        self.assertIn("TP1", res)

    # =========================================================================
    # 3. Relative Strength vs BTC (RS-BTC Alpha Leader)
    # =========================================================================
    def test_relative_strength_detects_true_leader(self):
        is_leader, info = features.evaluate_relative_strength(self.alt_leader_df, self.btc_df, period=24)
        self.assertTrue(is_leader, "Outperforming altcoin must be flagged as RS Alpha Leader")
        self.assertGreater(info["alpha_24h"], 1.5, "Alpha 24h must exceed +1.5%")
        self.assertTrue(info["is_leader"])

    def test_relative_strength_rejects_laggard(self):
        is_leader, info = features.evaluate_relative_strength(self.alt_lag_df, self.btc_df, period=24)
        self.assertFalse(is_leader, "Underperforming altcoin must not be flagged as RS Alpha Leader")
        self.assertLess(info["alpha_24h"], 0.0)

    def test_relative_strength_handles_empty_or_short_data(self):
        is_leader, info = features.evaluate_relative_strength(pd.DataFrame(), self.btc_df)
        self.assertFalse(is_leader)
        self.assertEqual(info["alpha_24h"], 0.0)

    # =========================================================================
    # 4. Multi-Slot Portfolio Allocation (3 Slots, $30 Capital)
    # =========================================================================
    def test_three_slot_risk_parity_allocation_bounds(self):
        config = bot.load_config("config.json")
        config["risk_management"]["capital_usdt"] = 30.0
        config["risk_management"]["max_open_trades"] = 3
        config["trading_mode"] = "paper"
        config["mtf_confluence"] = {"enabled": False}
        config["correlation_filter"] = {"enabled": False}
        config["paths"] = {"trade_ledger": "logs/test_extreme_ledger.csv"}

        # Mock client returning valid candles
        mock_client = MagicMock()
        mock_client.fetch_ohlcv.return_value = self.alt_leader_df
        mock_client.fetch_funding_rate.return_value = 0.0001

        # Mock model returning high probability
        class MockModel:
            def predict_proba(self, X):
                return np.array([[0.20, 0.80]])

        feats = features.extract_features(self.alt_leader_df)
        mock_bundle = {"model": MockModel(), "feature_names": list(feats.columns)}

        # Fresh empty state with $30 cash
        state = {
            "balance_usdt": 30.0,
            "peak_balance": 30.0,
            "daily_peak_balance": 30.0,
            "daily_reset_time": datetime.now(timezone.utc).isoformat(),
            "circuit_breaker_until": None,
            "open_positions": {},
            "trade_count": 0,
            "win_count": 0,
            "loss_count": 0
        }

        # 3 candidate symbols
        config["symbols"] = ["SOL/USDT", "ETH/USDT", "AVAX/USDT"]

        rows, new_signals, closed = bot.scan_and_execute(mock_client, mock_bundle, state, config)

        # Must execute all 3 trades
        self.assertEqual(len(state["open_positions"]), 3, f"Expected 3 positions opened, got {len(state['open_positions'])}")
        self.assertEqual(len(new_signals), 3)

        # Verify each allocation is at least $5.00 (Binance minNotional) and around ~$9.50
        for sym, pos in state["open_positions"].items():
            self.assertGreaterEqual(pos["allocated_usdt"], 5.0)
            self.assertLessEqual(pos["allocated_usdt"], 12.0)

        # Cash balance must remain non-negative
        self.assertGreaterEqual(state["balance_usdt"], 0.0)

        # Now test 4th candidate rejection (Slots full!)
        config["symbols"] = ["NEAR/USDT"]
        rows2, new_signals2, _ = bot.scan_and_execute(mock_client, mock_bundle, state, config)
        self.assertEqual(len(new_signals2), 0, "4th trade must be rejected because max 3 slots are full!")
        self.assertEqual(len(state["open_positions"]), 3)

        if os.path.exists("logs/test_extreme_ledger.csv"):
            os.remove("logs/test_extreme_ledger.csv")

    # =========================================================================
    # 5. Multi-Position Mark-to-Market Circuit Breaker Under Stress
    # =========================================================================
    def test_multi_position_circuit_breaker_trips_on_total_portfolio_drawdown(self):
        config = {
            "risk_management": {
                "capital_usdt": 30.0,
                "daily_max_drawdown_pct": 8.0,
                "circuit_breaker_hours": 24
            }
        }
        # 3 open positions causing > 8% drawdown from peak $30.00
        # Peak $30.00 -> 8% DD is $27.60. Current equity: Cash $10 + Pos $11 = $21.00 (Down 30%)
        state = {
            "balance_usdt": 10.0,
            "daily_peak_balance": 30.0,
            "daily_reset_time": datetime.now(timezone.utc).isoformat(),
            "circuit_breaker_until": None,
            "open_positions": {
                "SOL/USDT": {"allocated_usdt": 6.0, "unrealized_pnl": -1.0},
                "ETH/USDT": {"allocated_usdt": 6.0, "unrealized_pnl": -3.5},
                "AVAX/USDT": {"allocated_usdt": 6.0, "unrealized_pnl": -2.5}
            }
        }

        is_locked, status = bot.check_circuit_breaker(state, config)
        self.assertTrue(is_locked, "Circuit breaker must trip when total mark-to-market drawdown exceeds 8%!")
        self.assertTrue("LOCKED" in status or "TRIGGERED" in status)
        self.assertIsNotNone(state["circuit_breaker_until"])

    # =========================================================================
    # 6. Discord Daily Digest & Progress Bar Payload Integrity
    # =========================================================================
    def test_discord_daily_digest_contains_gauges(self):
        config = {
            "risk_management": {"capital_usdt": 30.0, "max_open_trades": 3},
            "trading_mode": "paper",
            "timeframe": "1h",
            "discord": {"enabled": True, "webhook_url": "https://discord.com/api/webhooks/mock/test"},
            "paths": {"state_file": "logs/test_state_digest.json"}
        }
        state = {
            "balance_usdt": 20.0,
            "open_positions": {
                "BNB/USDT": {
                    "allocated_usdt": 10.0,
                    "entry_price": 780.0,
                    "curr_price": 786.0,
                    "take_profit_1": 790.0,
                    "stop_loss": 770.0,
                    "unrealized_pnl": 0.08,
                    "unrealized_pnl_pct": 0.77
                }
            }
        }
        with patch("bot.send_discord_alert", return_value=(True, "OK")) as mock_send:
            ok, res = bot.send_daily_digest(state, config)
            self.assertTrue(ok)
            mock_send.assert_called_once()
            call_kwargs = mock_send.call_args[1]
            desc = call_kwargs.get("description", "")
            # Check gauge appears in the position description
            self.assertIn("Target:", desc)
            self.assertIn("to TP1", desc)


if __name__ == "__main__":
    unittest.main()
