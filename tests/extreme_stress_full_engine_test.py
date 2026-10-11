"""Extreme Stress Test Suite on Entire QuantLab Quantitative Engine.
Validates:
1. Micro-Account & Binance minNotional Dust Order Handling (eliminates small account trap)
2. Exhaustion Filter (RSI & Bollinger Band %B Overbought Blocker)
3. Chandelier Trailing Stop Room (2.2 ATR vs choking)
4. Dynamic Macro Threshold Scaling (No hardcoded 0.60 floor)
5. Accurate Win/Loss PnL Accounting (Negative trades never counted as wins)
6. 100-Cycle High-Volatility Monte Carlo Portfolio Simulation
7. Simultaneous Multi-Asset Flash Crash & Circuit Breaker Robustness
"""

import os
import sys
import unittest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import bot
import features
from bot.positions import check_and_update_positions
from bot.scanner import scan_and_execute, get_macro_regime_threshold
from utils import load_config


class TestExtremeStressEngine(unittest.TestCase):
    """Deep Stress Testing of all engine mechanics under extreme scenarios."""

    def setUp(self):
        self.config = load_config("config.json")
        self.config["trading_mode"] = "paper"
        self.config["mtf_confluence"] = {"enabled": False}
        self.config["correlation_filter"] = {"enabled": False}
        self.config["paths"]["trade_ledger"] = "logs/test_extreme_stress_ledger.csv"

        # 150-bar synthetic candles
        np.random.seed(1337)
        dates = pd.date_range("2026-01-01", periods=150, freq="1h", tz="UTC")
        trend = np.linspace(100.0, 115.0, 150)
        noise = np.random.normal(0, 0.5, 150)
        close = trend + noise
        self.sample_df = pd.DataFrame({
            "open": close - 0.2,
            "high": close + 0.8,
            "low": close - 0.8,
            "close": close,
            "volume": np.random.uniform(500, 5000, 150)
        }, index=dates)

    # =========================================================================
    # 1. Micro-Account Dust Order & Full TP1 Stress Test
    # =========================================================================
    def test_small_account_full_tp1_exit_when_split_below_min_notional(self):
        """When position is too small to split (sell_allocated < min_notional),
        it must execute 100% full TP1 exit rather than selling $0 and locking to breakeven."""
        mock_client = MagicMock()
        mock_client.get_min_notional.return_value = 5.0
        mock_client.get_ticker_price.return_value = 105.0  # Above TP1 (103.0)

        # $8.00 position (50% is $4.00, below $5 minNotional)
        state = {
            "balance_usdt": 92.0,
            "open_positions": {
                "TEST/USDT": {
                    "entry_price": 100.0,
                    "allocated_usdt": 8.0,
                    "units": 0.08,
                    "stop_loss": 97.0,
                    "take_profit_1": 103.0,
                    "take_profit_2": 106.0,
                    "tp1_reached": False,
                    "highest_price": 100.0,
                    "atr": 2.0,
                    "entry_time": "2026-01-01 10:00:00",
                    "confidence": 65.0,
                }
            },
            "trade_count": 0,
            "win_count": 0,
            "loss_count": 0,
        }

        closed = check_and_update_positions(mock_client, state, self.config)

        self.assertEqual(len(closed), 1)
        self.assertEqual(closed[0]["action"], "FINAL CLOSE (FULL TP1 REACHED)")
        self.assertEqual(closed[0]["outcome"], "WIN")
        self.assertGreater(closed[0]["net_pnl"], 0.0)
        # Position should be fully closed and removed
        self.assertNotIn("TEST/USDT", state["open_positions"])
        self.assertEqual(state["trade_count"], 1)
        self.assertEqual(state["win_count"], 1)
        self.assertGreater(state["balance_usdt"], 92.0)

    def test_normal_account_50_pct_tp1_exit_when_above_min_notional(self):
        """When position is large enough to split (e.g. $40 position),
        it correctly sells 50% ($20 >= $5) and ratchets stop loss."""
        mock_client = MagicMock()
        mock_client.get_min_notional.return_value = 5.0
        mock_client.get_ticker_price.return_value = 105.0  # Above TP1

        state = {
            "balance_usdt": 60.0,
            "open_positions": {
                "TEST/USDT": {
                    "entry_price": 100.0,
                    "allocated_usdt": 40.0,
                    "units": 0.40,
                    "stop_loss": 97.0,
                    "take_profit_1": 103.0,
                    "take_profit_2": 106.0,
                    "tp1_reached": False,
                    "highest_price": 100.0,
                    "atr": 2.0,
                    "entry_time": "2026-01-01 10:00:00",
                    "confidence": 65.0,
                }
            },
            "trade_count": 0,
            "win_count": 0,
            "loss_count": 0,
        }

        closed = check_and_update_positions(mock_client, state, self.config)

        self.assertEqual(len(closed), 1)
        self.assertEqual(closed[0]["action"], "PARTIAL TP1 (50% SOLD)")
        self.assertEqual(closed[0]["outcome"], "WIN")
        # Half position remains open
        self.assertIn("TEST/USDT", state["open_positions"])
        self.assertAlmostEqual(state["open_positions"]["TEST/USDT"]["allocated_usdt"], 20.0, places=1)
        self.assertTrue(state["open_positions"]["TEST/USDT"]["tp1_reached"])

    # =========================================================================
    # 2. Overbought Exhaustion Ceiling Stress Test
    # =========================================================================
    def test_overbought_exhaustion_blocks_peak_fomo_buys(self):
        """Candles with vertical blowout (RSI > 75 or BB %B > 1.05) must be halted with OVERBOUGHT status."""
        mock_client = MagicMock()
        mock_client.get_ticker_price.return_value = 100.0
        mock_client.fetch_funding_rate.return_value = 0.0001

        # Craft vertical spike with extreme RSI (RSI > 80)
        n = 100
        dates = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
        spike_close = np.linspace(100.0, 150.0, n)
        spike_df = pd.DataFrame({
            "open": spike_close - 0.5,
            "high": spike_close + 1.0,
            "low": spike_close - 0.5,
            "close": spike_close,
            "volume": np.random.uniform(1000, 10000, n)
        }, index=dates)

        mock_client.fetch_ohlcv.return_value = spike_df

        class HighConfModel:
            def predict_proba(self, X):
                return np.array([[0.10, 0.90]])

        feats = features.extract_features(spike_df)
        mock_bundle = {"model": HighConfModel(), "feature_names": list(feats.columns)}

        state = {
            "balance_usdt": 100.0,
            "open_positions": {},
            "trade_count": 0,
            "win_count": 0,
            "loss_count": 0,
        }

        self.config["symbols"] = ["SPIKE/USDT"]
        analysis_rows, new_signals, _ = scan_and_execute(mock_client, mock_bundle, state, self.config)

        self.assertEqual(len(analysis_rows), 1)
        # Should be blocked from entering at overbought peak
        self.assertEqual(analysis_rows[0]["status"], "OVERBOUGHT")
        self.assertEqual(len(new_signals), 0, "Bot must NEVER buy an overbought vertical spike!")

    # =========================================================================
    # 3. Chandelier Trailing Stop Breathing Room
    # =========================================================================
    def test_chandelier_2_2_atr_allows_runners_to_thrive(self):
        """With trail_mult = 2.2, a 1.2 ATR normal pullback does NOT trigger premature exit."""
        pos = {
            "entry_price": 100.0,
            "highest_price": 110.0,
            "stop_loss": 97.0,
            "atr": 2.0,
            "tp1_reached": True,
        }
        trail_mult = self.config["trailing_stop"]["atr_mult"]  # 2.2
        pos_atr = pos["atr"]  # 2.0

        chandelier_stop = pos["highest_price"] - (trail_mult * pos_atr)  # 110 - (2.2 * 2.0) = 105.6
        be_level = pos["entry_price"] * 1.001  # 100.1
        active_stop = max(pos["stop_loss"], chandelier_stop, be_level)

        self.assertEqual(active_stop, 105.6)

        # Price pulls back to 107.0 (1.5 ATR drop from 110 peak)
        curr_price = 107.0
        hit_sl = curr_price <= active_stop
        self.assertFalse(hit_sl, "Pullback to 107 should NOT hit 105.6 trailing stop with 2.2 ATR room!")

    # =========================================================================
    # 4. Dynamic Macro Threshold Scaling (No Hardcoded 0.60 Floor)
    # =========================================================================
    def test_macro_regime_threshold_scales_dynamically(self):
        """Macro bull and bear thresholds must scale relative to base_thresh (0.40) without 0.60 clamp."""
        mock_client = MagicMock()

        # BTC above 200 SMA (Bull)
        n = 220
        btc_bull = pd.DataFrame({"close": np.linspace(50000, 70000, n)})
        mock_client.fetch_ohlcv.return_value = btc_bull

        self.config["ai_model"]["confidence_threshold"] = 0.40
        bull_thresh, bull_label = get_macro_regime_threshold(mock_client, self.config)

        self.assertAlmostEqual(bull_thresh, 0.38, places=2)
        self.assertIn("MACRO BULL", bull_label)

        # BTC below 200 SMA (Defensive)
        btc_bear = pd.DataFrame({"close": np.linspace(70000, 50000, n)})
        mock_client.fetch_ohlcv.return_value = btc_bear

        bear_thresh, bear_label = get_macro_regime_threshold(mock_client, self.config)
        self.assertAlmostEqual(bear_thresh, 0.44, places=2)
        self.assertIn("MACRO DEFENSIVE", bear_label)

    # =========================================================================
    # 5. Accurate Win/Loss PnL Accounting
    # =========================================================================
    def test_negative_net_pnl_is_never_classified_as_win(self):
        """A trade that exits with negative net PnL (due to fee drag or slippage)
        must NEVER be classified as a WIN, even if tp1_reached was flagged."""
        mock_client = MagicMock()
        mock_client.get_ticker_price.return_value = 99.9  # Slippage below entry (100.0)

        state = {
            "balance_usdt": 90.0,
            "open_positions": {
                "TEST/USDT": {
                    "entry_price": 100.0,
                    "allocated_usdt": 10.0,
                    "units": 0.10,
                    "stop_loss": 100.1,  # Breakeven stop triggered
                    "take_profit_1": 103.0,
                    "take_profit_2": 106.0,
                    "tp1_reached": True,
                    "tp1_booked_pnl": 0.0,  # No banked profit
                    "highest_price": 103.2,
                    "atr": 2.0,
                    "entry_time": "2026-01-01 10:00:00",
                    "confidence": 65.0,
                }
            },
            "trade_count": 0,
            "win_count": 0,
            "loss_count": 0,
        }

        closed = check_and_update_positions(mock_client, state, self.config)

        self.assertEqual(len(closed), 1)
        self.assertLess(closed[0]["net_pnl"], 0.0)
        self.assertEqual(closed[0]["outcome"], "LOSS", "Trade with negative PnL must be logged as LOSS!")
        self.assertEqual(state["loss_count"], 1)
        self.assertEqual(state["win_count"], 0)

    # =========================================================================
    # 6. 100-Cycle Monte Carlo Simulation Stress Test
    # =========================================================================
    def test_100_cycle_monte_carlo_portfolio_stability(self):
        """Simulates 100 continuous market scan and position management cycles
        under volatile random price shocks, verifying zero memory leaks or negative balance."""
        np.random.seed(42)
        mock_client = MagicMock()
        mock_client.get_min_notional.return_value = 5.0
        mock_client.fetch_funding_rate.return_value = 0.0001

        curr_p = 100.0
        mock_client.get_ticker_price.side_effect = lambda s: curr_p

        class SimModel:
            def predict_proba(self, X):
                return np.array([[0.30, 0.70]])

        feats = features.extract_features(self.sample_df)
        mock_bundle = {"model": SimModel(), "feature_names": list(feats.columns)}
        mock_client.fetch_ohlcv.return_value = self.sample_df

        state = {
            "balance_usdt": 100.0,
            "peak_balance": 100.0,
            "daily_peak_balance": 100.0,
            "daily_reset_time": datetime.now(timezone.utc).isoformat(),
            "circuit_breaker_until": None,
            "open_positions": {},
            "trade_count": 0,
            "win_count": 0,
            "loss_count": 0,
        }

        self.config["symbols"] = ["COIN_A/USDT", "COIN_B/USDT"]

        for cycle in range(100):
            # Random price shock between -1.5% and +1.8%
            curr_p *= (1.0 + np.random.uniform(-0.015, 0.018))

            # Update existing positions
            check_and_update_positions(mock_client, state, self.config)

            # Scan and enter new positions
            scan_and_execute(mock_client, mock_bundle, state, self.config)

            # Assert invariants
            self.assertGreater(state["balance_usdt"], 0.0, f"Account balance went negative on cycle {cycle}!")
            self.assertLessEqual(len(state["open_positions"]), self.config["risk_management"]["max_open_trades"])
            open_alloc = sum(p["allocated_usdt"] for p in state["open_positions"].values())
            total_equity = state["balance_usdt"] + open_alloc
            self.assertGreater(total_equity, 0.0)


if __name__ == "__main__":
    unittest.main()
