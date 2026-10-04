"""Exhaustive Deep Stress-Test Suite for Binance Quantitative Bot.
Tests edge cases, extreme market conditions, mathematical anomalies,
ranking correctness, risk controls, and microstructure precision.
"""

import os
import sys
import json
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import features
import bot
import backtest
from binance_client import BinanceClient, retry_on_network_error


class TestExtremeMathematicalEdgeCases(unittest.TestCase):
    """Stress tests mathematical indicators against degenerate market data."""

    def test_zero_volume_handling(self):
        """Indicators must not divide by zero when volume is completely zero."""
        dates = pd.date_range("2026-01-01", periods=100, freq="1h", tz="UTC")
        df_zero_vol = pd.DataFrame({
            "open": np.linspace(100, 110, 100),
            "high": np.linspace(101, 112, 100),
            "low": np.linspace(99, 108, 100),
            "close": np.linspace(100, 111, 100),
            "volume": 0.0
        }, index=dates)

        feats = features.extract_features(df_zero_vol)
        self.assertFalse(feats.isna().any().any(), "Zero volume produced NaNs!")
        self.assertFalse(np.isinf(feats.values).any(), "Zero volume produced Infs!")

    def test_flatline_zero_volatility(self):
        """Indicators must not crash or output NaN when prices are completely flat."""
        dates = pd.date_range("2026-01-01", periods=100, freq="1h", tz="UTC")
        df_flat = pd.DataFrame({
            "open": 100.0,
            "high": 100.0,
            "low": 100.0,
            "close": 100.0,
            "volume": 1000.0
        }, index=dates)

        feats = features.extract_features(df_flat)
        self.assertFalse(feats.isna().any().any(), "Flatline data produced NaNs!")
        self.assertFalse(np.isinf(feats.values).any(), "Flatline data produced Infs!")

    def test_massive_volatility_spikes(self):
        """Indicators must remain finite even during 10,000% Flash-Crash or Flash-Pump spikes."""
        dates = pd.date_range("2026-01-01", periods=100, freq="1h", tz="UTC")
        close = np.full(100, 100.0)
        close[50] = 50000.0  # Massive spike
        close[51] = 1.0      # Massive crash
        df_spike = pd.DataFrame({
            "open": 100.0,
            "high": np.maximum(close, 105.0),
            "low": np.minimum(close, 95.0),
            "close": close,
            "volume": 10000.0
        }, index=dates)

        feats = features.extract_features(df_spike)
        self.assertFalse(feats.isna().any().any(), "Spike produced NaNs!")
        self.assertFalse(np.isinf(feats.values).any(), "Spike produced Infs!")

    def test_stoch_rsi_bounds(self):
        """StochRSI K and D must strictly lie within [0.0, 1.0]."""
        dates = pd.date_range("2026-01-01", periods=120, freq="1h", tz="UTC")
        close = 100.0 + np.sin(np.linspace(0, 20, 120)) * 20.0
        df = pd.DataFrame({
            "open": close,
            "high": close + 2.0,
            "low": close - 2.0,
            "close": close,
            "volume": 1000.0
        }, index=dates)

        k, d, cross = features.compute_stoch_rsi(df["close"], 14, 14, 3, 3)
        self.assertTrue((k >= 0.0).all() and (k <= 1.0).all(), f"StochRSI K out of bounds: [{k.min()}, {k.max()}]")
        self.assertTrue((d >= 0.0).all() and (d <= 1.0).all(), f"StochRSI D out of bounds: [{d.min()}, {d.max()}]")

    def test_cmf_bounds(self):
        """Chaikin Money Flow must strictly lie within [-1.0, 1.0]."""
        dates = pd.date_range("2026-01-01", periods=100, freq="1h", tz="UTC")
        df = pd.DataFrame({
            "open": np.random.uniform(90, 110, 100),
            "high": np.random.uniform(110, 120, 100),
            "low": np.random.uniform(80, 90, 100),
            "close": np.random.uniform(85, 115, 100),
            "volume": np.random.uniform(100, 5000, 100)
        }, index=dates)

        cmf = features.compute_cmf(df, 20)
        self.assertTrue((cmf >= -1.0).all() and (cmf <= 1.0).all(), f"CMF out of bounds: [{cmf.min()}, {cmf.max()}]")

    def test_strategy_score_bounds(self):
        """Strategy Score must strictly lie within [0.0, 100.0]."""
        dates = pd.date_range("2026-01-01", periods=100, freq="1h", tz="UTC")
        df = pd.DataFrame({
            "open": np.linspace(100, 150, 100),
            "high": np.linspace(102, 152, 100),
            "low": np.linspace(98, 148, 100),
            "close": np.linspace(101, 151, 100),
            "volume": 2000.0
        }, index=dates)

        setups, score, summary = features.detect_active_strategies(df)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 100.0)
        self.assertIsInstance(summary, str)
        self.assertIn("RSI", summary)


class TestAlphaRankingAndExecutionEngine(unittest.TestCase):
    """Stress tests Alpha Ranking and best-first trade allocation."""

    def test_alpha_ranking_selects_highest_conviction_first(self):
        """Verify bot strictly selects the highest probability setup across all symbols."""
        mock_client = MagicMock()
        dates = pd.date_range("2026-01-01", periods=100, freq="1h", tz="UTC")
        mock_df = pd.DataFrame({"open": 100, "high": 102, "low": 98, "close": 101, "volume": 1000}, index=dates)
        mock_client.fetch_ohlcv.return_value = mock_df
        mock_client.fetch_funding_rate.return_value = 0.0001

        feats = features.extract_features(mock_df)

        # Mock model that returns distinct probabilities per symbol
        prob_map = {
            "BTC/USDT": 0.58,  # Below threshold
            "ETH/USDT": 0.62,  # Good
            "SOL/USDT": 0.61,  # Good
            "BNB/USDT": 0.69,  # BEST (Highest conviction)
            "XRP/USDT": 0.64   # Good
        }

        class MockModel:
            curr_sym = "BTC/USDT"
            def predict_proba(self, X):
                p = prob_map.get(MockModel.curr_sym, 0.50)
                return np.array([[1 - p, p]])

        # Intercept fetch_ohlcv to update curr_sym
        def mock_fetch(sym, timeframe="1h", limit=300):
            MockModel.curr_sym = sym
            return mock_df
        mock_client.fetch_ohlcv.side_effect = mock_fetch

        mock_bundle = {"model": MockModel(), "feature_names": list(feats.columns)}

        # State with exactly 1 available slot
        state = {
            "balance_usdt": 1000.0,
            "open_positions": {},
            "daily_peak_balance": 1000.0,
            "daily_reset_time": datetime.now(timezone.utc).isoformat(),
            "circuit_breaker_until": None,
            "trade_count": 0,
            "win_count": 0,
            "loss_count": 0
        }
        config = {
            "trading_mode": "paper",
            "symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"],
            "timeframe": "1h",
            "ai_model": {"confidence_threshold": 0.60, "tp1_atr_mult": 1.2, "tp2_atr_mult": 2.4, "sl_atr_mult": 1.4},
            "risk_management": {
                "capital_usdt": 1000.0,
                "risk_per_trade_pct": 2.0,
                "max_open_trades": 1,  # Only 1 trade allowed!
                "dynamic_alpha_sizing": True,
                "fee_rate": 0.00075,
                "slippage_rate": 0.0005
            },
            "mtf_confluence": {"enabled": False},
            "paths": {"trade_ledger": "logs/test_ledger_rank.csv"}
        }

        rows, new_signals, _ = bot.scan_and_execute(mock_client, mock_bundle, state, config)

        # Must execute exactly 1 trade
        self.assertEqual(len(new_signals), 1)
        # And that single trade MUST BE BNB/USDT (69% conviction), not ETH or SOL!
        self.assertEqual(new_signals[0]["symbol"], "BNB/USDT", f"Bot selected {new_signals[0]['symbol']} instead of best (BNB/USDT)!")
        self.assertIn("BNB/USDT", state["open_positions"])

        if os.path.exists("logs/test_ledger_rank.csv"):
            os.remove("logs/test_ledger_rank.csv")

    def test_circuit_breaker_trips_on_daily_drawdown(self):
        """Verify circuit breaker trips and blocks buys when balance drops 8% below daily peak."""
        state = {
            "balance_usdt": 910.0,             # Down 9.0% from 1000.0
            "daily_peak_balance": 1000.0,
            "daily_reset_time": datetime.now(timezone.utc).isoformat(),
            "circuit_breaker_until": None,
            "open_positions": {}
        }
        config = {
            "risk_management": {
                "capital_usdt": 1000.0,
                "daily_max_drawdown_pct": 8.0,
                "circuit_breaker_hours": 24,
                "max_open_trades": 3,
                "fee_rate": 0.00075,
                "slippage_rate": 0.0005
            },
            "symbols": ["BTC/USDT"],
            "timeframe": "1h",
            "ai_model": {"confidence_threshold": 0.50},
            "mtf_confluence": {"enabled": False},
            "paths": {"trade_ledger": "logs/test_ledger_cb.csv"}
        }

        # Check circuit breaker function directly
        tripped, status = bot.check_circuit_breaker(state, config)
        self.assertTrue(tripped, "Circuit breaker did not trip on 9% daily drawdown!")
        self.assertTrue("LOCKED" in status or "TRIGGERED" in status)

        # Run scan_and_execute to ensure no orders are placed
        mock_client = MagicMock()
        dates = pd.date_range("2026-01-01", periods=100, freq="1h", tz="UTC")
        mock_df = pd.DataFrame({"open": 100, "high": 102, "low": 98, "close": 101, "volume": 1000}, index=dates)
        mock_client.fetch_ohlcv.return_value = mock_df
        mock_client.fetch_funding_rate.return_value = 0.0001
        feats = features.extract_features(mock_df)
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.1, 0.9]])
        mock_bundle = {"model": mock_model, "feature_names": list(feats.columns)}

        rows, new_signals, _ = bot.scan_and_execute(mock_client, mock_bundle, state, config)
        self.assertEqual(len(new_signals), 0, "Bot placed orders while circuit breaker was locked!")
        self.assertEqual(rows[0]["status"], "CIRCUIT PAUSE")

        if os.path.exists("logs/test_ledger_cb.csv"):
            os.remove("logs/test_ledger_cb.csv")

    def test_overheated_funding_rate_blocks_long_entry(self):
        """Verify overheated positive funding (> +0.03%) blocks buying over-leveraged longs."""
        mock_client = MagicMock()
        dates = pd.date_range("2026-01-01", periods=100, freq="1h", tz="UTC")
        mock_df = pd.DataFrame({"open": 100, "high": 102, "low": 98, "close": 101, "volume": 1000}, index=dates)
        mock_client.fetch_ohlcv.return_value = mock_df
        # Extremely high funding rate (+0.05%)
        mock_client.fetch_funding_rate.return_value = 0.0005

        feats = features.extract_features(mock_df)
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.1, 0.9]])  # 90% AI Conviction!
        mock_bundle = {"model": mock_model, "feature_names": list(feats.columns)}

        state = {
            "balance_usdt": 1000.0,
            "open_positions": {},
            "daily_peak_balance": 1000.0,
            "daily_reset_time": datetime.now(timezone.utc).isoformat(),
            "circuit_breaker_until": None
        }
        config = {
            "trading_mode": "paper",
            "symbols": ["BTC/USDT"],
            "timeframe": "1h",
            "ai_model": {"confidence_threshold": 0.60, "tp1_atr_mult": 1.2, "tp2_atr_mult": 2.4, "sl_atr_mult": 1.4},
            "risk_management": {"capital_usdt": 1000.0, "risk_per_trade_pct": 2.0, "max_open_trades": 1, "fee_rate": 0.00075, "slippage_rate": 0.0005},
            "mtf_confluence": {"enabled": False},
            "paths": {"trade_ledger": "logs/test_ledger_fund.csv"}
        }

        rows, new_signals, _ = bot.scan_and_execute(mock_client, mock_bundle, state, config)
        self.assertEqual(len(new_signals), 0, "Bot bought into overheated funding rate!")
        self.assertEqual(rows[0]["status"], "FUNDING HOT")

        if os.path.exists("logs/test_ledger_fund.csv"):
            os.remove("logs/test_ledger_fund.csv")


class TestMicrostructureSanitization(unittest.TestCase):
    """Stress tests lot size, tick size, and minimum notional rounding."""

    def test_amount_rounding_down_to_step_size(self):
        """Units must always round down to prevent exceeding account balance."""
        from binance_client import BinanceClient
        client = BinanceClient({"trading_mode": "paper", "binance": {}, "risk_management": {}})
        client.exchange.markets = {
            "BTC/USDT": {
                "symbol": "BTC/USDT",
                "precision": {"amount": 5, "price": 2},
                "limits": {"cost": {"min": 5.0}}
            }
        }
        client.exchange.amount_to_precision = MagicMock(return_value="0.01234")
        client.exchange.price_to_precision = MagicMock(return_value="80000.00")

        clean_amt, clean_price = client.sanitize_order_amount("BTC/USDT", 0.0123456, price=80000.0)
        self.assertAlmostEqual(clean_amt, 0.01234, places=5)
        self.assertLessEqual(clean_amt, 0.0123456)
        self.assertEqual(clean_price, 80000.00)


if __name__ == "__main__":
    unittest.main(verbosity=2)
