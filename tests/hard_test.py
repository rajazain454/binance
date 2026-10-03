"""Comprehensive Hard Stress-Test Suite for Binance Quantitative AI Bot.
Exhaustively tests all modules, mathematical models, risk rules, and edge cases.
"""

import os
import sys
import json
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

# Ensure root dir is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import features
from binance_client import BinanceClient
import bot
import backtest
import train


class TestFeaturesEngine(unittest.TestCase):
    """Stress tests for features.py indicators and labeling."""

    def setUp(self):
        # Create standard synthetic OHLCV data
        dates = pd.date_range("2026-01-01", periods=300, freq="1h", tz="UTC")
        np.random.seed(42)
        close = 100.0 + np.cumsum(np.random.normal(0, 1, 300))
        high = close + np.random.uniform(0.5, 2.0, 300)
        low = close - np.random.uniform(0.5, 2.0, 300)
        open_p = low + np.random.uniform(0, 1, 300) * (high - low)
        volume = np.random.uniform(100, 10000, 300)

        self.df_standard = pd.DataFrame({
            "open": open_p,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume
        }, index=dates)

    def test_extract_features_shape_and_cleanliness(self):
        """Verify all 30 indicators exist and have zero NaNs or Infs."""
        feats = features.extract_features(self.df_standard)
        self.assertFalse(feats.empty)
        self.assertFalse(feats.isna().any().any(), "Features contain NaNs!")
        self.assertFalse(np.isinf(feats.values).any(), "Features contain Infs!")
        self.assertGreaterEqual(len(feats.columns), 25, "Expected at least 25 features")

    def test_features_non_datetime_index(self):
        """Verify features don't crash when index is a RangeIndex (e.g., non-datetime)."""
        df_range = self.df_standard.reset_index(drop=True)
        # Should either compute safely or handle without unhandled TypeError
        try:
            feats = features.extract_features(df_range)
            self.assertFalse(feats.empty)
        except TypeError as e:
            self.fail(f"extract_features crashed on RangeIndex: {e}")

    def test_features_flat_market_zero_volatility(self):
        """Stress test with completely flat prices (divide by zero risk)."""
        dates = pd.date_range("2026-01-01", periods=250, freq="1h", tz="UTC")
        df_flat = pd.DataFrame({
            "open": 100.0,
            "high": 100.0,
            "low": 100.0,
            "close": 100.0,
            "volume": 500.0
        }, index=dates)
        feats = features.extract_features(df_flat)
        self.assertFalse(np.isinf(feats.values).any(), "Flat market produced Inf!")

    def test_triple_barrier_labeling(self):
        """Verify triple-barrier outcomes are strictly binary [0, 1]."""
        labels, fwd_ret = features.label_triple_barrier(self.df_standard, horizon_bars=12, tp_atr_mult=2.0, sl_atr_mult=1.4)
        self.assertEqual(len(labels), len(self.df_standard))
        self.assertTrue(set(labels.unique()).issubset({0, 1}))
        self.assertFalse(fwd_ret.isna().any())

    def test_macro_confluence_evaluation(self):
        """Verify 4h macro confluence evaluator handles both small and large dataframes."""
        is_bull, details = features.evaluate_macro_confluence(self.df_standard)
        self.assertIsInstance(is_bull, bool)
        self.assertIn("status", details)
        self.assertIn("is_bullish", details)

        # Tiny dataframe (< 20 bars) should safely return True without error
        tiny_df = self.df_standard.iloc[:10]
        is_bull_tiny, details_tiny = features.evaluate_macro_confluence(tiny_df)
        self.assertTrue(is_bull_tiny)
        self.assertEqual(details_tiny["status"], "INSUFFICIENT_DATA")


class TestRiskAndCircuitBreaker(unittest.TestCase):
    """Stress tests for Area 1: Risk Parity, Circuit Breaker, and TP1/TP2 scaling."""

    def test_circuit_breaker_trigger_and_lock(self):
        """Verify that a 4.5% drawdown locks the circuit breaker for 24 hours."""
        state = {
            "balance_usdt": 9500.0,
            "daily_peak_balance": 10000.0,
            "daily_reset_time": datetime.now(timezone.utc).isoformat(),
            "circuit_breaker_until": None
        }
        config = {
            "risk_management": {
                "daily_max_drawdown_pct": 4.0,
                "circuit_breaker_hours": 24
            }
        }
        # 9500 vs 10000 is 5% DD -> should trigger
        locked, msg = bot.check_circuit_breaker(state, config)
        self.assertTrue(locked)
        self.assertIn("TRIGGERED", msg)
        self.assertIsNotNone(state["circuit_breaker_until"])

        # Second check while locked should remain locked
        locked2, msg2 = bot.check_circuit_breaker(state, config)
        self.assertTrue(locked2)
        self.assertIn("LOCKED", msg2)

    def test_circuit_breaker_expiration(self):
        """Verify circuit breaker unlocks automatically after lock period expires."""
        past_time = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
        state = {
            "balance_usdt": 9500.0,
            "daily_peak_balance": 10000.0,
            "daily_reset_time": past_time,
            "circuit_breaker_until": past_time  # Expired
        }
        config = {
            "risk_management": {
                "daily_max_drawdown_pct": 4.0,
                "circuit_breaker_hours": 24
            }
        }
        locked, msg = bot.check_circuit_breaker(state, config)
        self.assertFalse(locked)
        self.assertIsNone(state["circuit_breaker_until"])
        self.assertEqual(state["daily_peak_balance"], 9500.0)

    def test_tp1_partial_exit_and_breakeven_ratchet(self):
        """Verify TP1 sells 50% of position and ratchets Stop Loss to Breakeven."""
        state = {
            "balance_usdt": 5000.0,
            "open_positions": {
                "BTC/USDT": {
                    "entry_price": 80000.0,
                    "allocated_usdt": 2000.0,
                    "units": 0.025,
                    "stop_loss": 79000.0,
                    "take_profit_1": 81000.0,
                    "take_profit_2": 82000.0,
                    "tp1_reached": False
                }
            },
            "trade_count": 0,
            "win_count": 0,
            "loss_count": 0
        }
        config = {
            "ai_model": {
                "partial_tp_ratio": 0.50,
                "breakeven_lock_enabled": True
            },
            "risk_management": {
                "fee_rate": 0.00075,
                "slippage_rate": 0.0005
            },
            "paths": {
                "trade_ledger": "logs/test_ledger.csv"
            }
        }

        mock_client = MagicMock()
        # Price hits TP1 (81,000)
        mock_client.get_ticker_price.return_value = 81200.0

        closed = bot.check_and_update_positions(mock_client, state, config)
        self.assertEqual(len(closed), 1)
        self.assertEqual(closed[0]["action"], "PARTIAL TP1 (50% SOLD)")
        self.assertTrue(closed[0]["net_pnl"] > 0)

        # Verify position is still open, but with 50% units, tp1_reached=True, and Stop Loss ratcheted up
        pos = state["open_positions"]["BTC/USDT"]
        self.assertAlmostEqual(pos["units"], 0.0125, places=5)
        self.assertAlmostEqual(pos["allocated_usdt"], 1000.0, places=2)
        self.assertTrue(pos["tp1_reached"])
        self.assertGreater(pos["stop_loss"], 80000.0, "Stop Loss was not ratcheted to Breakeven!")

        # Clean up test ledger
        if os.path.exists("logs/test_ledger.csv"):
            os.remove("logs/test_ledger.csv")

    def test_runner_final_exit_at_tp2(self):
        """Verify runner remaining 50% exits at TP2 and marks trade as WIN."""
        state = {
            "balance_usdt": 5000.0,
            "open_positions": {
                "BTC/USDT": {
                    "entry_price": 80000.0,
                    "allocated_usdt": 1000.0,
                    "units": 0.0125,
                    "stop_loss": 80080.0,
                    "take_profit_1": 81000.0,
                    "take_profit_2": 82000.0,
                    "tp1_reached": True
                }
            },
            "trade_count": 0,
            "win_count": 0,
            "loss_count": 0
        }
        config = {
            "ai_model": {"partial_tp_ratio": 0.50, "breakeven_lock_enabled": True},
            "risk_management": {"fee_rate": 0.00075, "slippage_rate": 0.0005},
            "paths": {"trade_ledger": "logs/test_ledger.csv"}
        }

        mock_client = MagicMock()
        mock_client.get_ticker_price.return_value = 82100.0  # Above TP2

        closed = bot.check_and_update_positions(mock_client, state, config)
        self.assertEqual(len(closed), 1)
        self.assertEqual(closed[0]["outcome"], "WIN")
        self.assertEqual(state["trade_count"], 1)
        self.assertEqual(state["win_count"], 1)
        self.assertNotIn("BTC/USDT", state["open_positions"], "Position was not closed!")

        if os.path.exists("logs/test_ledger.csv"):
            os.remove("logs/test_ledger.csv")


class TestBinanceMicrostructure(unittest.TestCase):
    """Stress tests for Area 2: Binance lot size, precision, and minNotional sanitization."""

    def test_sanitize_order_amount_rounding(self):
        """Test precision sanitization against real Binance market structure."""
        client = BinanceClient({"trading_mode": "paper", "binance": {"api_key": "", "api_secret": ""}, "risk_management": {"capital_usdt": 1000}})
        # Mock exchange market metadata
        client.exchange.markets = {
            "BTC/USDT": {
                "precision": {"amount": 5, "price": 2},
                "limits": {"cost": {"min": 5.0}}
            }
        }
        client.exchange.amount_to_precision = MagicMock(return_value="0.01234")
        client.exchange.price_to_precision = MagicMock(return_value="85432.10")

        clean_amt, clean_price = client.sanitize_order_amount("BTC/USDT", 0.012345678, 85432.1098)
        self.assertEqual(clean_amt, 0.01234)
        self.assertEqual(clean_price, 85432.10)

    def test_sanitize_order_rejects_sub_notional(self):
        """Verify that orders below Binance $5.00 minNotional raise a clear ValueError."""
        client = BinanceClient({"trading_mode": "paper", "binance": {"api_key": "", "api_secret": ""}, "risk_management": {"capital_usdt": 1000}})
        client.exchange.markets = {
            "BTC/USDT": {
                "precision": {"amount": 5, "price": 2},
                "limits": {"cost": {"min": 5.0}}
            }
        }
        client.exchange.amount_to_precision = MagicMock(return_value="0.00001")
        client.exchange.price_to_precision = MagicMock(return_value="100.00")

        # 0.00001 * 100.0 = $0.001 < $5.0 minNotional -> should raise ValueError
        with self.assertRaises(ValueError) as ctx:
            client.sanitize_order_amount("BTC/USDT", 0.00001, 100.0)
        self.assertIn("below Binance minimum notional", str(ctx.exception))


class TestDiscordAndDigest(unittest.TestCase):
    """Stress tests for Area 4: Discord embed generator and Daily Digest."""

    def test_send_discord_alert_unconfigured(self):
        """Verify send_discord_alert fails gracefully when webhook URL is blank."""
        ok, res = bot.send_discord_alert("", "Test", [])
        self.assertFalse(ok)
        self.assertIn("Invalid or unconfigured", res)

    def test_daily_digest_generation(self):
        """Verify daily digest generates full statistics without crashing."""
        state = {
            "balance_usdt": 10250.0,
            "trade_count": 10,
            "win_count": 8,
            "open_positions": {
                "SOL/USDT": {"allocated_usdt": 1500.0, "entry_price": 120.0, "stop_loss": 118.0}
            }
        }
        config = {
            "risk_management": {"capital_usdt": 10000.0, "max_open_trades": 3},
            "trading_mode": "paper",
            "timeframe": "1h",
            "discord": {"enabled": False, "webhook_url": ""}
        }
        # When discord is disabled, it should report disabled gracefully
        ok, res = bot.send_daily_digest(state, config)
        self.assertFalse(ok)
        self.assertEqual(res, "Discord alerts disabled.")


class TestBacktestEngine(unittest.TestCase):
    """Stress tests for Backtesting engine."""

    def test_backtest_with_synthetic_model(self):
        """Verify backtest runs without crashing on synthetic data and produces full metrics."""
        dates = pd.date_range("2026-01-01", periods=200, freq="1h", tz="UTC")
        np.random.seed(42)
        close = 100.0 + np.cumsum(np.random.normal(0.1, 1, 200))
        df = pd.DataFrame({
            "open": close - 0.2,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1000.0
        }, index=dates)

        # Mock model bundle
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.column_stack([np.zeros(200), np.ones(200) * 0.75])
        feats = features.extract_features(df)
        mock_bundle = {
            "model": mock_model,
            "feature_names": list(feats.columns)
        }
        config = {
            "ai_model": {
                "confidence_threshold": 0.60,
                "tp1_atr_mult": 1.2,
                "tp2_atr_mult": 2.4,
                "sl_atr_mult": 1.4,
                "partial_tp_ratio": 0.50,
                "breakeven_lock_enabled": True
            },
            "risk_management": {
                "capital_usdt": 10000.0,
                "risk_per_trade_pct": 2.0,
                "fee_rate": 0.00075,
                "slippage_rate": 0.0005
            },
            "mtf_confluence": {"enabled": False}
        }

        report = backtest.run_backtest(df, mock_bundle, config, symbol="TEST/USDT")
        self.assertNotIn("error", report)
        self.assertIn("total_return_pct", report)
        self.assertIn("win_rate", report)
        self.assertIn("profit_factor", report)

    def test_corrupted_state_recovery(self):
        """Verify load_state gracefully recovers from corrupted or incomplete JSON state files."""
        temp_state_file = "logs/test_corrupt_state.json"
        os.makedirs("logs", exist_ok=True)
        # Write corrupted garbage JSON
        with open(temp_state_file, "w") as f:
            f.write("{ invalid json content ::: 12345")

        recovered_state = bot.load_state(temp_state_file, initial_capital=5000.0)
        self.assertEqual(recovered_state["balance_usdt"], 5000.0)
        self.assertEqual(recovered_state["open_positions"], {})
        self.assertEqual(recovered_state["trade_count"], 0)

        # Test empty file
        with open(temp_state_file, "w") as f:
            f.write("")
        recovered_state2 = bot.load_state(temp_state_file, initial_capital=8000.0)
        self.assertEqual(recovered_state2["balance_usdt"], 8000.0)

        if os.path.exists(temp_state_file):
            os.remove(temp_state_file)

    def test_live_execution_error_isolation(self):
        """Verify that live Binance order rejection does NOT corrupt bot state or deduct balance."""
        state = {
            "balance_usdt": 1000.0,
            "open_positions": {},
            "daily_peak_balance": 1000.0,
            "daily_reset_time": datetime.now(timezone.utc).isoformat(),
            "circuit_breaker_until": None
        }
        config = {
            "trading_mode": "live",
            "symbols": ["BTC/USDT"],
            "timeframe": "1h",
            "ai_model": {"confidence_threshold": 0.50, "tp1_atr_mult": 1.2, "tp2_atr_mult": 2.4, "sl_atr_mult": 1.4},
            "risk_management": {"capital_usdt": 1000.0, "risk_per_trade_pct": 2.0, "max_open_trades": 3, "fee_rate": 0.00075, "slippage_rate": 0.0005},
            "mtf_confluence": {"enabled": False},
            "paths": {"trade_ledger": "logs/test_ledger.csv"}
        }

        # Mock client that raises exception on place_spot_order
        mock_client = MagicMock()
        dates = pd.date_range("2026-01-01", periods=100, freq="1h", tz="UTC")
        mock_df = pd.DataFrame({"open": 80000, "high": 80100, "low": 79900, "close": 80050, "volume": 100}, index=dates)
        mock_client.fetch_ohlcv.return_value = mock_df
        mock_client.fetch_funding_rate.return_value = 0.0001
        mock_client.place_spot_order.side_effect = Exception("Binance API Error: Insufficient balance")

        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.1, 0.9]])
        feats = features.extract_features(mock_df)
        mock_bundle = {"model": mock_model, "feature_names": list(feats.columns)}

        # Run scan_and_execute
        rows, signals, cb = bot.scan_and_execute(mock_client, mock_bundle, state, config)

        # Balance must NOT be deducted, and open_positions must be empty because live order failed!
        self.assertEqual(state["balance_usdt"], 1000.0)
        self.assertEqual(len(state["open_positions"]), 0)
        self.assertEqual(len(signals), 0)

    def test_exponential_network_retry(self):
        """Verify retry_on_network_error retries transient CCXT network errors up to 3 times."""
        import ccxt
        from binance_client import retry_on_network_error

        calls = [0]

        @retry_on_network_error(max_retries=3, initial_delay=0.01)
        def unstable_call():
            calls[0] += 1
            if calls[0] < 3:
                raise ccxt.NetworkError("Temporary Binance timeout")
            return "SUCCESS"

        result = unstable_call()
        self.assertEqual(result, "SUCCESS")
        self.assertEqual(calls[0], 3, "Did not retry exactly 3 times before succeeding!")

    def test_max_holding_time_decay_exit(self):
        """Verify position held longer than max_holding_hours closes automatically as time-decay."""
        past_time = (datetime.now(timezone.utc) - timedelta(hours=50)).strftime("%Y-%m-%d %H:%M:%S")
        state = {
            "balance_usdt": 5000.0,
            "open_positions": {
                "BTC/USDT": {
                    "entry_price": 80000.0,
                    "allocated_usdt": 2000.0,
                    "units": 0.025,
                    "stop_loss": 78000.0,
                    "take_profit_1": 82000.0,
                    "take_profit_2": 85000.0,
                    "tp1_reached": False,
                    "entry_time": past_time
                }
            },
            "trade_count": 0,
            "win_count": 0,
            "loss_count": 0
        }
        config = {
            "ai_model": {"partial_tp_ratio": 0.50, "breakeven_lock_enabled": True},
            "risk_management": {
                "fee_rate": 0.00075,
                "slippage_rate": 0.0005,
                "max_holding_hours": 48
            },
            "paths": {"trade_ledger": "logs/test_ledger.csv"}
        }

        mock_client = MagicMock()
        # Price is stagnant between stop loss and TP1
        mock_client.get_ticker_price.return_value = 80100.0

        closed = bot.check_and_update_positions(mock_client, state, config)
        self.assertEqual(len(closed), 1)
        self.assertIn("MAX HOLDING TIME EXPIRED", closed[0]["reason"])
        self.assertNotIn("BTC/USDT", state["open_positions"])

        if os.path.exists("logs/test_ledger.csv"):
            os.remove("logs/test_ledger.csv")

    def test_mark_to_market_unrealized_pnl(self):
        """Verify live unrealized PnL is accurately updated on active positions."""
        state = {
            "balance_usdt": 5000.0,
            "open_positions": {
                "SOL/USDT": {
                    "entry_price": 100.0,
                    "allocated_usdt": 1000.0,
                    "units": 10.0,
                    "stop_loss": 90.0,
                    "take_profit_1": 110.0,
                    "take_profit_2": 120.0,
                    "tp1_reached": False,
                    "entry_time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                }
            },
            "trade_count": 0,
            "win_count": 0,
            "loss_count": 0
        }
        config = {
            "ai_model": {"partial_tp_ratio": 0.50, "breakeven_lock_enabled": True},
            "risk_management": {"fee_rate": 0.00075, "slippage_rate": 0.0005, "max_holding_hours": 48},
            "paths": {"trade_ledger": "logs/test_ledger.csv"}
        }

        mock_client = MagicMock()
        mock_client.get_ticker_price.return_value = 105.0  # +5% gain

        closed = bot.check_and_update_positions(mock_client, state, config)
        self.assertEqual(len(closed), 0)  # Still holding

        pos = state["open_positions"]["SOL/USDT"]
        self.assertEqual(pos["curr_price"], 105.0)
        self.assertEqual(pos["unrealized_pnl"], 50.0)  # 10 units * $5 = $50
        self.assertEqual(pos["unrealized_pnl_pct"], 5.0)

    def test_dynamic_alpha_sizing(self):
        """Verify that high AI conviction (75%) scales position size higher than base conviction (60%)."""
        mock_client = MagicMock()
        dates = pd.date_range("2026-01-01", periods=100, freq="1h", tz="UTC")
        mock_df = pd.DataFrame({"open": 100, "high": 102, "low": 98, "close": 101, "volume": 1000}, index=dates)
        mock_client.fetch_ohlcv.return_value = mock_df
        mock_client.fetch_funding_rate.return_value = 0.0001

        feats = features.extract_features(mock_df)

        # Case 1: Standard confidence (60%)
        mock_model_60 = MagicMock()
        mock_model_60.predict_proba.return_value = np.array([[0.40, 0.60]])
        bundle_60 = {"model": mock_model_60, "feature_names": list(feats.columns)}

        state_60 = {"balance_usdt": 10000.0, "open_positions": {}, "daily_peak_balance": 10000.0, "daily_reset_time": datetime.now(timezone.utc).isoformat(), "circuit_breaker_until": None}
        config_60 = {
            "trading_mode": "paper",
            "symbols": ["SOL/USDT"],
            "timeframe": "1h",
            "ai_model": {"confidence_threshold": 0.60, "tp1_atr_mult": 1.2, "tp2_atr_mult": 2.4, "sl_atr_mult": 1.4},
            "risk_management": {"capital_usdt": 10000.0, "risk_per_trade_pct": 2.0, "max_open_trades": 3, "dynamic_alpha_sizing": True, "fee_rate": 0.00075, "slippage_rate": 0.0005},
            "mtf_confluence": {"enabled": False},
            "paths": {"trade_ledger": "logs/test_ledger.csv"}
        }
        _, sig_60, _ = bot.scan_and_execute(mock_client, bundle_60, state_60, config_60)

        # Case 2: High conviction confidence (75%)
        mock_model_75 = MagicMock()
        mock_model_75.predict_proba.return_value = np.array([[0.25, 0.75]])
        bundle_75 = {"model": mock_model_75, "feature_names": list(feats.columns)}

        state_75 = {"balance_usdt": 10000.0, "open_positions": {}, "daily_peak_balance": 10000.0, "daily_reset_time": datetime.now(timezone.utc).isoformat(), "circuit_breaker_until": None}
        config_75 = dict(config_60)
        _, sig_75, _ = bot.scan_and_execute(mock_client, bundle_75, state_75, config_75)

        self.assertEqual(len(sig_60), 1)
        self.assertEqual(len(sig_75), 1)
        # 75% setup must allocate larger dollar size than 60% setup!
        self.assertGreater(sig_75[0]["size_usdt"], sig_60[0]["size_usdt"])

        if os.path.exists("logs/test_ledger.csv"):
            os.remove("logs/test_ledger.csv")


if __name__ == "__main__":
    unittest.main(verbosity=2)
