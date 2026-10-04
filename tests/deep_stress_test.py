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


class TestCrossAssetCorrelationFilter(unittest.TestCase):
    """Stress tests for Cross-Asset 30-Day Pearson Correlation Filter."""

    def test_correlation_filter_blocks_correlated_asset_and_picks_diversified(self):
        """When holding BTC, highly correlated ETH (>0.75) must be blocked, allowing low-corr SOL."""
        mock_client = MagicMock()
        mock_client.fetch_funding_rate.return_value = 0.0001
        dates = pd.date_range("2026-01-01", periods=100, freq="1h", tz="UTC")
        mock_df = pd.DataFrame({"open": 100, "high": 102, "low": 98, "close": 101, "volume": 1000}, index=dates)
        mock_client.fetch_ohlcv.return_value = mock_df

        feats = features.extract_features(mock_df)
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])
        mock_bundle = {"model": mock_model, "feature_names": list(feats.columns)}

        state = {
            "balance_usdt": 2000.0,
            "open_positions": {
                "BTC/USDT": {"entry_price": 80000.0, "allocated_usdt": 500.0, "units": 0.00625, "stop_loss": 78000.0}
            },
            "daily_peak_balance": 2000.0,
            "daily_reset_time": datetime.now(timezone.utc).isoformat(),
            "circuit_breaker_until": None
        }

        config = {
            "trading_mode": "paper",
            "symbols": ["ETH/USDT", "SOL/USDT"],
            "timeframe": "1h",
            "ai_model": {"confidence_threshold": 0.60, "tp1_atr_mult": 1.2, "tp2_atr_mult": 2.4, "sl_atr_mult": 1.4},
            "risk_management": {"capital_usdt": 2000.0, "risk_per_trade_pct": 2.0, "max_open_trades": 2, "fee_rate": 0.00075, "slippage_rate": 0.0005},
            "mtf_confluence": {"enabled": False},
            "correlation_filter": {"enabled": True, "max_correlation": 0.75},
            "paths": {"trade_ledger": "logs/test_ledger_corr.csv", "data_dir": "data"}
        }

        def mock_corr(s1, s2, data_dir="data"):
            # BTC vs ETH is high correlation 0.88, BTC vs SOL is low correlation 0.42
            pair = tuple(sorted([s1, s2]))
            if ("BTC/USDT", "ETH/USDT") == pair:
                return 0.88
            elif ("BTC/USDT", "SOL/USDT") == pair:
                return 0.42
            return 0.10

        with patch("bot.compute_asset_correlation", side_effect=mock_corr):
            rows, new_signals, _ = bot.scan_and_execute(mock_client, mock_bundle, state, config)

        eth_row = next(r for r in rows if r["symbol"] == "ETH/USDT")
        sol_row = next(r for r in rows if r["symbol"] == "SOL/USDT")

        self.assertEqual(eth_row["status"], "CORR BLOCKED", "ETH was not blocked despite 0.88 correlation with open BTC!")
        self.assertEqual(sol_row["status"], "BUY TRIGGER", "SOL was not executed despite low correlation with open BTC!")
        self.assertEqual(len(new_signals), 1)
        self.assertEqual(new_signals[0]["symbol"], "SOL/USDT")

        if os.path.exists("logs/test_ledger_corr.csv"):
            os.remove("logs/test_ledger_corr.csv")

    def test_compute_asset_correlation_calculation(self):
        """Verify rolling Pearson correlation calculation on generated CSV files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            dates = pd.date_range("2026-01-01", periods=100, freq="1h")
            # Returns series
            np.random.seed(42)
            rets = np.random.normal(0, 0.02, 100)
            p1 = 100.0 * np.cumprod(1.0 + rets)
            p2 = 50.0 * np.cumprod(1.0 + rets)       # Identical returns -> corr = 1.0
            p3 = 100.0 * np.cumprod(1.0 - rets)      # Exact opposite returns -> corr = -1.0

            df1 = pd.DataFrame({"close": p1}, index=dates)
            df2 = pd.DataFrame({"close": p2}, index=dates)
            df3 = pd.DataFrame({"close": p3}, index=dates)

            df1.to_csv(os.path.join(tmp_dir, "binance_AAA_USDT_1h.csv"))
            df2.to_csv(os.path.join(tmp_dir, "binance_BBB_USDT_1h.csv"))
            df3.to_csv(os.path.join(tmp_dir, "binance_CCC_USDT_1h.csv"))

            corr_pos = bot.compute_asset_correlation("AAA/USDT", "BBB/USDT", data_dir=tmp_dir)
            corr_neg = bot.compute_asset_correlation("AAA/USDT", "CCC/USDT", data_dir=tmp_dir)

            self.assertAlmostEqual(corr_pos, 1.0, places=2)
            self.assertAlmostEqual(corr_neg, -1.0, places=2)


class TestChandelierDynamicTrailingStop(unittest.TestCase):
    """Stress tests for Chandelier Volatility Dynamic Trailing Stop for runners."""

    def test_chandelier_trailing_stop_ratchets_and_exits(self):
        """Verify that stop loss ratchets upward on new highs and triggers exit on reversal."""
        state = {
            "balance_usdt": 5000.0,
            "open_positions": {
                "SOL/USDT": {
                    "entry_price": 100.0,
                    "allocated_usdt": 500.0,
                    "units": 5.0,
                    "atr": 3.0,
                    "stop_loss": 100.1,  # Breakeven locked after TP1
                    "take_profit_1": 103.6,
                    "take_profit_2": 107.2,
                    "tp1_reached": True,
                    "highest_price": 103.6
                }
            },
            "trade_count": 0,
            "win_count": 0,
            "loss_count": 0
        }
        config = {
            "trading_mode": "paper",
            "trailing_stop": {
                "enabled": True,
                "type": "chandelier",
                "atr_mult": 2.0  # Chandelier trail = peak - 2.0 * ATR
            },
            "ai_model": {"partial_tp_ratio": 0.50, "breakeven_lock_enabled": True},
            "risk_management": {"fee_rate": 0.00075, "slippage_rate": 0.0005},
            "paths": {"trade_ledger": "logs/test_ledger_trail.csv"}
        }

        mock_client = MagicMock()

        # Step 1: Price rallies to $120 (Peak). Chandelier stop = 120 - 2.0 * 3.0 = 114.0
        mock_client.get_ticker_price.return_value = 120.0
        closed = bot.check_and_update_positions(mock_client, state, config)
        self.assertEqual(len(closed), 0, "Trade closed prematurely during rally!")
        pos = state["open_positions"]["SOL/USDT"]
        self.assertEqual(pos["highest_price"], 120.0)
        self.assertAlmostEqual(pos["stop_loss"], 114.0, places=2, msg="Stop loss did not ratchet to 114.0!")

        # Step 2: Price dips to $116 (above SL 114.0). Stop loss must NOT decrease!
        mock_client.get_ticker_price.return_value = 116.0
        closed = bot.check_and_update_positions(mock_client, state, config)
        self.assertEqual(len(closed), 0, "Trade closed while price remained above ratcheted SL!")
        self.assertAlmostEqual(pos["stop_loss"], 114.0, places=2, msg="Stop loss decreased on price pullback!")

        # Step 3: Price drops to $113.5 (breaches ratcheted SL 114.0). Chandelier exit fires!
        mock_client.get_ticker_price.return_value = 113.5
        closed = bot.check_and_update_positions(mock_client, state, config)
        self.assertEqual(len(closed), 1, "Chandelier trailing exit did not trigger!")
        exit_trade = closed[0]
        self.assertEqual(exit_trade["outcome"], "WIN")
        self.assertIn("CHANDELIER TRAILING EXIT", exit_trade["reason"])
        self.assertGreater(exit_trade["net_pnl"], 0.0)
        self.assertNotIn("SOL/USDT", state["open_positions"])

        if os.path.exists("logs/test_ledger_trail.csv"):
            os.remove("logs/test_ledger_trail.csv")


class TestStackingEnsembleModel(unittest.TestCase):
    """Stress tests for Multi-Model Stacking Ensemble (LightGBM + HistGradientBoosting)."""

    def test_ensemble_train_and_predict(self):
        """Ensemble must train both trees and output calibrated probabilities summing to 1."""
        from features import StackingEnsembleModel
        from sklearn.ensemble import HistGradientBoostingClassifier
        import lightgbm as lgb

        hgb = HistGradientBoostingClassifier(max_iter=20, random_state=42)
        lgbm = lgb.LGBMClassifier(n_estimators=20, random_state=42, verbose=-1)
        ensemble = StackingEnsembleModel(hgb_model=hgb, lgb_model=lgbm)

        # Synthetic dataset
        np.random.seed(42)
        X = np.random.randn(200, 10)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        ensemble.fit(X, y)
        probs = ensemble.predict_proba(X)
        preds = ensemble.predict(X)
        acc = ensemble.score(X, y)

        self.assertEqual(probs.shape, (200, 2))
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(200), rtol=1e-5)
        self.assertTrue(all(p in [0, 1] for p in preds))
        self.assertGreater(acc, 0.70, "Stacking ensemble accuracy failed baseline on separable synthetic data!")


if __name__ == "__main__":
    unittest.main(verbosity=2)

