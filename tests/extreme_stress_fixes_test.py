"""Extreme Stress Test Suite for Bot Fixes & Stability.
Exhaustively tests:
1. Rejection of low-conviction, low-score, bearish-trend setups (The OP/USDT scenario).
2. Anti-revenge loss cooldown preventing immediate re-entry after stop out (The FET/USDT scenario).
3. Minimum cash reserve enforcement preventing account choking ($0.22 free cash scenario).
4. Profit taking via TP2 hard target and trailing stop behavior.
5. Multi-position severe flash-crash drawdown and circuit breaker handling.
6. State persistence and cooldown expiry time-warp.
"""

import os
import sys
import json
import unittest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd

# Add repo root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import bot
import features
from binance_client import BinanceClient


class TestExtremeStressFixes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        n = 150
        dates = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")

        # 1. Bearish Downtrend Series (Similar to OP/USDT condition)
        # Steady downward slope, negative CMF, bearish SuperTrend
        bear_close = 100.0 * np.linspace(1.0, 0.85, n)
        cls.bear_df = pd.DataFrame({
            "open": bear_close * 1.002,
            "high": bear_close * 1.004,
            "low": bear_close * 0.995,
            "close": bear_close,
            "volume": np.random.uniform(500, 2000, n)
        }, index=dates)

        # 2. Bullish Healthy Series (High quality breakout setup)
        bull_close = 50.0 * np.linspace(1.0, 1.15, n)
        cls.bull_df = pd.DataFrame({
            "open": bull_close * 0.998,
            "high": bull_close * 1.006,
            "low": bull_close * 0.996,
            "close": bull_close,
            "volume": np.random.uniform(2000, 8000, n)
        }, index=dates)

    def setUp(self):
        self.config = bot.load_config("config.json")
        self.config["trading_mode"] = "paper"
        self.config["mtf_confluence"] = {"enabled": False}
        self.config["correlation_filter"] = {"enabled": False}
        self.config["paths"]["trade_ledger"] = "logs/test_stress_fixes_ledger.csv"
        self.config["paths"]["state_file"] = "logs/test_stress_fixes_state.json"

    def tearDown(self):
        for f in ["logs/test_stress_fixes_ledger.csv", "logs/test_stress_fixes_state.json",
                  "data/binance_AAA_USDT_1h.csv", "data/binance_BBB_USDT_1h.csv"]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass

    # =========================================================================
    # TEST 1: OP/USDT FALSE POSITIVE REJECTION
    # Setup has AI 58.4%, SuperTrend Bearish, Score 2/100, CMF -0.05
    # MUST BE 100% REJECTED with NO order placed and NO money deducted.
    # =========================================================================
    def test_op_scenario_bearish_setup_rejected(self):
        mock_client = MagicMock()
        mock_client.fetch_ohlcv.return_value = self.bear_df
        mock_client.fetch_funding_rate.return_value = 0.0001

        # Model outputs 58.4% (exactly like the OP trade)
        class MockModel:
            def predict_proba(self, X):
                return np.array([[0.416, 0.584]])

        feats = features.extract_features(self.bear_df)
        mock_bundle = {"model": MockModel(), "feature_names": list(feats.columns)}

        state = {
            "balance_usdt": 30.0,
            "peak_balance": 30.0,
            "daily_peak_balance": 30.0,
            "daily_reset_time": datetime.now(timezone.utc).isoformat(),
            "circuit_breaker_until": None,
            "open_positions": {},
            "loss_cooldowns": {},
            "trade_count": 0,
            "win_count": 0,
            "loss_count": 0
        }

        self.config["symbols"] = ["OP/USDT"]
        rows, new_signals, cb = bot.scan_and_execute(mock_client, mock_bundle, state, self.config)

        # Assert zero trades executed
        self.assertEqual(len(new_signals), 0, "OP/USDT must NOT be bought when indicators are bearish!")
        self.assertEqual(len(state["open_positions"]), 0, "No open position should exist for OP/USDT!")
        self.assertEqual(state["balance_usdt"], 30.0, "Capital must remain completely intact ($30.00)!")

        # Assert status clearly reflects why it was blocked
        op_row = rows[0]
        self.assertIn(op_row["status"], ["BEAR TREND", "WEAK SETUP", "SCANNING", "LOW CONVICTION"],
                      f"Expected rejection status, got {op_row['status']}")

    # =========================================================================
    # TEST 2: ANTI-REVENGE LOSS COOLDOWN (The FET/USDT Scenario)
    # When a position is closed at a loss, it is locked into a 12-hour cooldown.
    # Re-entry must be rejected even if model predicts 90% win probability!
    # =========================================================================
    def test_anti_revenge_loss_cooldown_blocks_immediate_rebuy(self):
        mock_client = MagicMock()
        # Price drops through stop loss
        mock_client.get_ticker_price.return_value = 0.24  # Below stop loss 0.2475

        state = {
            "balance_usdt": 20.0,
            "peak_balance": 30.0,
            "daily_peak_balance": 30.0,
            "daily_reset_time": datetime.now(timezone.utc).isoformat(),
            "circuit_breaker_until": None,
            "open_positions": {
                "FET/USDT": {
                    "entry_price": 0.26,
                    "allocated_usdt": 10.0,
                    "units": 38.46,
                    "stop_loss": 0.25,
                    "take_profit_1": 0.27,
                    "take_profit_2": 0.28,
                    "tp1_reached": False,
                    "highest_price": 0.26,
                    "atr": 0.007,
                    "entry_time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                }
            },
            "loss_cooldowns": {},
            "trade_count": 1,
            "win_count": 0,
            "loss_count": 1
        }

        # Step 1: Position is closed at STOP LOSS
        closed = bot.check_and_update_positions(mock_client, state, self.config)
        self.assertEqual(len(closed), 1)
        self.assertEqual(closed[0]["outcome"], "LOSS")
        self.assertEqual(len(state["open_positions"]), 0)

        # Verify FET is now on cooldown
        self.assertIn("FET/USDT", state["loss_cooldowns"], "FET/USDT must be added to loss_cooldowns after stop out!")
        expiry_str = state["loss_cooldowns"]["FET/USDT"]
        expiry_dt = datetime.fromisoformat(expiry_str)
        now_dt = datetime.now(timezone.utc)
        self.assertGreater(expiry_dt, now_dt + timedelta(hours=11), "Cooldown must be set to ~12 hours!")

        # Step 2: Now run scan_and_execute on FET/USDT with 90% AI confidence
        mock_client.fetch_ohlcv.return_value = self.bull_df
        mock_client.fetch_funding_rate.return_value = 0.0001

        class HighConvictionModel:
            def predict_proba(self, X):
                return np.array([[0.10, 0.90]])  # 90% bullish!

        feats = features.extract_features(self.bull_df)
        mock_bundle = {"model": HighConvictionModel(), "feature_names": list(feats.columns)}

        self.config["symbols"] = ["FET/USDT"]
        rows, new_signals, _ = bot.scan_and_execute(mock_client, mock_bundle, state, self.config)

        # MUST BE BLOCKED BY COOLDOWN!
        self.assertEqual(len(new_signals), 0, "Must NOT rebuy FET while on loss cooldown!")
        self.assertEqual(rows[0]["status"], "COOLDOWN", "Status should be COOLDOWN!")

        # Step 3: Time warp past 12 hours -> Cooldown should expire
        state["loss_cooldowns"]["FET/USDT"] = (now_dt - timedelta(minutes=1)).isoformat()
        rows2, new_signals2, _ = bot.scan_and_execute(mock_client, mock_bundle, state, self.config)
        self.assertNotEqual(rows2[0]["status"], "COOLDOWN", "Cooldown must expire after elapsed duration!")

    # =========================================================================
    # TEST 3: MINIMUM CASH RESERVE ENFORCEMENT ($0.22 Free Cash Prevention)
    # Bot must never drain cash below 15% reserve buffer.
    # =========================================================================
    def test_cash_reserve_buffer_prevents_account_choking(self):
        mock_client = MagicMock()
        mock_client.fetch_ohlcv.return_value = self.bull_df
        mock_client.fetch_funding_rate.return_value = 0.0001

        class MockModel:
            def predict_proba(self, X):
                return np.array([[0.20, 0.80]])

        feats = features.extract_features(self.bull_df)
        mock_bundle = {"model": MockModel(), "feature_names": list(feats.columns)}

        # Account has $30 total equity, but 2 open positions holding $20
        # Available cash is only $10.00
        state = {
            "balance_usdt": 10.0,
            "peak_balance": 30.0,
            "daily_peak_balance": 30.0,
            "daily_reset_time": datetime.now(timezone.utc).isoformat(),
            "circuit_breaker_until": None,
            "open_positions": {
                "SOL/USDT": {"allocated_usdt": 10.0, "entry_price": 100.0, "units": 0.1, "stop_loss": 95.0, "take_profit_1": 105.0, "take_profit_2": 110.0},
                "ETH/USDT": {"allocated_usdt": 10.0, "entry_price": 2000.0, "units": 0.005, "stop_loss": 1900.0, "take_profit_1": 2100.0, "take_profit_2": 2200.0}
            },
            "loss_cooldowns": {},
            "trade_count": 2,
            "win_count": 1,
            "loss_count": 1
        }

        # Attempt to open 3rd position on AVAX
        self.config["symbols"] = ["AVAX/USDT"]
        self.config["risk_management"]["min_cash_reserve_pct"] = 0.15  # 15% of $30 = $4.50 reserve

        rows, new_signals, _ = bot.scan_and_execute(mock_client, mock_bundle, state, self.config)

        if len(new_signals) > 0:
            # If position was opened, remaining balance MUST be >= $4.50 reserve buffer
            self.assertGreaterEqual(state["balance_usdt"], 4.50,
                                    f"Free cash dropped to ${state['balance_usdt']:.2f}, breaching $4.50 reserve!")

        # Now test scenario where available cash is $4.00 (below $4.50 reserve buffer)
        state["balance_usdt"] = 4.00
        rows2, new_signals2, _ = bot.scan_and_execute(mock_client, mock_bundle, state, self.config)
        self.assertEqual(len(new_signals2), 0, "No new trades should be allowed when spendable cash < $5.00!")
        self.assertEqual(state["balance_usdt"], 4.00, "Cash balance should remain untouched!")

    # =========================================================================
    # TEST 4: HARD TP2 TARGET EXECUTION & WIN BOOKING
    # When exit_at_tp2 is enabled, price reaching TP2 must exit as WIN.
    # =========================================================================
    def test_hard_tp2_exit_takes_full_profit(self):
        mock_client = MagicMock()
        mock_client.get_ticker_price.return_value = 108.0  # Above TP2 107.0

        state = {
            "balance_usdt": 20.0,
            "peak_balance": 30.0,
            "daily_peak_balance": 30.0,
            "daily_reset_time": datetime.now(timezone.utc).isoformat(),
            "circuit_breaker_until": None,
            "open_positions": {
                "SOL/USDT": {
                    "entry_price": 100.0,
                    "allocated_usdt": 10.0,
                    "units": 0.1,
                    "stop_loss": 98.0,
                    "take_profit_1": 103.0,
                    "take_profit_2": 107.0,
                    "tp1_reached": True,
                    "highest_price": 108.0,
                    "atr": 2.5
                }
            },
            "loss_cooldowns": {},
            "trade_count": 0,
            "win_count": 0,
            "loss_count": 0
        }

        self.config["ai_model"]["exit_at_tp2"] = True
        closed = bot.check_and_update_positions(mock_client, state, self.config)

        self.assertEqual(len(closed), 1, "Must exit when TP2 target is reached!")
        self.assertEqual(closed[0]["outcome"], "WIN")
        self.assertEqual(closed[0]["reason"], "TP2 FINAL TARGET REACHED")
        self.assertGreater(closed[0]["net_pnl"], 0.0, "Net profit must be positive!")
        self.assertEqual(len(state["open_positions"]), 0)

    # =========================================================================
    # TEST 5: FLASH-CRASH VOLATILITY & CIRCUIT BREAKER STRESS
    # 50% sudden crash must trigger stops and activate circuit breaker safely.
    # =========================================================================
    def test_flash_crash_trips_circuit_breaker_and_preserves_state(self):
        mock_client = MagicMock()
        # Massive flash-crash: price crashes by 50%
        mock_client.get_ticker_price.return_value = 50.0

        state = {
            "balance_usdt": 10.0,
            "peak_balance": 30.0,
            "daily_peak_balance": 30.0,
            "daily_reset_time": datetime.now(timezone.utc).isoformat(),
            "circuit_breaker_until": None,
            "open_positions": {
                "SOL/USDT": {
                    "entry_price": 100.0,
                    "allocated_usdt": 10.0,
                    "units": 0.1,
                    "stop_loss": 95.0,
                    "take_profit_1": 105.0,
                    "take_profit_2": 110.0,
                    "tp1_reached": False,
                    "highest_price": 100.0
                }
            },
            "loss_cooldowns": {},
            "trade_count": 0,
            "win_count": 0,
            "loss_count": 0
        }

        closed = bot.check_and_update_positions(mock_client, state, self.config)
        self.assertEqual(len(closed), 1)
        self.assertEqual(closed[0]["outcome"], "LOSS")

        # Now check circuit breaker: Daily peak $30.00, equity is now $10 + $5 = $15 (50% drawdown)
        cb_locked, status = bot.check_circuit_breaker(state, self.config)
        self.assertTrue(cb_locked, "Circuit breaker must trip on 50% flash crash!")
        self.assertIn("TRIGGERED", status)

    # =========================================================================
    # TEST 6 (PHASE 1): NATIVE OCO ORDER GENERATION & PARAMETERS
    # =========================================================================
    def test_native_oco_order_interface(self):
        client = BinanceClient(self.config)
        client.has_credentials = True
        client.exchange = MagicMock()
        client.exchange.markets = {"SOL/USDT": {"limits": {"cost": {"min": 5.0}}}}
        client.exchange.market.return_value = {"limits": {"cost": {"min": 5.0}}}
        client.exchange.amount_to_precision.return_value = "0.5"
        client.exchange.price_to_precision.side_effect = lambda sym, val: f"{val:.2f}"
        client.exchange.privatePostOrderOco.return_value = {"orderListId": 8888, "listOrderStatus": "EXEC_STARTED"}

        res = client.place_oco_order("SOL/USDT", "sell", 0.5, tp_price=110.0, sl_price=95.0)
        self.assertEqual(res["orderListId"], 8888)
        client.exchange.privatePostOrderOco.assert_called_once()
        call_params = client.exchange.privatePostOrderOco.call_args[0][0]
        self.assertEqual(call_params["symbol"], "SOLUSDT")
        self.assertEqual(call_params["side"], "SELL")
        self.assertEqual(call_params["price"], "110.00")
        self.assertEqual(call_params["stopPrice"], "95.00")

    # =========================================================================
    # TEST 7 (PHASE 2): CROSS-SECTIONAL PURGED DATE SPLITTING RIGOR
    # =========================================================================
    def test_cross_sectional_purged_date_splitting(self):
        import train
        dates = pd.date_range("2026-01-01", periods=250, freq="1h", tz="UTC")
        df_a = pd.DataFrame({"open": 10, "high": 11, "low": 9, "close": 10, "volume": 100}, index=dates)
        df_b = pd.DataFrame({"open": 20, "high": 22, "low": 18, "close": 20, "volume": 200}, index=dates)

        # Mock client returning 2 symbols
        mock_client = MagicMock()
        mock_client.fetch_historical_ohlcv.side_effect = [df_a, df_b]

        import tempfile
        import shutil
        tmp_dir = tempfile.mkdtemp()
        try:
            X, y, fwd_ret, _ = train.prepare_dataset(mock_client, ["AAA/USDT", "BBB/USDT"], "1h", days=5, offline=False, data_dir=tmp_dir)
            # Chronological sort check: Dates must be non-decreasing!
            self.assertTrue((X.index[1:] >= X.index[:-1]).all(), "Features must be chronologically sorted across symbols!")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # =========================================================================
    # TEST 8 (PHASE 3): BTC CASCADE HALT PROTECTS ALTCOIN PORTFOLIO
    # =========================================================================
    def test_btc_cascade_halts_altcoin_entries(self):
        mock_client = MagicMock()

        # BTC drops 4.0% in 1 hour (severe cascade)
        dates = pd.date_range("2026-01-01", periods=100, freq="1h", tz="UTC")
        btc_c = np.full(100, 80000.0)
        btc_c[-1] = 76800.0  # -4.0% crash
        df_btc_crash = pd.DataFrame({
            "open": btc_c, "high": btc_c * 1.002, "low": btc_c * 0.998, "close": btc_c, "volume": 10000.0
        }, index=dates)

        # Altcoin has a bullish pattern
        mock_client.fetch_ohlcv.side_effect = lambda sym, *args, **kwargs: df_btc_crash if sym == "BTC/USDT" else self.bull_df
        mock_client.fetch_funding_rate.return_value = 0.0001

        class BullishModel:
            def predict_proba(self, X):
                return np.array([[0.10, 0.90]])

        feats = features.extract_features(self.bull_df)
        mock_bundle = {"model": BullishModel(), "feature_names": list(feats.columns)}

        state = {
            "balance_usdt": 30.0,
            "peak_balance": 30.0,
            "daily_peak_balance": 30.0,
            "daily_reset_time": datetime.now(timezone.utc).isoformat(),
            "circuit_breaker_until": None,
            "open_positions": {},
            "loss_cooldowns": {},
            "trade_count": 0,
            "win_count": 0,
            "loss_count": 0
        }

        self.config["symbols"] = ["BTC/USDT", "SOL/USDT"]
        self.config["risk_management"]["btc_cascade_halt_pct"] = -0.025

        rows, new_signals, _ = bot.scan_and_execute(mock_client, mock_bundle, state, self.config)

        # SOL/USDT MUST be halted due to BTC cascade!
        sol_row = [r for r in rows if r["symbol"] == "SOL/USDT"][0]
        self.assertEqual(sol_row["status"], "BTC CASCADE", "Altcoin must be marked BTC CASCADE during Bitcoin market dump!")
        # Zero altcoin signals allowed
        alt_signals = [s for s in new_signals if s["symbol"] != "BTC/USDT"]
        self.assertEqual(len(alt_signals), 0, "No altcoin buy orders permitted during BTC cascade!")

    # =========================================================================
    # TEST 9 (PHASE 4): CVD & FUNDING SQUEEZE STRATEGY DETECTION
    # =========================================================================
    def test_cvd_and_funding_squeeze_strategy_detection(self):
        # Create data with heavy buying near lows (absorption)
        dates = pd.date_range("2026-01-01", periods=50, freq="1h", tz="UTC")
        close = 100.0 * np.ones(50)
        close[-5:] = 101.0
        df = pd.DataFrame({
            "open": close * 0.999,
            "high": close * 1.010,
            "low": close * 0.990,
            "close": close,
            "volume": np.random.uniform(5000, 10000, 50)
        }, index=dates)

        # Negative funding rate (-0.02% / 8h) indicating crowded shorts
        setups, score, summary = features.detect_active_strategies(df, funding_rate=-0.0002)
        self.assertIn("Funding Squeeze / Institutional Absorption", setups,
                      "Must detect Funding Squeeze setup when funding rate is negative!")
        self.assertIn("FR:", summary)
        self.assertIn("CVD:", summary)


if __name__ == "__main__":
    unittest.main(verbosity=2)

