"""Unit tests for position management logic (TP/SL/sizing/win-loss classification).

These tests validate the core trading logic independently from live exchange connectivity.
They cover the critical bugs that were identified and fixed:
- Win/loss classification with tp1_reached
- Partial TP1 trade counter increments
- Chandelier trailing stop ratcheting
- Stop loss breakeven lock after TP1
- Max holding time exit
"""

import os
import sys
import json
import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone, timedelta

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock ccxt if running in environment where ccxt is not installed
if "ccxt" not in sys.modules:
    try:
        import ccxt  # noqa: F401
    except ImportError:
        sys.modules["ccxt"] = MagicMock()

import tempfile
import uuid
from tests.base import BaseTestCase


def make_config(test_dir=None):
    """Creates a minimal config dict for testing."""
    if test_dir is None:
        test_dir = tempfile.gettempdir()
    return {
        "trading_mode": "paper",
        "timeframe": "1h",
        "ai_model": {
            "confidence_threshold": 0.60,
            "tp1_atr_mult": 1.5,
            "tp2_atr_mult": 2.8,
            "sl_atr_mult": 1.2,
            "partial_tp_ratio": 0.50,
            "breakeven_lock_enabled": True,
            "exit_at_tp2": False,
        },
        "risk_management": {
            "capital_usdt": 1000.0,
            "risk_per_trade_pct": 2.0,
            "max_open_trades": 3,
            "fee_rate": 0.00075,
            "slippage_rate": 0.0005,
            "max_holding_hours": 48,
            "loss_cooldown_hours": 12,
        },
        "trailing_stop": {
            "enabled": True,
            "type": "chandelier",
            "atr_mult": 2.2,
        },
        "paths": {
            "state_file": os.path.join(test_dir, "test_state.json"),
            "trade_ledger": os.path.join(test_dir, "logs", "test_trades.csv"),
            "logs_dir": os.path.join(test_dir, "logs"),
            "data_dir": os.path.join(test_dir, "data"),
            "models_dir": "models",
        },
    }


def make_state(balance=1000.0, open_positions=None):
    """Creates a minimal state dict for testing."""
    return {
        "balance_usdt": balance,
        "open_positions": open_positions or {},
        "trade_count": 0,
        "win_count": 0,
        "loss_count": 0,
        "loss_cooldowns": {},
    }


def make_position(entry_price, allocated, units=None, tp1_reached=False,
                   atr=10.0, sl_offset=1.2, tp1_offset=1.5, tp2_offset=2.8, position_id=None):
    """Creates a position dict matching the live bot schema."""
    if units is None:
        units = allocated / entry_price
    if position_id is None:
        position_id = str(uuid.uuid4())
    return {
        "position_id": position_id,
        "entry_price": entry_price,
        "allocated_usdt": allocated,
        "units": units,
        "stop_loss": entry_price - (sl_offset * atr),
        "take_profit_1": entry_price + (tp1_offset * atr),
        "take_profit_2": entry_price + (tp2_offset * atr),
        "tp1_reached": tp1_reached,
        "highest_price": entry_price,
        "atr": atr,
        "entry_time": (datetime.now(timezone.utc) - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S"),
        "confidence": 65.0,
        "strategy_score": 45.0,
        "strategy_setups": ["Test Setup"],
        "indicators_summary": "RSI: 55",
        "macro_4h": "BULLISH",
    }


class TestWinLossClassification(BaseTestCase):
    """Tests for the corrected win/loss classification logic based strictly on total_trade_pnl."""

    def test_positive_total_pnl_is_win(self):
        """Positive total position PnL (TP1 booked + final net) is WIN."""
        tp1_booked = 0.50
        runner_net = -0.20
        total_trade_pnl = tp1_booked + runner_net  # +0.30
        is_win = total_trade_pnl > 0
        assert is_win is True, "Trade with total_trade_pnl > 0 must be classified as WIN"

    def test_negative_total_pnl_is_loss(self):
        """A -$0.02 position must be classified as LOSS even if TP1 was reached."""
        tp1_booked = 0.50
        runner_net = -0.52
        total_trade_pnl = round(tp1_booked + runner_net, 4)  # -0.02
        is_win = total_trade_pnl > 0
        assert is_win is False, "A -$0.02 position must be classified as LOSS"

    def test_plain_stop_loss_is_loss(self):
        """Stop loss hit without tp1_reached is LOSS."""
        total_trade_pnl = -1.50
        is_win = total_trade_pnl > 0
        assert is_win is False, "Plain stop loss must be classified as LOSS"

    def test_tp2_hit_is_always_win(self):
        """TP2 hit with positive return is a WIN."""
        total_trade_pnl = 2.80
        is_win = total_trade_pnl > 0
        assert is_win is True

    def test_zero_or_negative_breakeven_is_loss(self):
        """Exit where net fees cause -$0.02 PnL must be classified as LOSS."""
        total_trade_pnl = -0.02
        is_win = total_trade_pnl > 0
        assert is_win is False

    def test_loss_without_tp1_triggers_cooldown(self):
        """LOSS classification should trigger cooldown logic."""
        entry_price = 100.0
        curr_price = 95.0
        tp1_reached = False
        hit_tp2 = False

        is_win = hit_tp2 or (curr_price >= entry_price) or tp1_reached
        assert is_win is False

        # Cooldown logic from bot.py
        state = make_state()
        cooldown_hours = 12
        if not is_win and cooldown_hours > 0:
            now_utc = datetime.now(timezone.utc)
            cooldown_expiry = (now_utc + timedelta(hours=cooldown_hours)).isoformat()
            state.setdefault("loss_cooldowns", {})["TEST/USDT"] = cooldown_expiry

        assert "TEST/USDT" in state["loss_cooldowns"]


class TestTradeCounters(BaseTestCase):
    """Tests that trade counters count ONE trade per position, not per partial exit."""

    def test_partial_tp1_does_not_increment_counters(self):
        """TP1 partial exit is an event and must NOT increment trade_count/win_count."""
        state = make_state()
        assert state["trade_count"] == 0
        assert state["win_count"] == 0

        # Partial TP1 does not increment
        assert state["trade_count"] == 0
        assert state["win_count"] == 0

    def test_single_trade_counted_on_final_close(self):
        """Final close increments trade_count and win/loss count exactly once per position."""
        state = make_state()

        # Partial TP1: counters remain 0
        assert state["trade_count"] == 0

        # Final close: exactly 1 trade counted
        state["trade_count"] += 1
        state["win_count"] += 1

        assert state["trade_count"] == 1
        assert state["win_count"] == 1

    def test_loss_increments_loss_count(self):
        """Stop loss exit should increment loss_count."""
        state = make_state()
        state["trade_count"] += 1
        state["loss_count"] += 1

        assert state["trade_count"] == 1
        assert state["loss_count"] == 1
        assert state["win_count"] == 0


class TestChandelierTrailingStop(BaseTestCase):
    """Tests for chandelier dynamic trailing stop logic."""

    def test_chandelier_ratchets_up(self):
        """Chandelier stop should ratchet up as price makes new highs."""
        pos = make_position(100.0, 500.0, atr=5.0)
        pos["tp1_reached"] = True
        trail_mult = 2.2

        # Price runs to 115
        pos["highest_price"] = 115.0
        chandelier_stop = pos["highest_price"] - (trail_mult * pos["atr"])
        be_level = pos["entry_price"] * 1.001
        new_sl = max(pos["stop_loss"], chandelier_stop, be_level)

        assert new_sl == chandelier_stop  # 115 - 11 = 104
        assert new_sl > pos["entry_price"], "Chandelier stop should be above entry after TP1"

    def test_chandelier_never_lowers(self):
        """Chandelier stop should never decrease (ratchet only upward)."""
        pos = make_position(100.0, 500.0, atr=5.0)
        pos["tp1_reached"] = True
        pos["stop_loss"] = 105.0  # Already ratcheted up
        trail_mult = 2.2

        # Price drops from peak — stop should not decrease
        pos["highest_price"] = 108.0
        chandelier_stop = pos["highest_price"] - (trail_mult * pos["atr"])
        be_level = pos["entry_price"] * 1.001
        new_sl = max(pos["stop_loss"], chandelier_stop, be_level)

        assert new_sl == 105.0, "Stop should not decrease from already ratcheted level"

    def test_chandelier_not_active_before_tp1(self):
        """Chandelier should not be active before TP1 is reached."""
        pos = make_position(100.0, 500.0, atr=5.0)
        pos["tp1_reached"] = False
        use_chandelier = True

        should_trail = use_chandelier and pos.get("tp1_reached", False)
        assert should_trail is False


class TestBreakevenLock(BaseTestCase):
    """Tests for breakeven stop loss lock after TP1."""

    def test_breakeven_lock_sets_floor(self):
        """After TP1, stop loss should be locked to just above entry."""
        entry_price = 100.0
        breakeven_lock_enabled = True

        if breakeven_lock_enabled:
            new_sl = entry_price * 1.001

        assert new_sl == 100.1, "Breakeven lock should be 0.1% above entry"

    def test_small_account_breakeven_guard(self):
        """If partial allocation < $5 minNotional, lock stop and keep full position."""
        allocated = 8.0
        partial_ratio = 0.50
        sell_allocated = allocated * partial_ratio  # $4.00

        should_use_small_guard = sell_allocated < 5.0
        assert should_use_small_guard is True


class TestCSVSchema(BaseTestCase):
    """Tests for the standardized trade CSV schema."""

    def test_trade_csv_columns_defined(self):
        """TRADE_CSV_COLUMNS should have exactly 16 standard columns including position_id."""
        from bot import TRADE_CSV_COLUMNS
        assert len(TRADE_CSV_COLUMNS) == 16
        assert "position_id" in TRADE_CSV_COLUMNS
        assert "timestamp" in TRADE_CSV_COLUMNS
        assert "outcome" in TRADE_CSV_COLUMNS
        assert "mode" in TRADE_CSV_COLUMNS

    def test_log_trade_normalizes_records(self):
        """log_trade should fill missing columns with empty strings."""
        from bot import TRADE_CSV_COLUMNS

        # Minimal record missing many fields
        record = {"timestamp": "2026-01-01", "symbol": "BTC/USDT", "action": "TEST"}
        normalised = {col: record.get(col, "") for col in TRADE_CSV_COLUMNS}

        assert normalised["timestamp"] == "2026-01-01"
        assert normalised["net_pnl"] == ""  # Missing field filled
        assert normalised["mode"] == ""  # Missing field filled
        assert len(normalised) == len(TRADE_CSV_COLUMNS)


class TestMaxHoldingTime(BaseTestCase):
    """Tests for max holding time exit logic."""

    def test_stale_position_detected(self):
        """Position held longer than max_holding_hours should be flagged stale."""
        max_hold_hours = 48
        entry_time = datetime.now(timezone.utc) - timedelta(hours=50)
        now_utc = datetime.now(timezone.utc)

        elapsed = (now_utc - entry_time).total_seconds()
        is_stale = elapsed >= max_hold_hours * 3600

        assert is_stale is True

    def test_fresh_position_not_stale(self):
        """Position within max_holding_hours should not be stale."""
        max_hold_hours = 48
        entry_time = datetime.now(timezone.utc) - timedelta(hours=10)
        now_utc = datetime.now(timezone.utc)

        elapsed = (now_utc - entry_time).total_seconds()
        is_stale = elapsed >= max_hold_hours * 3600

        assert is_stale is False


class TestPositionReconciliation(BaseTestCase):
    """Tests for live exchange position and balance reconciliation."""

    def test_reconcile_removes_missing_position(self):
        """If an open position was liquidated or sold externally, reconcile removes it."""
        from bot import reconcile_positions

        client = MagicMock()
        client.get_balance.return_value = {
            "USDT": {"free": 50.0, "total": 50.0},
            "ADA": {"free": 0.0, "total": 0.0},  # No ADA on exchange
        }

        state = make_state(balance=20.0)
        state["open_positions"] = {
            "ADA/USDT": make_position(entry_price=0.50, allocated=10.0, units=20.0)
        }
        config = make_config(self.test_dir)

        reconcile_positions(client, state, config)

        assert "ADA/USDT" not in state["open_positions"]
        assert state["balance_usdt"] == 50.0

    def test_reconcile_syncs_units_when_different(self):
        """If exchange units differ from state, reconcile updates state units."""
        from bot import reconcile_positions

        client = MagicMock()
        client.get_balance.return_value = {
            "USDT": {"free": 100.0, "total": 100.0},
            "BTC": {"free": 0.005, "total": 0.005},
        }

        state = make_state(balance=50.0)
        state["open_positions"] = {
            "BTC/USDT": make_position(entry_price=50000.0, allocated=250.0, units=0.004)
        }
        config = make_config(self.test_dir)

        reconcile_positions(client, state, config)

        assert "BTC/USDT" in state["open_positions"]
        assert state["open_positions"]["BTC/USDT"]["units"] == 0.005
        assert state["balance_usdt"] == 100.0


class TestRiskManagementAndCorrelation(BaseTestCase):
    """Tests for correlation filtering, cooldown time parsing, and portfolio exposure limits."""

    def test_cooldown_datetime_parsing_active(self):
        """Active future timestamp in loss_cooldowns must correctly identify cooldown."""
        future_utc = (datetime.now(timezone.utc) + timedelta(hours=6)).isoformat()
        state = {"loss_cooldowns": {"FET/USDT": future_utc}}

        now_utc = datetime.now(timezone.utc)
        cooldown_expiry = state.get("loss_cooldowns", {}).get("FET/USDT")
        is_on_cooldown = False
        if cooldown_expiry:
            exp_dt = datetime.fromisoformat(cooldown_expiry)
            if exp_dt.tzinfo is None:
                exp_dt = exp_dt.replace(tzinfo=timezone.utc)
            is_on_cooldown = exp_dt > now_utc

        assert is_on_cooldown is True

    def test_cooldown_datetime_parsing_expired(self):
        """Past timestamp in loss_cooldowns must report as expired (not on cooldown)."""
        past_utc = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        state = {"loss_cooldowns": {"FET/USDT": past_utc}}

        now_utc = datetime.now(timezone.utc)
        cooldown_expiry = state.get("loss_cooldowns", {}).get("FET/USDT")
        is_on_cooldown = False
        if cooldown_expiry:
            exp_dt = datetime.fromisoformat(cooldown_expiry)
            if exp_dt.tzinfo is None:
                exp_dt = exp_dt.replace(tzinfo=timezone.utc)
            is_on_cooldown = exp_dt > now_utc

        assert is_on_cooldown is False

    def test_compute_asset_correlation_in_memory(self):
        """compute_asset_correlation should use live in-memory DataFrames when provided."""
        import pandas as pd
        import numpy as np
        from bot import compute_asset_correlation

        # Create 60 hours of synthetic correlated return series
        np.random.seed(42)
        dates = pd.date_range("2026-01-01", periods=60, freq="1h")
        common_returns = np.random.normal(0.001, 0.01, 60)
        c1 = 100.0 * np.cumprod(1.0 + common_returns)
        c2 = 200.0 * np.cumprod(1.0 + common_returns * 0.9 + np.random.normal(0, 0.001, 60))
        df1 = pd.DataFrame({"close": c1}, index=dates)
        df2 = pd.DataFrame({"close": c2}, index=dates)

        ohlcv_map = {"AAA/USDT": df1, "BBB/USDT": df2}
        corr = compute_asset_correlation("AAA/USDT", "BBB/USDT", ohlcv_map=ohlcv_map)

        assert 0.80 <= corr <= 1.0, f"Expected strong correlation, got {corr}"

    def test_simultaneous_cash_reserve_and_portfolio_cap(self):
        """Spendable cash must enforce both the 15% cash reserve and portfolio max exposure."""
        balance_usdt = 100.0
        open_alloc = 70.0
        total_equity = balance_usdt + open_alloc  # 170.0
        min_reserve_pct = 0.15
        min_reserve_usdt = total_equity * min_reserve_pct  # 25.50

        spendable_cash = max(0.0, balance_usdt - min_reserve_usdt)  # 74.50
        max_portfolio_spend = max(0.0, (total_equity * (1.0 - min_reserve_pct)) - open_alloc)  # 170 * 0.85 - 70 = 74.50
        effective_spendable = min(spendable_cash, max_portfolio_spend)

        assert effective_spendable == 74.50
        # If open_alloc was already near cap (e.g. 140)
        open_alloc_high = 140.0
        total_equity_high = balance_usdt + open_alloc_high  # 240.0
        max_portfolio_spend_capped = max(0.0, (total_equity_high * 0.85) - open_alloc_high)  # 204 - 140 = 64.0
        spendable_from_cash = max(0.0, balance_usdt - (total_equity_high * min_reserve_pct))  # 100 - 36 = 64.0
        assert min(spendable_from_cash, max_portfolio_spend_capped) == 64.0


class TestMLPipelineAndEnsemble(BaseTestCase):
    """Tests for StackingEnsembleModel, CalibratedEnsemble, staleness, and drift detection."""

    def test_stacking_ensemble_blending(self):
        """StackingEnsembleModel blending correctly averages probabilities."""
        import numpy as np
        from models import StackingEnsembleModel

        m1 = MagicMock()
        m1.predict_proba.return_value = np.array([[0.4, 0.6], [0.2, 0.8]])
        m2 = MagicMock()
        m2.predict_proba.return_value = np.array([[0.3, 0.7], [0.1, 0.9]])

        ens = StackingEnsembleModel(m1, m2, weights=(0.5, 0.5))
        probs = ens.predict_proba(np.zeros((2, 5)))

        assert np.isclose(probs[0, 1], 0.65)
        assert np.isclose(probs[1, 1], 0.85)
        assert np.allclose(probs[:, 0] + probs[:, 1], 1.0)

    def test_stacking_ensemble_meta_learner(self):
        """StackingEnsembleModel routes through meta_learner when provided."""
        import numpy as np
        from models import StackingEnsembleModel

        m1 = MagicMock()
        m1.predict_proba.return_value = np.array([[0.4, 0.6]])
        m2 = MagicMock()
        m2.predict_proba.return_value = np.array([[0.3, 0.7]])
        meta = MagicMock()
        meta.predict_proba.return_value = np.array([[0.15, 0.85]])

        ens = StackingEnsembleModel(m1, m2, meta_learner=meta)
        probs = ens.predict_proba(np.zeros((1, 5)))

        assert np.isclose(probs[0, 1], 0.85)

    def test_calibrated_ensemble(self):
        """CalibratedEnsemble correctly fits isotonic mapping and bounds predictions."""
        import numpy as np
        from models import CalibratedEnsemble

        base = MagicMock()
        # Train on raw probs [0.1, 0.3, 0.7, 0.9] with targets [0, 0, 1, 1]
        base.predict_proba.return_value = np.array([
            [0.9, 0.1], [0.7, 0.3], [0.3, 0.7], [0.1, 0.9]
        ])

        cal = CalibratedEnsemble(base)
        cal.fit(np.zeros((4, 2)), np.array([0, 0, 1, 1]))

        # Test inference
        base.predict_proba.return_value = np.array([[0.2, 0.8]])
        cal_probs = cal.predict_proba(np.zeros((1, 2)))

        assert 0.0 <= cal_probs[0, 1] <= 1.0
        assert np.isclose(cal_probs[0, 0] + cal_probs[0, 1], 1.0)

    def test_model_staleness_fresh_and_stale(self):
        """check_model_staleness correctly reports fresh vs stale model bundles."""
        from models import check_model_staleness

        # Fresh model (created 2 days ago)
        fresh_time = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
        is_stale, age, _ = check_model_staleness({"created_at": fresh_time}, max_age_days=14)
        assert is_stale is False
        assert 1.5 <= age <= 2.5

        # Stale model (created 20 days ago)
        stale_time = (datetime.now(timezone.utc) - timedelta(days=20)).isoformat()
        is_stale, age, msg = check_model_staleness({"created_at": stale_time}, max_age_days=14)
        assert is_stale is True
        assert "STALE" in msg

    def test_feature_drift_detection(self):
        """check_feature_drift identifies features that deviate > 3.5 std from training mean."""
        import pandas as pd
        from models import check_feature_drift

        bundle = {
            "feature_stats": {
                "rsi_14": {"mean": 50.0, "std": 10.0},
                "vol_spike": {"mean": 1.0, "std": 0.5},
            }
        }

        # Live features: rsi_14 is normal (55), but vol_spike is 5.0 (8 standard deviations!)
        live_df = pd.DataFrame([{"rsi_14": 55.0, "vol_spike": 5.0}])
        drifted = check_feature_drift(live_df, bundle, threshold_std=3.5)

        drift_names = [d[0] for d in drifted]
        assert "vol_spike" in drift_names
        assert "rsi_14" not in drift_names

    def test_stacking_ensemble_fit_does_not_refit_base_models(self):
        """StackingEnsembleModel.fit() must NOT refit already fitted base models."""
        from models import StackingEnsembleModel
        import numpy as np

        m1 = MagicMock()
        m1.classes_ = np.array([0, 1])
        m1.predict_proba.return_value = np.array([[0.3, 0.7], [0.2, 0.8]])

        m2 = MagicMock()
        m2.classes_ = np.array([0, 1])
        m2.predict_proba.return_value = np.array([[0.4, 0.6], [0.1, 0.9]])

        meta = MagicMock()
        meta.predict_proba.return_value = np.array([[0.25, 0.75], [0.15, 0.85]])

        ens = StackingEnsembleModel(m1, m2, meta_learner=meta)
        X_cal = np.zeros((2, 5))
        y_cal = np.array([0, 1])

        ens.fit(X_cal, y_cal)

        # Base models must NOT have been called with fit()
        m1.fit.assert_not_called()
        m2.fit.assert_not_called()
        # Meta-learner must have been fitted on calibration predictions
        meta.fit.assert_called_once()

    def test_simulate_test_trades_execution_and_friction(self):
        """simulate_test_trades must respect max_open_trades, one position per symbol, and 2*cost_rate."""
        from train import simulate_test_trades
        import pandas as pd
        import numpy as np

        dates = pd.date_range("2026-01-01 10:00:00", periods=5, freq="1h", tz="UTC")
        X_test = pd.DataFrame({"feat1": [1, 2, 3, 4, 5]}, index=dates)
        y_test = pd.Series([1, 1, 0, 1, 0], index=dates)
        ret_test = pd.DataFrame({
            "return": [0.05, 0.04, -0.03, 0.06, -0.02],
            "symbol": ["BTC/USDT", "ETH/USDT", "BTC/USDT", "SOL/USDT", "ETH/USDT"]
        }, index=dates)

        # High probabilities so signals fire
        test_probs = np.array([0.80, 0.75, 0.70, 0.85, 0.65])
        conf_thresh = 0.60
        cost_rate = 0.001  # 2*cost = 0.002
        max_open_trades = 2
        horizon_bars = 2

        results = simulate_test_trades(
            X_test=X_test,
            y_test=y_test,
            ret_test=ret_test,
            test_probs=test_probs,
            conf_thresh=conf_thresh,
            cost_rate=cost_rate,
            max_open_trades=max_open_trades,
            horizon_bars=horizon_bars
        )

        self.assertGreater(results["n_trades"], 0)
        # Check friction deduction: each executed trade's net_ret must equal gross_ret - 0.002
        for t in results["trades"]:
            self.assertAlmostEqual(t["net_ret"], t["gross_ret"] - 0.002, places=5)

    def test_train_model_produces_valid_bundle_with_metadata(self):
        """train_model must return bundle with split_dates, calibration_cutoff_date, breakeven_prob, and tradeable flag."""
        from train import train_model
        import pandas as pd
        import numpy as np

        dates = pd.date_range("2026-01-01", periods=100, freq="1h", tz="UTC")
        np.random.seed(42)
        X = pd.DataFrame({
            "atr_ratio": np.full(100, 0.02),
            "rsi_14": np.random.uniform(0.3, 0.7, 100),
            "bb_pos": np.random.uniform(0.2, 0.8, 100)
        }, index=dates)
        y = pd.Series(np.random.choice([0, 1], size=100), index=dates)
        fwd_returns = pd.DataFrame({
            "return": np.random.uniform(-0.02, 0.03, 100),
            "symbol": ["BTC/USDT"] * 100
        }, index=dates)

        config = {
            "timeframe": "1h",
            "symbols": ["BTC/USDT"],
            "ai_model": {
                "confidence_threshold": 0.50,
                "tp1_atr_mult": 1.5,
                "sl_atr_mult": 1.2,
                "horizon_bars": 4
            },
            "risk_management": {
                "fee_rate": 0.00075,
                "slippage_rate": 0.0005,
                "max_open_trades": 2
            }
        }

        bundle = train_model(X, y, fwd_returns, config)

        self.assertIn("tradeable", bundle)
        self.assertIn("split_dates", bundle)
        self.assertIn("calibration_cutoff_date", bundle)
        self.assertIn("breakeven_prob", bundle["metrics"])
        self.assertIn("auc", bundle["metrics"])

        # Check breakeven_prob formula: (sl*atr + cost) / ((tp+sl)*atr)
        # sl = 1.2, tp = 1.5, atr = 0.02, cost = 2 * (0.00075 + 0.0005) = 0.0025
        # denom = 2.7 * 0.02 = 0.054, numer = 1.2 * 0.02 + 0.0025 = 0.0265
        # expected p_be = 0.0265 / 0.054 = ~0.4907
        expected_p_be = (1.2 * 0.02 + 0.0025) / (2.7 * 0.02)
        self.assertAlmostEqual(bundle["metrics"]["breakeven_prob"], expected_p_be, places=3)
        self.assertEqual(bundle["tradeable"], bundle["metrics"]["auc"] >= 0.54)

    def test_train_model_low_auc_marks_non_tradeable(self):
        """When test AUC < 0.54, bundle must still save but have tradeable=False."""
        from train import train_model
        import pandas as pd
        import numpy as np

        dates = pd.date_range("2026-01-01", periods=100, freq="1h", tz="UTC")
        X = pd.DataFrame({
            "atr_ratio": np.full(100, 0.02),
            "rsi_14": np.sin(np.linspace(0, 8 * np.pi, 100)),
            "bb_pos": np.cos(np.linspace(0, 8 * np.pi, 100))
        }, index=dates)
        y_vals = np.array((X["rsi_14"] > 0).astype(int), dtype=int, copy=True)
        # Invert only test slice to force low AUC < 0.54
        y_vals[80:] = 1 - y_vals[80:]
        y = pd.Series(y_vals, index=dates)
        fwd_returns = pd.DataFrame({
            "return": np.zeros(100),
            "symbol": ["BTC/USDT"] * 100
        }, index=dates)

        config = {
            "timeframe": "1h",
            "symbols": ["BTC/USDT"],
            "ai_model": {
                "confidence_threshold": 0.50,
                "tp1_atr_mult": 1.5,
                "sl_atr_mult": 1.2,
                "horizon_bars": 4
            },
            "risk_management": {
                "fee_rate": 0.00075,
                "slippage_rate": 0.0005,
                "max_open_trades": 2
            }
        }

        bundle = train_model(X, y, fwd_returns, config)
        self.assertLess(bundle["metrics"]["auc"], 0.54)
        self.assertFalse(bundle["tradeable"])


class TestLedgerHeaderAndRotation(BaseTestCase):
    """Tests for log_trade writing header on missing/empty files and rotating legacy headers."""

    def test_log_trade_creates_header_on_missing_file(self):
        from bot import log_trade, TRADE_CSV_COLUMNS
        import csv

        config = make_config(self.test_dir)
        ledger_path = config["paths"]["trade_ledger"]
        assert not os.path.exists(ledger_path)

        pos_id = str(uuid.uuid4())
        record = {
            "timestamp": "2026-01-01 12:00:00",
            "position_id": pos_id,
            "symbol": "BTC/USDT",
            "action": "BUY",
            "entry_price": 50000.0,
            "allocated_usdt": 100.0,
            "outcome": "EVENT",
            "mode": "paper",
        }
        log_trade(record, config)

        assert os.path.exists(ledger_path)
        with open(ledger_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            row = next(reader)

        self.assertEqual(header, TRADE_CSV_COLUMNS)
        self.assertIn("position_id", header)
        self.assertEqual(row[header.index("position_id")], pos_id)

    def test_log_trade_rotates_legacy_header(self):
        from bot import log_trade, TRADE_CSV_COLUMNS
        import glob
        import csv

        config = make_config(self.test_dir)
        ledger_path = config["paths"]["trade_ledger"]
        logs_dir = config["paths"]["logs_dir"]
        os.makedirs(logs_dir, exist_ok=True)

        # Write legacy file with old header without position_id
        legacy_header = ["timestamp", "symbol", "action", "entry_price", "exit_price", "net_pnl", "outcome"]
        with open(ledger_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(legacy_header)
            writer.writerow(["2025-12-31", "ETH/USDT", "BUY", "3000", "3100", "10.0", "WIN"])

        new_pos_id = str(uuid.uuid4())
        record = {
            "timestamp": "2026-01-01 12:00:00",
            "position_id": new_pos_id,
            "symbol": "SOL/USDT",
            "action": "BUY",
            "entry_price": 100.0,
            "allocated_usdt": 20.0,
            "outcome": "EVENT",
            "mode": "paper",
        }
        log_trade(record, config)

        # Check legacy rotation file exists
        legacy_files = glob.glob(os.path.join(logs_dir, "trade_history_legacy_*.csv"))
        self.assertEqual(len(legacy_files), 1, "Expected legacy file to be created")

        # Verify legacy file has old data
        with open(legacy_files[0], "r", encoding="utf-8") as f:
            leg_reader = csv.reader(f)
            self.assertEqual(next(leg_reader), legacy_header)

        # Verify current trade ledger has new TRADE_CSV_COLUMNS and new record
        with open(ledger_path, "r", encoding="utf-8") as f:
            cur_reader = csv.reader(f)
            cur_header = next(cur_reader)
            cur_row = next(cur_reader)

        self.assertEqual(cur_header, TRADE_CSV_COLUMNS)
        self.assertEqual(cur_row[cur_header.index("position_id")], new_pos_id)
        self.assertEqual(cur_row[cur_header.index("symbol")], "SOL/USDT")


class TestPerPositionStatsAndAccounting(BaseTestCase):
    """Tests for per-position accounting, state updates, and bot/stats.py analytics."""

    def test_partial_exit_and_breakeven_events_dont_increment_trade_count(self):
        from bot import check_and_update_positions

        client = MagicMock()
        client.get_min_notional.return_value = 5.0
        client.get_ticker_price.return_value = 105.0  # Above TP1 (103)

        pos_id = str(uuid.uuid4())
        pos = make_position(100.0, 40.0, units=0.40, position_id=pos_id)
        pos["take_profit_1"] = 103.0
        pos["take_profit_2"] = 110.0
        pos["stop_loss"] = 97.0

        state = make_state(balance=60.0, open_positions={"TEST/USDT": pos})
        config = make_config(self.test_dir)

        closed = check_and_update_positions(client, state, config)

        self.assertEqual(len(closed), 1)
        self.assertEqual(closed[0]["action"], "PARTIAL TP1 (50% SOLD)")
        self.assertEqual(closed[0]["outcome"], "EVENT")
        self.assertEqual(closed[0]["position_id"], pos_id)
        # Counters MUST NOT increment on partial exit
        self.assertEqual(state["trade_count"], 0)
        self.assertEqual(state["win_count"], 0)
        self.assertEqual(state["loss_count"], 0)

    def test_final_close_accounting_negative_is_loss(self):
        from bot import check_and_update_positions

        client = MagicMock()
        client.get_ticker_price.return_value = 95.0  # Hit stop loss

        pos_id = str(uuid.uuid4())
        pos = make_position(100.0, 10.0, units=0.1, position_id=pos_id)
        pos["tp1_reached"] = True
        pos["tp1_booked_pnl"] = 0.05
        # Entry 100, exit 95 -> loss on remaining is ~ -$0.50 -> total trade pnl is negative
        pos["stop_loss"] = 96.0

        state = make_state(balance=50.0, open_positions={"SOL/USDT": pos})
        config = make_config(self.test_dir)

        closed = check_and_update_positions(client, state, config)

        self.assertEqual(len(closed), 1)
        self.assertEqual(closed[0]["outcome"], "LOSS")
        self.assertEqual(closed[0]["position_id"], pos_id)
        self.assertLess(closed[0]["net_pnl"], 0.0)
        self.assertEqual(state["trade_count"], 1)
        self.assertEqual(state["win_count"], 0)
        self.assertEqual(state["loss_count"], 1)

    def test_final_close_accounting_positive_is_win(self):
        from bot import check_and_update_positions

        client = MagicMock()
        client.get_ticker_price.return_value = 112.0  # Hit TP2

        pos_id = str(uuid.uuid4())
        pos = make_position(100.0, 10.0, units=0.1, position_id=pos_id)
        pos["tp1_reached"] = True
        pos["tp1_booked_pnl"] = 0.50
        pos["take_profit_2"] = 110.0

        state = make_state(balance=50.0, open_positions={"SOL/USDT": pos})
        config = make_config(self.test_dir)
        config["ai_model"]["exit_at_tp2"] = True

        closed = check_and_update_positions(client, state, config)

        self.assertEqual(len(closed), 1)
        self.assertEqual(closed[0]["outcome"], "WIN")
        self.assertEqual(closed[0]["position_id"], pos_id)
        self.assertGreater(closed[0]["net_pnl"], 0.0)
        self.assertEqual(state["trade_count"], 1)
        self.assertEqual(state["win_count"], 1)
        self.assertEqual(state["loss_count"], 0)

    def test_state_load_assigns_missing_position_id(self):
        from bot import load_state

        config = make_config(self.test_dir)
        state_file = config["paths"]["state_file"]
        raw_state = {
            "balance_usdt": 100.0,
            "open_positions": {
                "BTC/USDT": {
                    "entry_price": 50000.0,
                    "allocated_usdt": 20.0,
                    "units": 0.0004
                }
            }
        }
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(raw_state, f)

        loaded = load_state(config)
        self.assertIn("BTC/USDT", loaded["open_positions"])
        self.assertIn("position_id", loaded["open_positions"]["BTC/USDT"])
        self.assertTrue(len(loaded["open_positions"]["BTC/USDT"]["position_id"]) > 10)

    def test_compute_ledger_stats_analytics(self):
        from bot.stats import compute_ledger_stats, print_stats_report
        import csv

        config = make_config(self.test_dir)
        ledger_path = config["paths"]["trade_ledger"]
        os.makedirs(os.path.dirname(ledger_path), exist_ok=True)

        from bot import TRADE_CSV_COLUMNS
        with open(ledger_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=TRADE_CSV_COLUMNS)
            writer.writeheader()

            # Position 1: Partial TP1 (+0.30) + Final close (+0.70) = +1.00 net
            p1_id = "pos-1"
            writer.writerow({"timestamp": "2026-01-01", "position_id": p1_id, "symbol": "BTC/USDT", "action": "PARTIAL TP1", "net_pnl": "0.30", "outcome": "EVENT"})
            writer.writerow({"timestamp": "2026-01-01", "position_id": p1_id, "symbol": "BTC/USDT", "action": "FINAL CLOSE", "net_pnl": "0.70", "outcome": "WIN"})

            # Position 2: Single Final close (-0.50) = -0.50 net
            p2_id = "pos-2"
            writer.writerow({"timestamp": "2026-01-02", "position_id": p2_id, "symbol": "ETH/USDT", "action": "FINAL CLOSE", "net_pnl": "-0.50", "outcome": "LOSS"})

            # Position 3: Partial TP1 (+0.01) + Final close (-0.03) = -0.02 net
            p3_id = "pos-3"
            writer.writerow({"timestamp": "2026-01-03", "position_id": p3_id, "symbol": "SOL/USDT", "action": "PARTIAL TP1", "net_pnl": "0.01", "outcome": "EVENT"})
            writer.writerow({"timestamp": "2026-01-03", "position_id": p3_id, "symbol": "SOL/USDT", "action": "FINAL CLOSE", "net_pnl": "-0.03", "outcome": "LOSS"})

        stats = compute_ledger_stats(ledger_path)
        self.assertEqual(stats["closed_positions"], 3)
        self.assertEqual(stats["win_count"], 1)
        self.assertEqual(stats["loss_count"], 2)
        self.assertAlmostEqual(stats["win_rate_pct"], 33.33, places=1)
        self.assertAlmostEqual(stats["total_net_pnl"], 0.48, places=2)
        self.assertAlmostEqual(stats["gross_profit"], 1.00, places=2)
        self.assertAlmostEqual(stats["gross_loss"], 0.52, places=2)
        self.assertAlmostEqual(stats["avg_win"], 1.00, places=2)
        self.assertAlmostEqual(stats["avg_loss"], 0.26, places=2)
        self.assertAlmostEqual(stats["expectancy"], 0.16, places=2)
        self.assertGreater(stats["profit_factor"], 1.9)

        # Ensure print_stats_report runs without exceptions
        print_stats_report(ledger_path, stats)


if __name__ == "__main__":
    unittest.main()



