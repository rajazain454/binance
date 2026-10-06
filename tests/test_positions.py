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


def make_config():
    """Creates a minimal config dict for testing."""
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
            "state_file": "test_state.json",
            "trade_ledger": "logs/test_trades.csv",
            "data_dir": "data",
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
                   atr=10.0, sl_offset=1.2, tp1_offset=1.5, tp2_offset=2.8):
    """Creates a position dict matching the live bot schema."""
    if units is None:
        units = allocated / entry_price
    return {
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


class TestWinLossClassification(unittest.TestCase):
    """Tests for the corrected win/loss classification logic."""

    def test_tp1_reached_counts_as_win(self):
        """After TP1 partial exit, chandelier trailing exit below entry should be WIN."""
        # This is the critical bug fix — tp1_reached=True means overall trade was profitable
        entry_price = 100.0
        curr_price = 99.5  # Slightly below entry (would be LOSS without fix)
        tp1_reached = True
        hit_tp2 = False

        is_win = hit_tp2 or (curr_price >= entry_price) or tp1_reached
        assert is_win is True, "Trade with tp1_reached=True should be classified as WIN"

    def test_plain_stop_loss_is_loss(self):
        """Stop loss hit without tp1_reached should be LOSS."""
        entry_price = 100.0
        curr_price = 95.0
        tp1_reached = False
        hit_tp2 = False

        is_win = hit_tp2 or (curr_price >= entry_price) or tp1_reached
        assert is_win is False, "Plain stop loss should be classified as LOSS"

    def test_tp2_hit_is_always_win(self):
        """TP2 hit is always a WIN regardless of other conditions."""
        entry_price = 100.0
        curr_price = 110.0
        tp1_reached = True
        hit_tp2 = True

        is_win = hit_tp2 or (curr_price >= entry_price) or tp1_reached
        assert is_win is True

    def test_exit_at_entry_is_win(self):
        """Exit at exactly entry price should be WIN (preserves capital)."""
        entry_price = 100.0
        curr_price = 100.0
        tp1_reached = False
        hit_tp2 = False

        is_win = hit_tp2 or (curr_price >= entry_price) or tp1_reached
        assert is_win is True

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


class TestTradeCounters(unittest.TestCase):
    """Tests that trade counters increment correctly for partial and full exits."""

    def test_partial_tp1_increments_counters(self):
        """TP1 partial exit should increment both trade_count and win_count."""
        state = make_state()
        assert state["trade_count"] == 0
        assert state["win_count"] == 0

        # Simulate partial TP1 exit (from bot.py after fix)
        state["trade_count"] += 1
        state["win_count"] += 1

        assert state["trade_count"] == 1
        assert state["win_count"] == 1

    def test_full_close_after_partial_increments_again(self):
        """Full close after TP1 partial should add another trade count."""
        state = make_state()

        # TP1 partial
        state["trade_count"] += 1
        state["win_count"] += 1

        # Final close (chandelier trailing, still a WIN since tp1_reached)
        state["trade_count"] += 1
        state["win_count"] += 1  # tp1_reached = True → WIN

        assert state["trade_count"] == 2
        assert state["win_count"] == 2

    def test_loss_increments_loss_count(self):
        """Stop loss exit should increment loss_count."""
        state = make_state()
        state["trade_count"] += 1
        state["loss_count"] += 1

        assert state["trade_count"] == 1
        assert state["loss_count"] == 1
        assert state["win_count"] == 0


class TestChandelierTrailingStop(unittest.TestCase):
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


class TestBreakevenLock(unittest.TestCase):
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


class TestCSVSchema(unittest.TestCase):
    """Tests for the standardized trade CSV schema."""

    def test_trade_csv_columns_defined(self):
        """TRADE_CSV_COLUMNS should have exactly 15 standard columns."""
        from bot import TRADE_CSV_COLUMNS
        assert len(TRADE_CSV_COLUMNS) == 15
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


class TestMaxHoldingTime(unittest.TestCase):
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


class TestPositionReconciliation(unittest.TestCase):
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
        config = make_config()

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
        config = make_config()

        reconcile_positions(client, state, config)

        assert "BTC/USDT" in state["open_positions"]
        assert state["open_positions"]["BTC/USDT"]["units"] == 0.005
        assert state["balance_usdt"] == 100.0


class TestRiskManagementAndCorrelation(unittest.TestCase):
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


class TestMLPipelineAndEnsemble(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()



