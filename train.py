"""Machine Learning Training Pipeline for Binance Quantitative Trading.
Pulls Binance market data, extracts multi-tool strategy features,
applies Triple-Barrier labeling, trains an ensemble Gradient Boosting model,
validates out-of-sample performance, and saves the calibrated model artifact.
"""

import os
import sys
import json
import argparse
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score

import features
from binance_client import BinanceClient
from utils import load_config


def prepare_dataset(client, symbols, tf, days=365, ai_cfg=None, data_dir="data", offline=False):
    """Fetches Binance data or loads 365d cache, computes features and labels for each symbol."""
    ai_cfg = ai_cfg or {}
    horizon = ai_cfg.get("horizon_bars", 12)
    # Read tp_atr_mult under tp1_atr_mult (as used by live exit logic)
    tp_mult = ai_cfg.get("tp1_atr_mult", ai_cfg.get("tp_atr_mult", 1.5))
    sl_mult = ai_cfg.get("sl_atr_mult", 1.2)

    all_X = []
    all_y = []
    all_ret = []
    symbol_frames = {}

    os.makedirs(data_dir, exist_ok=True)

    for sym in symbols:
        clean_sym = sym.replace("/", "_")
        cache_file = os.path.join(data_dir, f"binance_{clean_sym}_{tf}.csv")

        if (offline or os.path.exists(cache_file)):
            print(f"Loading {sym} from local cache ({cache_file})...")
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            if len(df) < 500 and not offline:
                print(f"  Cached data has only {len(df)} bars, fetching fresh {days} days from Binance...")
                try:
                    fresh_df = client.fetch_historical_ohlcv(sym, timeframe=tf, days=days)
                    if not fresh_df.empty:
                        df = fresh_df
                        df.to_csv(cache_file)
                except Exception as e:
                    print(f"  [Warning] Could not fetch fresh data: {e}")
        else:
            print(f"Fetching Binance {sym} ({tf}) history ({days} days)...")
            try:
                df = client.fetch_historical_ohlcv(sym, timeframe=tf, days=days)
                if not df.empty:
                    df.to_csv(cache_file)
            except Exception as e:
                print(f"  [Warning] Failed to fetch live data for {sym}: {e}")
                df = pd.DataFrame()

        if len(df) < 200:
            print(f"  [Skip] Insufficient bars for {sym} ({len(df)} bars)")
            continue

        symbol_frames[sym] = df

        # Compute features & labels
        feat_df = features.extract_features(df)
        labels, fwd_ret = features.label_triple_barrier(df, horizon_bars=horizon, tp_atr_mult=tp_mult, sl_atr_mult=sl_mult)

        # Align
        common_idx = feat_df.index.intersection(labels.index).intersection(fwd_ret.index)
        valid_idx = common_idx[:-horizon] if len(common_idx) > horizon else common_idx

        X_sym = feat_df.loc[valid_idx]
        y_sym = labels.loc[valid_idx]
        ret_sym = pd.DataFrame({"return": fwd_ret.loc[valid_idx], "symbol": sym}, index=valid_idx)

        all_X.append(X_sym)
        all_y.append(y_sym)
        all_ret.append(ret_sym)
        print(f"  [OK] {sym}: {len(X_sym):,} feature samples with full mathematical indicators.")

    if not all_X:
        raise ValueError("No data could be processed. Please check your network connection or Binance symbols.")

    X = pd.concat(all_X, axis=0)
    y = pd.concat(all_y, axis=0)
    fwd_returns = pd.concat(all_ret, axis=0)

    # Cross-sectional chronological sort across ALL symbols simultaneously
    if isinstance(X.index, pd.DatetimeIndex):
        sort_order = np.argsort(X.index.values)
        X = X.iloc[sort_order]
        y = y.iloc[sort_order]
        fwd_returns = fwd_returns.iloc[sort_order]

    return X, y, fwd_returns, symbol_frames


def simulate_test_trades(X_test, y_test, ret_test, test_probs, conf_thresh, cost_rate, max_open_trades, horizon_bars):
    """Simulates realistic chronological execution on the test slice:
    - Enforces max_open_trades across portfolio
    - Enforces one-position-at-a-time per symbol
    - Position held up to horizon_bars
    - Subtracts 2 * cost_rate (entry fee+slippage + exit fee+slippage)
    """
    if len(X_test) == 0:
        return {
            "n_trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
            "avg_return": 0.0, "total_pnl": 0.0, "trades": []
        }

    if isinstance(ret_test, pd.DataFrame):
        rets = ret_test["return"].values
        syms = ret_test["symbol"].values if "symbol" in ret_test.columns else np.array(["ASSET"] * len(ret_test))
    else:
        rets = ret_test.values if hasattr(ret_test, "values") else np.asarray(ret_test)
        syms = np.array(["ASSET"] * len(ret_test))

    labels = y_test.values if hasattr(y_test, "values") else np.asarray(y_test)
    probs = np.asarray(test_probs)
    timestamps = list(X_test.index)

    open_positions = {}  # {symbol: exit_timestamp}
    executed_trades = []

    from collections import defaultdict
    ts_candidates = defaultdict(list)
    for idx, ts in enumerate(timestamps):
        prob = probs[idx]
        if prob >= conf_thresh:
            ts_candidates[ts].append({
                "symbol": syms[idx],
                "prob": prob,
                "label": labels[idx],
                "ret": rets[idx],
                "index": idx
            })

    unique_ts = list(dict.fromkeys(timestamps))

    for current_ts in unique_ts:
        # 1. Close expired positions
        expired = [s for s, exit_ts in open_positions.items() if current_ts >= exit_ts]
        for s in expired:
            del open_positions[s]

        # 2. Evaluate candidate signals
        candidates = ts_candidates.get(current_ts, [])
        if candidates and len(open_positions) < max_open_trades:
            # Highest conviction first
            candidates.sort(key=lambda c: c["prob"], reverse=True)
            for cand in candidates:
                sym = cand["symbol"]
                if sym not in open_positions and len(open_positions) < max_open_trades:
                    if isinstance(current_ts, pd.Timestamp):
                        exit_ts = current_ts + pd.Timedelta(hours=horizon_bars)
                    else:
                        exit_ts = current_ts + horizon_bars
                    open_positions[sym] = exit_ts

                    net_ret = float(cand["ret"]) - (2.0 * cost_rate)
                    is_win = bool(net_ret > 0.0)
                    executed_trades.append({
                        "timestamp": current_ts,
                        "symbol": sym,
                        "prob": cand["prob"],
                        "label": cand["label"],
                        "gross_ret": float(cand["ret"]),
                        "net_ret": net_ret,
                        "is_win": is_win
                    })

    n_trades = len(executed_trades)
    if n_trades > 0:
        net_returns = np.array([t["net_ret"] for t in executed_trades])
        win_count = sum(t["is_win"] for t in executed_trades)
        win_rate = win_count / n_trades
        avg_ret = float(np.mean(net_returns))
        total_pnl = float(np.sum(net_returns))

        profit_trades = net_returns[net_returns > 0]
        loss_trades = net_returns[net_returns <= 0]
        gross_profit = float(np.sum(profit_trades)) if len(profit_trades) > 0 else 0.0
        gross_loss = float(np.abs(np.sum(loss_trades))) if len(loss_trades) > 0 else 1e-9
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    else:
        win_rate = 0.0
        avg_ret = 0.0
        total_pnl = 0.0
        profit_factor = 0.0

    return {
        "n_trades": n_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_return": avg_ret,
        "total_pnl": total_pnl,
        "trades": executed_trades
    }


def train_model(X, y, fwd_returns, config):
    """Trains an optimized Gradient Boosting ensemble with exponential time-decay weighting,
    a 3-way chronological split (60% Train, 20% Calibration, 20% Test) with purge gaps,
    and out-of-sample portfolio simulation.
    """
    feature_names = list(X.columns)
    ai_cfg = config.get("ai_model", {})
    conf_thresh = ai_cfg.get("confidence_threshold", 0.60)
    cost_rate = config["risk_management"]["fee_rate"] + config["risk_management"]["slippage_rate"]
    horizon = ai_cfg.get("horizon_bars", 12)
    tp_mult = ai_cfg.get("tp1_atr_mult", ai_cfg.get("tp_atr_mult", 1.5))
    sl_mult = ai_cfg.get("sl_atr_mult", 1.2)
    max_open_trades = config.get("risk_management", {}).get("max_open_trades", 3)

    # Cross-Sectional Chronological 3-Way Split with Purge Gaps of horizon_bars
    # 60% Train, 20% Calibration, 20% Test
    if isinstance(X.index, pd.DatetimeIndex) and len(X.index.unique()) >= 30:
        unique_dates = X.index.unique().sort_values()
        n_unique = len(unique_dates)

        train_end_idx = int(n_unique * 0.60)
        cal_end_idx = int(n_unique * 0.80)

        idx_train_end = max(1, train_end_idx)
        idx_cal_start = min(idx_train_end + horizon, cal_end_idx)
        idx_cal_end = max(idx_cal_start + 1, cal_end_idx)
        idx_test_start = min(idx_cal_end + horizon, n_unique - 1)

        train_start_date = unique_dates[0]
        train_end_date = unique_dates[idx_train_end - 1]
        cal_start_date = unique_dates[idx_cal_start]
        cal_end_date = unique_dates[idx_cal_end - 1]
        test_start_date = unique_dates[idx_test_start]
        test_end_date = unique_dates[-1]

        train_mask = (X.index >= train_start_date) & (X.index <= train_end_date)
        cal_mask = (X.index >= cal_start_date) & (X.index <= cal_end_date)
        test_mask = (X.index >= test_start_date) & (X.index <= test_end_date)
    else:
        n = len(X)
        i_train_end = max(1, int(n * 0.60))
        i_cal_start = min(i_train_end + horizon, int(n * 0.80))
        i_cal_end = max(i_cal_start + 1, int(n * 0.80))
        i_test_start = min(i_cal_end + horizon, n)

        train_mask = np.zeros(n, dtype=bool)
        train_mask[:i_train_end] = True
        cal_mask = np.zeros(n, dtype=bool)
        cal_mask[i_cal_start:i_cal_end] = True
        test_mask = np.zeros(n, dtype=bool)
        test_mask[i_test_start:] = True

        train_start_date = str(X.index[0]) if len(X) else "N/A"
        train_end_date = str(X.index[i_train_end - 1]) if i_train_end > 0 else "N/A"
        cal_start_date = str(X.index[i_cal_start]) if i_cal_start < n else "N/A"
        cal_end_date = str(X.index[i_cal_end - 1]) if i_cal_end > 0 else "N/A"
        test_start_date = str(X.index[i_test_start]) if i_test_start < n else "N/A"
        test_end_date = str(X.index[-1]) if len(X) else "N/A"

    split_dates = {
        "train_start": str(train_start_date),
        "train_end": str(train_end_date),
        "calibration_start": str(cal_start_date),
        "calibration_end": str(cal_end_date),
        "test_start": str(test_start_date),
        "test_end": str(test_end_date),
    }
    calibration_cutoff_date = str(cal_end_date)

    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    X_cal, y_cal = X.loc[cal_mask], y.loc[cal_mask]
    X_test, y_test = X.loc[test_mask], y.loc[test_mask]
    ret_test = fwd_returns.loc[test_mask]

    print("\n" + "=" * 75)
    print("  TRAINING BINANCE AI MODEL (HistGradientBoosting + LightGBM Ensemble)")
    print("=" * 75)
    print(f"  Total Clean Samples: {len(X):,}  |  Train: {len(X_train):,}  |  Calibration: {len(X_cal):,}  |  Test: {len(X_test):,}")
    print(f"  Mathematical Alpha Features: {len(feature_names)} indicators")
    print(f"  Baseline Setup Win Rate: {y.mean():.1%}")

    # Compute Exponential Time-Decay Sample Weights (half-life weighting on Train slice)
    n_train = len(X_train)
    half_life_samples = max(1000, n_train // 4)
    decay_rate = np.log(2.0) / half_life_samples
    time_weights = np.exp(-decay_rate * (n_train - 1 - np.arange(n_train)))
    time_weights = time_weights / time_weights.mean()

    # Model 1: Scikit-Learn Histogram Gradient Boosting with fixed max_iter (no random validation subsample)
    hgb_model = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.03,
        max_leaf_nodes=40,
        min_samples_leaf=30,
        l2_regularization=3.0,
        early_stopping=False,
        class_weight="balanced",
        random_state=42
    )
    print("  Fitting Model 1: HistGradientBoostingClassifier (Train slice)...")
    hgb_model.fit(X_train, y_train, sample_weight=time_weights)

    # Model 2: LightGBM (leaf-wise gradient boosting)
    import lightgbm as lgb
    lgb_model = lgb.LGBMClassifier(
        n_estimators=150,
        learning_rate=0.03,
        num_leaves=35,
        min_child_samples=30,
        reg_lambda=3.0,
        class_weight="balanced",
        random_state=42,
        verbose=-1
    )
    print("  Fitting Model 2: LightGBM Classifier (Train slice)...")
    lgb_model.fit(X_train, y_train, sample_weight=time_weights)

    # Combine into Stacking Ensemble; fit meta-learner ONLY on CALIBRATION slice
    from sklearn.linear_model import LogisticRegression
    from models import StackingEnsembleModel, CalibratedEnsemble

    meta_clf = LogisticRegression(C=1.0, random_state=42)
    raw_ensemble = StackingEnsembleModel(
        hgb_model, lgb_model, weights=(0.50, 0.50), meta_learner=meta_clf
    )
    print("  Fitting Meta-Learner (Calibration slice)...")
    raw_ensemble.fit(X_cal, y_cal)

    # Probability calibration via isotonic regression ONLY on the CALIBRATION slice
    print("  Calibrating probabilities via Isotonic Regression (Calibration slice)...")
    calibrated_model = CalibratedEnsemble(raw_ensemble)
    calibrated_model.fit(X_cal, y_cal)
    model = calibrated_model

    # Report ALL metrics ONLY on the untouched TEST slice
    test_probs = model.predict_proba(X_test)[:, 1]
    if len(np.unique(y_test)) > 1:
        auc_score = float(roc_auc_score(y_test, test_probs))
    else:
        auc_score = 0.50

    # Break-even probability from label payoff:
    # p_be = (sl_mult*atr% + cost) / ((tp_mult+sl_mult)*atr% ), using median atr_ratio on the train set
    if "atr_ratio" in X_train.columns:
        median_atr_pct = float(X_train["atr_ratio"].median())
    else:
        median_atr_pct = 0.02
    median_atr_pct = max(1e-5, median_atr_pct)

    cost = 2.0 * cost_rate
    denom = (tp_mult + sl_mult) * median_atr_pct
    numer = (sl_mult * median_atr_pct) + cost
    p_be = float(numer / denom) if denom > 0 else 0.50

    # Realistic portfolio execution simulation on test slice
    sim_results = simulate_test_trades(
        X_test=X_test,
        y_test=y_test,
        ret_test=ret_test,
        test_probs=test_probs,
        conf_thresh=conf_thresh,
        cost_rate=cost_rate,
        max_open_trades=max_open_trades,
        horizon_bars=horizon
    )

    tradeable = bool(auc_score >= 0.54)
    pf_str = f"{sim_results['profit_factor']:.2f}" if sim_results['profit_factor'] != float("inf") else "Inf"

    print("-" * 75)
    print("  OUT-OF-SAMPLE TEST RESULTS (Untouched Test Slice)")
    print("-" * 75)
    print(f"  ROC-AUC Score:                 {auc_score:.3f}")
    print(f"  Confidence Threshold:          {conf_thresh:.0%}")
    print(f"  Break-Even Probability (p_be): {p_be:.1%}")
    print(f"  Simulated Executed Trades:     {sim_results['n_trades']} trades (max_open={max_open_trades})")
    print(f"  Simulated Win Rate:            {sim_results['win_rate']:.1%}")
    print(f"  Profit Factor (after fees):    {pf_str}")
    print(f"  Avg Return per Trade:          {sim_results['avg_return']:+.2%}")
    print(f"  Simulated Portfolio PnL:       {sim_results['total_pnl']:+.1%}")
    print(f"  Tradeable Status:              {'ACTIVE' if tradeable else 'NON-TRADEABLE (AUC < 0.54)'}")

    # Top Feature Importance
    try:
        from sklearn.inspection import permutation_importance
        perm = permutation_importance(model, X_test, y_test, scoring="roc_auc", n_repeats=3, random_state=42)
        top_feats = pd.Series(perm.importances_mean, index=feature_names).nlargest(8)
        print("\n  Top Mathematical Alpha Indicators:")
        for rank, (fname, fval) in enumerate(top_feats.items(), 1):
            print(f"    {rank}. {fname:<28} (Importance: {fval:+.4f})")
    except Exception as e:
        print(f"  [Note] Feature importance skipped: {e}")
    print("=" * 75 + "\n")

    # Save feature distribution statistics for live drift detection
    feature_stats = {
        col: {
            "mean": float(X_train[col].mean()),
            "std": float(X_train[col].std() + 1e-9),
            "min": float(X_train[col].min()),
            "max": float(X_train[col].max())
        }
        for col in feature_names
    }

    # Package the calibrated bundle with complete metadata
    bundle = {
        "model": model,
        "feature_names": feature_names,
        "confidence_threshold": conf_thresh,
        "timeframe": config["timeframe"],
        "symbols": config["symbols"],
        "version": "2.1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tradeable": tradeable,
        "split_dates": split_dates,
        "calibration_cutoff_date": calibration_cutoff_date,
        "feature_stats": feature_stats,
        "metrics": {
            "auc": round(auc_score, 4),
            "win_rate": round(sim_results["win_rate"], 4),
            "profit_factor": round(sim_results["profit_factor"], 2) if sim_results["profit_factor"] != float("inf") else "Inf",
            "avg_return": round(sim_results["avg_return"], 4),
            "n_test_trades": sim_results["n_trades"],
            "simulated_pnl": round(sim_results["total_pnl"], 4),
            "breakeven_prob": round(p_be, 4),
        }
    }
    return bundle


def main():
    parser = argparse.ArgumentParser(description="Train Binance Quantitative AI Model")
    parser.add_argument("--config", default="config.json", help="Path to config file")
    parser.add_argument("--days", type=int, default=365, help="Days of historical Binance data")
    parser.add_argument("--offline", action="store_true", help="Use locally cached data without network calls")
    parser.add_argument("--symbols", nargs="+", default=None, help="Custom list of symbols to train on")
    parser.add_argument("--timeframe", default=None, help="Candle timeframe (e.g. 1h, 4h)")

    args = parser.parse_args()
    config = load_config(args.config)

    if args.symbols:
        config["symbols"] = args.symbols
    if args.timeframe:
        config["timeframe"] = args.timeframe

    client = BinanceClient(config)
    models_dir = config["paths"]["models_dir"]
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(config["paths"]["data_dir"], exist_ok=True)

    X, y, fwd_ret, symbol_frames = prepare_dataset(
        client=client,
        symbols=config["symbols"],
        tf=config["timeframe"],
        days=args.days,
        ai_cfg=config.get("ai_model", {}),
        data_dir=config["paths"]["data_dir"],
        offline=args.offline
    )

    bundle = train_model(X, y, fwd_ret, config)

    model_path = os.path.join(models_dir, "binance_ai_model.joblib")
    joblib.dump(bundle, model_path)
    print(f"[SUCCESS] Trained AI model saved to: {model_path}")


if __name__ == "__main__":
    main()
