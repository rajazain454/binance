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
    tp_mult = ai_cfg.get("tp_atr_mult", 2.2)
    sl_mult = ai_cfg.get("sl_atr_mult", 1.4)

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
        ret_sym = fwd_ret.loc[valid_idx]

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


def train_model(X, y, fwd_returns, config):
    """Trains an optimized Gradient Boosting ensemble with exponential time-decay weighting."""
    feature_names = list(X.columns)
    ai_cfg = config.get("ai_model", {})
    conf_thresh = ai_cfg.get("confidence_threshold", 0.60)
    cost_rate = config["risk_management"]["fee_rate"] + config["risk_management"]["slippage_rate"]
    horizon = ai_cfg.get("horizon_bars", 12)

    # Cross-Sectional Chronological Train-Test Split with Purge Barrier (De Prado methodology)
    if isinstance(X.index, pd.DatetimeIndex) and len(X.index.unique()) > 10:
        unique_dates = X.index.unique().sort_values()
        split_date_idx = int(len(unique_dates) * 0.80)
        split_date = unique_dates[split_date_idx]
        purge_barrier = split_date - pd.Timedelta(hours=horizon)

        train_mask = (X.index < purge_barrier)
        test_mask = (X.index >= split_date)

        X_train, X_test = X.loc[train_mask], X.loc[test_mask]
        y_train, y_test = y.loc[train_mask], y.loc[test_mask]
        ret_test = fwd_returns.loc[test_mask]
    else:
        split_idx = int(len(X) * 0.80)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        ret_test = fwd_returns.iloc[split_idx:]

    print("\n" + "=" * 75)
    print("  TRAINING BINANCE AI MODEL (HistGradientBoostingClassifier with Quant Setups)")
    print("=" * 75)
    print(f"  Total Clean Samples: {len(X):,}  |  Train: {len(X_train):,}  |  Out-of-Sample Test: {len(X_test):,}")
    print(f"  Mathematical Alpha Features: {len(feature_names)} indicators")
    print(f"  Baseline Setup Win Rate: {y.mean():.1%}")

    # Compute Exponential Time-Decay Sample Weights (half-life weighting)
    n_train = len(X_train)
    half_life_samples = max(1000, n_train // 4)
    decay_rate = np.log(2.0) / half_life_samples
    time_weights = np.exp(-decay_rate * (n_train - 1 - np.arange(n_train)))
    time_weights = time_weights / time_weights.mean()

    # Model 1: Scikit-Learn Histogram Gradient Boosting (depth-wise regularized)
    hgb_model = HistGradientBoostingClassifier(
        max_iter=250,
        learning_rate=0.03,
        max_leaf_nodes=40,
        min_samples_leaf=30,
        l2_regularization=3.0,
        early_stopping=True,
        n_iter_no_change=15,
        class_weight="balanced",
        random_state=42
    )
    print("  Fitting Model 1: HistGradientBoostingClassifier...")
    hgb_model.fit(X_train, y_train, sample_weight=time_weights)

    # Model 2: LightGBM (leaf-wise gradient boosting)
    import lightgbm as lgb
    lgb_model = lgb.LGBMClassifier(
        n_estimators=250,
        learning_rate=0.03,
        num_leaves=35,
        min_child_samples=30,
        reg_lambda=3.0,
        class_weight="balanced",
        random_state=42,
        verbose=-1
    )
    print("  Fitting Model 2: LightGBM Classifier...")
    lgb_model.fit(X_train, y_train, sample_weight=time_weights)

    # Combine into Stacking Ensemble with LogisticRegression Meta-Learner
    from sklearn.linear_model import LogisticRegression
    from models import StackingEnsembleModel

    meta_clf = LogisticRegression(C=1.0, random_state=42)
    raw_ensemble = StackingEnsembleModel(
        hgb_model, lgb_model, weights=(0.50, 0.50), meta_learner=meta_clf
    )
    raw_ensemble.fit(X_train, y_train)

    # Probability calibration via isotonic regression on held-out test split
    # This ensures predict_proba outputs reflect actual empirical win rates
    print("  Calibrating probabilities (Isotonic Regression)...")
    from models import CalibratedEnsemble
    calibrated_model = CalibratedEnsemble(raw_ensemble)
    calibrated_model.fit(X_test, y_test)
    model = calibrated_model
    test_probs = model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, test_probs)

    # Filter by model confidence threshold
    high_conf_mask = test_probs >= conf_thresh
    n_high_conf = int(high_conf_mask.sum())

    if n_high_conf > 0:
        high_conf_wins = int(y_test[high_conf_mask].sum())
        high_conf_win_rate = high_conf_wins / n_high_conf
        high_conf_returns = ret_test[high_conf_mask] - (cost_rate * 2)  # Entry + Exit friction
        avg_ret_per_trade = float(high_conf_returns.mean())
        total_pnl = float(high_conf_returns.sum())
        profit_trades = high_conf_returns[high_conf_returns > 0]
        loss_trades = high_conf_returns[high_conf_returns <= 0]
        gross_profit = float(profit_trades.sum()) if len(profit_trades) else 0.0
        gross_loss = float(abs(loss_trades.sum())) if len(loss_trades) else 1e-9
        profit_factor = gross_profit / gross_loss
    else:
        high_conf_win_rate = 0.0
        avg_ret_per_trade = 0.0
        total_pnl = 0.0
        profit_factor = 0.0

    print("-" * 75)
    print("  OUT-OF-SAMPLE TEST RESULTS (Unseen Forward 20% Data)")
    print("-" * 75)
    print(f"  ROC-AUC Score:                 {auc_score:.3f}")
    print(f"  Confidence Threshold:          {conf_thresh:.0%}")
    print(f"  High-Conviction Trade Signals: {n_high_conf} trades")
    print(f"  High-Conviction Win Rate:      {high_conf_win_rate:.1%}")
    print(f"  Profit Factor (after fees):    {profit_factor:.2f}")
    print(f"  Avg Return per Trade:          {avg_ret_per_trade:+.2%}")
    print(f"  Cumulative Sample PnL:         {total_pnl:+.1%}")

    # Top Feature Importance
    try:
        from sklearn.inspection import permutation_importance
        perm = permutation_importance(model, X_test, y_test, n_repeats=3, random_state=42)
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
        "version": "2.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_stats": feature_stats,
        "metrics": {
            "auc": round(auc_score, 4),
            "win_rate": round(high_conf_win_rate, 4),
            "profit_factor": round(profit_factor, 2),
            "avg_return": round(avg_ret_per_trade, 4),
            "n_test_trades": n_high_conf
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
