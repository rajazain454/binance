"""Quantitative Feature Engineering and Triple-Barrier Labeling Engine.
Computes predictive technical, momentum, volatility, and volume indicators.
Implements the institutional Triple-Barrier Method for asymmetric trade outcome labeling.
"""

import numpy as np
import pandas as pd


def compute_atr(df, period=14):
    high, low, prev_close = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_rsi(close, period=14):
    delta = close.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(period).mean()
    loss = ((-delta.where(delta < 0, 0.0))).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100.0 - (100.0 / (1.0 + rs))


def extract_features(df):
    """Generates normalized, stationary features from raw OHLCV bars."""
    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]

    feats = pd.DataFrame(index=df.index)

    # 1. Volatility & ATR
    atr14 = compute_atr(df, period=14)
    feats["atr_ratio"] = atr14 / (c + 1e-9)

    # 2. Trend Distances
    ema20 = c.ewm(span=20, adjust=False).mean()
    # 2. Trend Distances
    ema20 = c.ewm(span=20, adjust=False).mean()
    sma50 = c.rolling(50, min_periods=10).mean()
    sma200 = c.rolling(200, min_periods=20).mean()
    feats["dist_ema20"] = (c / ema20) - 1.0
    feats["dist_sma50"] = (c / (sma50 + 1e-9)) - 1.0
    feats["dist_sma200"] = (c / (sma200 + 1e-9)) - 1.0
    feats["trend_slope20"] = (ema20 - ema20.shift(5)) / (ema20.shift(5) + 1e-9)

    # 3. Momentum Oscillators
    feats["rsi_14"] = compute_rsi(c, 14) / 100.0  # Normalized [0, 1]
    feats["roc_6"] = c.pct_change(6)
    feats["roc_12"] = c.pct_change(12)
    feats["roc_24"] = c.pct_change(24)

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    feats["macd_hist_norm"] = (macd_line - signal_line) / (atr14 + 1e-9)

    # 4. Volatility Envelope (Bollinger Bands)
    bb_mid = c.rolling(20, min_periods=5).mean()
    bb_std = c.rolling(20, min_periods=5).std()
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std
    feats["bb_pos"] = (c - bb_lower) / ((bb_upper - bb_lower) + 1e-9)
    feats["bb_width"] = (bb_upper - bb_lower) / (bb_mid + 1e-9)

    # Volatility expansion/contraction
    std5 = c.pct_change().rolling(5, min_periods=3).std()
    std30 = c.pct_change().rolling(30, min_periods=10).std()
    feats["vol_ratio"] = std5 / (std30 + 1e-9)

    # 5. Volume Dynamics
    vol_mean20 = v.rolling(20, min_periods=5).mean()
    feats["vol_spike"] = v / (vol_mean20 + 1e-9)
    feats["vol_trend"] = v.rolling(5, min_periods=2).mean() / (vol_mean20 + 1e-9)

    # 6. Advanced Mathematical Volatility (Garman-Klass & Parkinson)
    log_hl = np.log(h / (l + 1e-9))
    log_co = np.log(c / (df["open"] + 1e-9))
    feats["vol_garman_klass"] = np.sqrt(np.maximum(0.0, 0.5 * (log_hl**2) - (2 * np.log(2) - 1) * (log_co**2)))
    feats["vol_parkinson"] = np.sqrt(np.maximum(0.0, (log_hl**2) / (4 * np.log(2))))

    # 7. Institutional VWAP 24h & Distance
    tp = (h + l + c) / 3.0
    vwap_24 = (tp * v).rolling(24, min_periods=5).sum() / (v.rolling(24, min_periods=5).sum() + 1e-9)
    feats["dist_vwap24"] = (c / (vwap_24 + 1e-9)) - 1.0

    # 8. Statistical Distribution Anomaly (Z-Score & Skewness)
    ret1 = c.pct_change()
    feats["ret_zscore_24"] = (ret1 - ret1.rolling(24, min_periods=5).mean()) / (ret1.rolling(24, min_periods=5).std() + 1e-9)
    feats["ret_skew_30"] = ret1.rolling(30, min_periods=10).skew().fillna(0.0)

    # 9. Zero-Lag DEMA (Double Exponential Moving Average)
    dema20 = 2 * ema20 - ema20.ewm(span=20, adjust=False).mean()
    feats["dist_dema20"] = (c / (dema20 + 1e-9)) - 1.0

    # 10. Trend Strength & Directional Movement (ADX Proxy)
    up_move = h - h.shift(1)
    down_move = l.shift(1) - l
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_di = pd.Series(plus_dm, index=df.index).rolling(14, min_periods=5).mean() / (atr14 + 1e-9)
    minus_di = pd.Series(minus_dm, index=df.index).rolling(14, min_periods=5).mean() / (atr14 + 1e-9)
    dx = (plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-9)
    feats["adx_trend_strength"] = dx.rolling(14, min_periods=5).mean()

    # 11. Chaikin Spread Expansion Volatility
    hl_ema = (h - l).ewm(span=10, adjust=False).mean()
    feats["chaikin_vol"] = (hl_ema - hl_ema.shift(10)) / (hl_ema.shift(10) + 1e-9)

    # 12. Candlestick Price Action
    bar_range = (h - l) + 1e-9
    feats["body_ratio"] = (c - df["open"]) / bar_range
    feats["upper_shadow"] = (h - np.maximum(df["open"], c)) / bar_range
    feats["lower_shadow"] = (np.minimum(df["open"], c) - l) / bar_range

    # 13. Multi-Timeframe (MTF) Macro Confluence (4h Resampled)
    has_dt = isinstance(df.index, pd.DatetimeIndex)
    if not has_dt and "timestamp" in df.columns:
        try:
            resample_source = df.set_index(pd.to_datetime(df["timestamp"]))
            has_dt = True
        except Exception:
            resample_source = df
    else:
        resample_source = df

    if has_dt and isinstance(resample_source.index, pd.DatetimeIndex):
        try:
            df_4h = resample_source.resample("4h").agg({
                "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
            }).dropna()
        except Exception:
            df_4h = pd.DataFrame()
    else:
        df_4h = pd.DataFrame()

    if len(df_4h) >= 20:
        c_4h = df_4h["close"]
        ema50_4h = c_4h.ewm(span=50, adjust=False).mean()
        rsi_4h = compute_rsi(c_4h, 14) / 100.0
        slope_4h = (ema50_4h - ema50_4h.shift(3)) / (ema50_4h.shift(3) + 1e-9)

        # Shift by 1 bar to strictly use closed 4h candles (zero look-ahead bias)
        ema50_aligned = ema50_4h.shift(1).reindex(df.index, method="ffill")
        rsi_aligned = rsi_4h.shift(1).reindex(df.index, method="ffill")
        slope_aligned = slope_4h.shift(1).reindex(df.index, method="ffill")

        feats["mtf_4h_dist_ema50"] = (c / (ema50_aligned + 1e-9)) - 1.0
        feats["mtf_4h_rsi"] = rsi_aligned.fillna(0.50)
        feats["mtf_4h_slope"] = slope_aligned.fillna(0.0)
        feats["mtf_4h_bullish"] = ((c >= ema50_aligned * 0.995) & (rsi_aligned >= 0.48)).astype(float).fillna(1.0)
    else:
        feats["mtf_4h_dist_ema50"] = 0.0
        feats["mtf_4h_rsi"] = 0.50
        feats["mtf_4h_slope"] = 0.0
        feats["mtf_4h_bullish"] = 1.0

    return feats.dropna()


def evaluate_macro_confluence(df_1h, min_rsi=0.48):
    """
    Evaluates the live 4h macro trend from 1h data.
    Returns: (is_bullish: bool, details: dict)
    """
    has_dt = isinstance(df_1h.index, pd.DatetimeIndex)
    if not has_dt and "timestamp" in df_1h.columns:
        try:
            df_source = df_1h.set_index(pd.to_datetime(df_1h["timestamp"]))
            has_dt = True
        except Exception:
            df_source = df_1h
    else:
        df_source = df_1h

    if not has_dt or not isinstance(df_source.index, pd.DatetimeIndex):
        return True, {"status": "NO_DATETIME_INDEX", "ema50": 0.0, "rsi": 50.0, "is_bullish": True}

    try:
        df_4h = df_source.resample("4h").agg({
            "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
        }).dropna()
    except Exception:
        return True, {"status": "RESAMPLE_ERROR", "ema50": 0.0, "rsi": 50.0, "is_bullish": True}

    if len(df_4h) < 20:
        return True, {"status": "INSUFFICIENT_DATA", "ema50": 0.0, "rsi": 50.0, "is_bullish": True}

    c_4h = df_4h["close"]
    ema50_4h = float(c_4h.ewm(span=50, adjust=False).mean().iloc[-1])
    rsi_4h = float(compute_rsi(c_4h, 14).iloc[-1])
    curr_c = float(c_4h.iloc[-1])

    # Bullish confluence: 4h price above or at 4h EMA50, and 4h RSI >= 48
    req_rsi = min_rsi * 100.0 if min_rsi < 1.0 else min_rsi
    is_bullish = bool((curr_c >= ema50_4h * 0.995) and (rsi_4h >= req_rsi))

    details = {
        "status": "BULLISH" if is_bullish else "BEARISH",
        "curr_price": round(float(curr_c), 4),
        "ema50_4h": round(float(ema50_4h), 4),
        "rsi_4h": round(float(rsi_4h), 1),
        "is_bullish": bool(is_bullish)
    }
    return is_bullish, details


def label_triple_barrier(df, horizon_bars=12, tp_atr_mult=2.2, sl_atr_mult=1.4):
    """
    Labels each bar using Triple Barrier Method:
    1 = Upper barrier (Take Profit) reached first within horizon.
    0 = Lower barrier (Stop Loss) reached first or timeout.
    """
    c = df["close"]
    h = df["high"]
    l = df["low"]
    atr = compute_atr(df, period=14)

    n = len(df)
    labels = np.zeros(n, dtype=int)
    returns = np.zeros(n, dtype=float)

    for i in range(n - horizon_bars):
        entry_price = c.iloc[i]
        curr_atr = atr.iloc[i]
        if np.isnan(curr_atr) or curr_atr <= 0:
            continue

        tp_barrier = entry_price + (tp_atr_mult * curr_atr)
        sl_barrier = entry_price - (sl_atr_mult * curr_atr)

        sub_h = h.iloc[i + 1: i + 1 + horizon_bars].values
        sub_l = l.iloc[i + 1: i + 1 + horizon_bars].values
        sub_c = c.iloc[i + 1: i + 1 + horizon_bars].values

        outcome = 0
        final_ret = 0.0

        for t in range(len(sub_h)):
            hit_tp = sub_h[t] >= tp_barrier
            hit_sl = sub_l[t] <= sl_barrier

            if hit_tp and not hit_sl:
                outcome = 1
                final_ret = (tp_barrier / entry_price) - 1.0
                break
            elif hit_sl and not hit_tp:
                outcome = 0
                final_ret = (sl_barrier / entry_price) - 1.0
                break
            elif hit_tp and hit_sl:
                # Worst case assumption: hit stop loss first
                outcome = 0
                final_ret = (sl_barrier / entry_price) - 1.0
                break

        # If timeout reached without touching either barrier
        if outcome == 0 and final_ret == 0.0:
            exit_price = sub_c[-1]
            final_ret = (exit_price / entry_price) - 1.0
            outcome = 1 if final_ret > 0.01 else 0

        labels[i] = outcome
        returns[i] = final_ret

    label_series = pd.Series(labels, index=df.index, name="target")
    ret_series = pd.Series(returns, index=df.index, name="forward_ret")
    return label_series, ret_series
