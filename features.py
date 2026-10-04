"""Quantitative Feature Engineering and Triple-Barrier Labeling Engine.
Computes predictive technical, momentum, volatility, volume, and multi-indicator strategy signals.
Implements Bollinger Squeeze (TTM), Stochastic RSI, Chaikin Money Flow, SuperTrend,
and the institutional Triple-Barrier Method for asymmetric trade outcome labeling.
"""

import numpy as np
import pandas as pd


def compute_atr(df, period=14):
    """Computes Average True Range over `period`."""
    high, low, prev_close = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=3).mean().bfill()


def compute_rsi(close, period=14):
    """Computes standard Relative Strength Index [0, 100]."""
    delta = close.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(period, min_periods=3).mean()
    loss = ((-delta.where(delta < 0, 0.0))).rolling(period, min_periods=3).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.bfill().fillna(50.0)


def compute_bollinger_bands(close, period=20, std_dev=2.0):
    """Computes Bollinger Bands, %B position, and Bandwidth."""
    mid = close.rolling(period, min_periods=5).mean()
    std = close.rolling(period, min_periods=5).std().fillna(0.0)
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    pct_b = (close - lower) / ((upper - lower) + 1e-9)
    width = (upper - lower) / (mid + 1e-9)
    return upper, lower, mid, pct_b, width


def compute_keltner_channels(df, period=20, atr_mult=1.5):
    """Computes Keltner Channels based on EMA and ATR."""
    mid = df["close"].ewm(span=period, adjust=False).mean()
    atr = compute_atr(df, period=14)
    upper = mid + atr_mult * atr
    lower = mid - atr_mult * atr
    return upper, lower, mid


def compute_bollinger_squeeze(df):
    """
    Computes John Carter's TTM Squeeze indicator:
    - Squeeze ON: Bollinger Bands contract inside Keltner Channels (volatility coiled).
    - Squeeze FIRED: Bollinger Bands expand outside Keltner Channels (explosive momentum).
    - Squeeze Momentum: Momentum histogram measuring direction of breakout.
    """
    bb_u, bb_l, bb_m, _, _ = compute_bollinger_bands(df["close"], 20, 2.0)
    kc_u, kc_l, kc_m = compute_keltner_channels(df, 20, 1.5)
    atr = compute_atr(df, 14)

    squeeze_on = ((bb_l > kc_l) & (bb_u < kc_u)).astype(float)
    squeeze_fired = ((bb_u >= kc_u) | (bb_l <= kc_l)).astype(float)

    hh = df["high"].rolling(20, min_periods=5).max()
    ll = df["low"].rolling(20, min_periods=5).min()
    mean_val = (hh + ll + kc_m) / 3.0
    mom = (df["close"] - mean_val) / (atr + 1e-9)
    return squeeze_on, squeeze_fired, mom.fillna(0.0)


def compute_stoch_rsi(close, rsi_period=14, stoch_period=14, k_period=3, d_period=3):
    """
    Computes Stochastic RSI (%K and %D lines).
    Pinpoints oversold bounce opportunities when K crosses above D below 0.30.
    """
    rsi = compute_rsi(close, period=rsi_period)
    rsi_low = rsi.rolling(stoch_period, min_periods=3).min()
    rsi_high = rsi.rolling(stoch_period, min_periods=3).max()
    stoch = (rsi - rsi_low) / ((rsi_high - rsi_low) + 1e-9)
    stoch = stoch.clip(0.0, 1.0)
    k = stoch.rolling(k_period, min_periods=1).mean().fillna(0.5)
    d = k.rolling(d_period, min_periods=1).mean().fillna(0.5)
    bull_cross = ((k > d) & (k.shift(1) <= d.shift(1)) & (k < 0.35)).astype(float).fillna(0.0)
    return k, d, bull_cross


def compute_cmf(df, period=20):
    """
    Chaikin Money Flow (CMF): Measures institutional accumulation/distribution.
    Values > +0.05 confirm smart money buying inflow.
    """
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    hl_range = (h - l) + 1e-9
    mf_mult = ((c - l) - (h - c)) / hl_range
    mf_vol = mf_mult * v
    cmf = mf_vol.rolling(period, min_periods=5).sum() / (v.rolling(period, min_periods=5).sum() + 1e-9)
    return cmf.clip(-1.0, 1.0).fillna(0.0)


def compute_obv_trend(df, slope_period=10):
    """Computes On-Balance-Volume (OBV) trend slope normalized by volume."""
    c, v = df["close"], df["volume"]
    direction = np.sign(c.diff()).fillna(0.0)
    obv = (direction * v).cumsum()
    v_mean = v.rolling(slope_period, min_periods=3).mean() + 1e-9
    obv_slope = (obv - obv.shift(slope_period)) / (v_mean * slope_period)
    return obv_slope.clip(-5.0, 5.0).fillna(0.0)


def compute_supertrend(df, period=10, multiplier=3.0):
    """
    SuperTrend: Volatility trailing stop trend indicator.
    Returns: (is_bullish: Series [1.0/0.0], dist_to_band: Series)
    """
    c, h, l = df["close"], df["high"], df["low"]
    atr = compute_atr(df, period=period)
    hl2 = (h + l) / 2.0
    upper_basic = hl2 + (multiplier * atr)
    lower_basic = hl2 - (multiplier * atr)

    n = len(df)
    final_ub = np.zeros(n)
    final_lb = np.zeros(n)
    direction = np.zeros(n)

    c_vals = c.values
    ub_vals = upper_basic.values
    lb_vals = lower_basic.values

    if n > 0:
        final_ub[0] = ub_vals[0]
        final_lb[0] = lb_vals[0]
        direction[0] = 1.0 if c_vals[0] >= lb_vals[0] else 0.0

        for i in range(1, n):
            # Lower band
            if lb_vals[i] > final_lb[i - 1] or c_vals[i - 1] < final_lb[i - 1]:
                final_lb[i] = lb_vals[i]
            else:
                final_lb[i] = final_lb[i - 1]

            # Upper band
            if ub_vals[i] < final_ub[i - 1] or c_vals[i - 1] > final_ub[i - 1]:
                final_ub[i] = ub_vals[i]
            else:
                final_ub[i] = final_ub[i - 1]

            # Direction switch
            if direction[i - 1] == 1.0:
                direction[i] = 0.0 if c_vals[i] < final_lb[i] else 1.0
            else:
                direction[i] = 1.0 if c_vals[i] > final_ub[i] else 0.0

    dir_series = pd.Series(direction, index=df.index, name="supertrend_bullish")
    dist_band = (c - pd.Series(final_lb, index=df.index)) / (atr + 1e-9)
    return dir_series, dist_band.fillna(0.0)


def compute_cci(df, period=20):
    """Commodity Channel Index (CCI) normalized around [-3, 3]."""
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    sma = tp.rolling(period, min_periods=5).mean()
    mad = (tp - sma).abs().rolling(period, min_periods=5).mean()
    cci = (tp - sma) / (0.015 * mad + 1e-9)
    return (cci / 100.0).clip(-3.0, 3.0).fillna(0.0)


def evaluate_strategy_setups(df):
    """
    Evaluates 4 high-probability quantitative setups:
    1. Bollinger Squeeze Breakout: Squeeze fired with volume expansion.
    2. StochRSI Mean Reversion: StochRSI bullish crossover from oversold near lower BB.
    3. SuperTrend Trend Pullback: Bullish regime with price retesting EMA20/BB Mid.
    4. Institutional Volume Flow Breakout: CMF accumulation with positive OBV slope.
    Returns: DataFrame with setup scores [0, 1] and composite score [0, 1].
    """
    c, v = df["close"], df["volume"]
    atr = compute_atr(df, 14)

    # 1. Squeeze
    _, sq_fired, sq_mom = compute_bollinger_squeeze(df)
    _, _, _, pct_b, _ = compute_bollinger_bands(c, 20, 2.0)
    vol_mean = v.rolling(20, min_periods=5).mean() + 1e-9
    vol_spike = v / vol_mean
    strat_squeeze = ((sq_fired > 0.5) & (pct_b > 0.50) & (vol_spike > 1.15) & (sq_mom > 0)).astype(float)

    # 2. StochRSI Bounce
    rsi14 = compute_rsi(c, 14)
    k, d, stoch_cross = compute_stoch_rsi(c, 14, 14, 3, 3)
    strat_stoch = ((stoch_cross > 0.5) & (rsi14 < 48.0) & (pct_b < 0.45)).astype(float)

    # 3. SuperTrend Pullback
    st_bull, _ = compute_supertrend(df, 10, 3.0)
    ema20 = c.ewm(span=20, adjust=False).mean()
    dist_ema20 = (c / (ema20 + 1e-9)) - 1.0
    rsi7 = compute_rsi(c, 7)
    strat_pullback = ((st_bull > 0.5) & (dist_ema20.abs() < 0.018) & (rsi7 > 40.0) & (rsi7 < 65.0)).astype(float)

    # 4. Volume Flow Breakout
    cmf = compute_cmf(df, 20)
    obv_s = compute_obv_trend(df, 10)
    roc6 = c.pct_change(6).fillna(0.0)
    strat_volume = ((cmf > 0.06) & (obv_s > 0.10) & (roc6 > 0.0)).astype(float)

    # Composite Strategy Score (Continuous 0.0 to 1.0)
    comp_score = (
        (strat_squeeze * 0.30) +
        (strat_stoch * 0.30) +
        (strat_pullback * 0.25) +
        (strat_volume * 0.25) +
        (st_bull * 0.15) +
        (np.clip(cmf, 0, 0.3) / 0.3 * 0.15) +
        (np.clip(k, 0, 1) * 0.10)
    ).clip(0.0, 1.0)

    res = pd.DataFrame(index=df.index)
    res["strat_squeeze_breakout"] = strat_squeeze
    res["strat_stoch_rsi_bounce"] = strat_stoch
    res["strat_supertrend_pullback"] = strat_pullback
    res["strat_volume_flow_breakout"] = strat_volume
    res["composite_strategy_score"] = comp_score
    return res


def detect_active_strategies(df):
    """
    Returns live strategy identification for trading decision & alert generation.
    Returns: (active_setups: list[str], strategy_score: float, summary_text: str)
    """
    if len(df) < 30:
        return [], 50.0, "Insufficient bars"

    c = df["close"]
    rsi14 = float(compute_rsi(c, 14).iloc[-1])
    rsi7 = float(compute_rsi(c, 7).iloc[-1])
    k, d, stoch_cross = compute_stoch_rsi(c, 14, 14, 3, 3)
    k_val, d_val = float(k.iloc[-1]), float(d.iloc[-1])
    is_stoch_cross = bool(stoch_cross.iloc[-1] > 0.5 or (k_val > d_val and k_val < 0.35))

    _, _, _, pct_b, _ = compute_bollinger_bands(c, 20, 2.0)
    pct_b_val = float(pct_b.iloc[-1])

    sq_on, sq_fired, _ = compute_bollinger_squeeze(df)
    is_sq_fired = bool(sq_fired.iloc[-1] > 0.5)

    st_bull, _ = compute_supertrend(df, 10, 3.0)
    is_supertrend_bull = bool(st_bull.iloc[-1] > 0.5)

    cmf = float(compute_cmf(df, 20).iloc[-1])
    obv_s = float(compute_obv_trend(df, 10).iloc[-1])

    strat_df = evaluate_strategy_setups(df)
    strat_score = float(strat_df["composite_strategy_score"].iloc[-1]) * 100.0

    setups = []
    if is_sq_fired and pct_b_val > 0.50:
        setups.append("Bollinger Squeeze Breakout")
    if is_stoch_cross:
        setups.append("StochRSI Oversold Bullish Cross")
    if is_supertrend_bull and abs(pct_b_val - 0.50) < 0.25:
        setups.append("SuperTrend Dip Pullback")
    if cmf > 0.06 and obv_s > 0.10:
        setups.append("Institutional Volume Flow Accumulation")

    summary_text = (
        f"RSI(14): {rsi14:.1f} | StochRSI: {k_val*100:.0f}/{d_val*100:.0f} | "
        f"BB %B: {pct_b_val:.2f} | CMF: {cmf:+.2f} | "
        f"SuperTrend: {'BULLISH' if is_supertrend_bull else 'BEARISH'}"
    )

    return setups, round(strat_score, 1), summary_text


def extract_features(df):
    """Generates normalized, stationary, predictive quantitative features from raw OHLCV bars."""
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
    sma50 = c.rolling(50, min_periods=10).mean()
    sma200 = c.rolling(200, min_periods=20).mean()
    feats["dist_ema20"] = (c / (ema20 + 1e-9)) - 1.0
    feats["dist_sma50"] = (c / (sma50 + 1e-9)) - 1.0
    feats["dist_sma200"] = (c / (sma200 + 1e-9)) - 1.0
    feats["trend_slope20"] = (ema20 - ema20.shift(5)) / (ema20.shift(5) + 1e-9)

    # 3. Momentum Oscillators & Multi-Speed RSI
    feats["rsi_14"] = compute_rsi(c, 14) / 100.0  # Normalized [0, 1]
    feats["rsi_7"] = compute_rsi(c, 7) / 100.0
    rsi_series = compute_rsi(c, 14)
    feats["rsi_delta_3"] = (rsi_series - rsi_series.shift(3)) / 100.0
    feats["roc_6"] = c.pct_change(6)
    feats["roc_12"] = c.pct_change(12)
    feats["roc_24"] = c.pct_change(24)

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    feats["macd_hist_norm"] = (macd_line - signal_line) / (atr14 + 1e-9)

    # 4. Volatility Envelope & Bollinger Squeeze (TTM)
    bb_u, bb_l, bb_m, pct_b, bb_width = compute_bollinger_bands(c, 20, 2.0)
    feats["bb_pos"] = pct_b
    feats["bb_width"] = bb_width

    sq_on, sq_fired, sq_mom = compute_bollinger_squeeze(df)
    feats["bb_squeeze_on"] = sq_on
    feats["bb_squeeze_fired"] = sq_fired
    feats["bb_squeeze_mom"] = sq_mom

    # Volatility expansion/contraction
    std5 = c.pct_change().rolling(5, min_periods=3).std()
    std30 = c.pct_change().rolling(30, min_periods=10).std()
    feats["vol_ratio"] = std5 / (std30 + 1e-9)

    # 5. Stochastic RSI
    stoch_k, stoch_d, stoch_cross = compute_stoch_rsi(c, 14, 14, 3, 3)
    feats["stoch_rsi_k"] = stoch_k
    feats["stoch_rsi_d"] = stoch_d
    feats["stoch_rsi_cross_bull"] = stoch_cross

    # 6. Volume Dynamics & Institutional Flow (CMF & OBV)
    vol_mean20 = v.rolling(20, min_periods=5).mean()
    feats["vol_spike"] = v / (vol_mean20 + 1e-9)
    feats["vol_trend"] = v.rolling(5, min_periods=2).mean() / (vol_mean20 + 1e-9)
    feats["cmf_20"] = compute_cmf(df, 20)
    feats["obv_slope_10"] = compute_obv_trend(df, 10)

    # 7. Advanced Mathematical Volatility (Garman-Klass & Parkinson)
    log_hl = np.log(h / (l + 1e-9))
    log_co = np.log(c / (df["open"] + 1e-9))
    feats["vol_garman_klass"] = np.sqrt(np.maximum(0.0, 0.5 * (log_hl**2) - (2 * np.log(2) - 1) * (log_co**2)))
    feats["vol_parkinson"] = np.sqrt(np.maximum(0.0, (log_hl**2) / (4 * np.log(2))))

    # 8. Institutional VWAP 24h & Distance
    tp = (h + l + c) / 3.0
    vwap_24 = (tp * v).rolling(24, min_periods=5).sum() / (v.rolling(24, min_periods=5).sum() + 1e-9)
    feats["dist_vwap24"] = (c / (vwap_24 + 1e-9)) - 1.0

    # 9. SuperTrend & Cyclical CCI
    st_bull, st_dist = compute_supertrend(df, 10, 3.0)
    feats["supertrend_bullish"] = st_bull
    feats["dist_supertrend"] = st_dist
    feats["cci_20"] = compute_cci(df, 20)

    # 10. Statistical Distribution Anomaly (Z-Score & Skewness)
    ret1 = c.pct_change()
    feats["ret_zscore_24"] = (ret1 - ret1.rolling(24, min_periods=5).mean()) / (ret1.rolling(24, min_periods=5).std() + 1e-9)
    feats["ret_skew_30"] = ret1.rolling(30, min_periods=10).skew().fillna(0.0)

    # 11. Zero-Lag DEMA & EMA Ribbon
    dema20 = 2 * ema20 - ema20.ewm(span=20, adjust=False).mean()
    feats["dist_dema20"] = (c / (dema20 + 1e-9)) - 1.0
    ema9 = c.ewm(span=9, adjust=False).mean()
    ribbon_score = ((ema9 > ema20).astype(float) + (ema20 > sma50).astype(float) + (sma50 > sma200).astype(float)) / 3.0
    feats["ema_ribbon_bull"] = ribbon_score

    # 12. Trend Strength & Directional Movement (ADX Proxy)
    up_move = h - h.shift(1)
    down_move = l.shift(1) - l
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_di = pd.Series(plus_dm, index=df.index).rolling(14, min_periods=5).mean() / (atr14 + 1e-9)
    minus_di = pd.Series(minus_dm, index=df.index).rolling(14, min_periods=5).mean() / (atr14 + 1e-9)
    dx = (plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-9)
    feats["adx_trend_strength"] = dx.rolling(14, min_periods=5).mean()

    # 13. Chaikin Spread Expansion Volatility
    hl_ema = (h - l).ewm(span=10, adjust=False).mean()
    feats["chaikin_vol"] = (hl_ema - hl_ema.shift(10)) / (hl_ema.shift(10) + 1e-9)

    # 14. Candlestick Price Action
    bar_range = (h - l) + 1e-9
    feats["body_ratio"] = (c - df["open"]) / bar_range
    feats["upper_shadow"] = (h - np.maximum(df["open"], c)) / bar_range
    feats["lower_shadow"] = (np.minimum(df["open"], c) - l) / bar_range

    # 15. Concrete Strategy Setups & Composite Confluence Alpha
    strat_df = evaluate_strategy_setups(df)
    feats["strat_squeeze_breakout"] = strat_df["strat_squeeze_breakout"]
    feats["strat_stoch_rsi_bounce"] = strat_df["strat_stoch_rsi_bounce"]
    feats["strat_supertrend_pullback"] = strat_df["strat_supertrend_pullback"]
    feats["strat_volume_flow_breakout"] = strat_df["strat_volume_flow_breakout"]
    feats["composite_strategy_score"] = strat_df["composite_strategy_score"]

    # 16. Multi-Timeframe (MTF) Macro Confluence (4h Resampled)
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

    # Ensure completely clean data with zero NaNs or infinities
    feats = feats.bfill().fillna(0.0)
    feats = feats.replace([np.inf, -np.inf], 0.0)
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
