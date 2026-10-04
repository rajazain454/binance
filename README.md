# ⚡ Binance Quantitative AI Trading Bot (Institutional V4.0)

A battle-tested, high-performance quantitative machine learning trading bot engineered specifically for **Binance Spot** trading. Powered by **174,960 historical candles (1 full year across 20 liquid crypto assets)**, a **49-indicator mathematical alpha engine**, self-formulating strategy setups (Bollinger Squeeze, StochRSI, CMF, SuperTrend), a **Multi-Model Stacking Ensemble (`LightGBM` + `HistGradientBoosting`)**, **Parallel Multi-Threaded Scanning (`ThreadPoolExecutor`)**, **Relative Strength vs BTC (RS-BTC Alpha Decoupling Engine)**, **Cross-Asset Correlation Risk Filtering**, **Chandelier Volatility Trailing Stops**, **Visual Progress Gauges**, and **24/7 serverless cloud automation via GitHub Actions & external cron trigger**.

---

## 🎯 Core Strategy & Quantitative Alpha Suite

### 1. Multi-Model Stacking Ensemble (`LightGBM` + `HistGradientBoosting`)
Rather than relying on a single classifier with potential blind spots during macro regime shifts, the bot deploys a dual-tree stacking ensemble trained on **174,960 historical hourly bars**:
* **`HistGradientBoostingClassifier`**: Optimizes non-linear tabular interactions with monotonic constraints.
* **`LightGBM (LGBMClassifier)`**: Deploys leaf-wise tree growth with gradient-based one-side sampling (GOSS) for superior edge capture in asymmetric price distributions.
* **Calibrated Blending**: Combines probability predictions to smooth outlier confidence spikes, achieving an out-of-sample win rate of **58.8%** and a profit factor of **2.38**.

---

### 2. Expanded 20-Coin Universe & Parallel Multi-Threaded Scanner
* **20 High-Liquidity Binance Assets**:
  `BTC`, `ETH`, `SOL`, `BNB`, `XRP`, `DOGE`, `ADA`, `AVAX`, `LINK`, `NEAR`, `SUI`, `APT`, `INJ`, `TIA`, `RENDER`, `FET`, `SEI`, `ARB`, `OP`, `DOT`.
* **Parallel Execution (`ThreadPoolExecutor`)**: Concurrently fetches candles across all 20 pairs using 6 worker threads, reducing total market scan latency from ~30 seconds down to **under 2.5 seconds**.

---

### 3. Relative Strength vs BTC (RS-BTC Alpha Decoupling Engine)
When Bitcoin consolidates or trends downward on the 4H timeframe, traditional bots block all altcoins (`4H BLOCKED`). The RS-BTC Engine evaluates:
$$RS_{\text{ratio}} = \frac{\text{Close}_{\text{Alt}}}{\text{Close}_{\text{BTC}}}$$
* **Alpha Leader Criteria**:
  1. $RS_{\text{ratio}}$ is trending above its 20-period exponential moving average ($EMA_{20}$).
  2. 24-hour alpha spread ($\text{Return}_{\text{Alt}} - \text{Return}_{\text{BTC}}$) is $\ge +1.5\%$.
  3. Altcoin absolute 24-hour return is positive ($> 0\%$).
* When confirmed, the coin is unlocked as an **`RS ALPHA LEADER`**, allowing high-conviction decoupled momentum trades even during broader market chop.

---

### 4. Self-Formulating Strategy Setups
The bot synthesizes multiple technical tools to classify and rank institutional setups in real-time:
* **`Bollinger Squeeze Breakout` (John Carter's TTM Squeeze)**: Identifies volatility contraction when Bollinger Bands ($SMA_{20} \pm 2\sigma$) compress inside Keltner Channels ($EMA_{20} \pm 1.5 ATR$). Triggers upon **Squeeze Fire** when volatility explodes outward with volume confirmation.
* **`StochRSI Oversold Bullish Cross`**: Measures Stochastic RSI over 14 periods. Catches high-conviction momentum reversals when $\%K$ crosses above $\%D$ in extreme oversold territory ($< 0.35$) while price retests lower Bollinger Bands ($\%B < 0.45$).
* **`SuperTrend Dip Pullback`**: Operates in a confirmed bullish regime using the dynamic SuperTrend trailing stop $(H+L)/2 \pm 3.0 \times ATR_{10}$, buying the dip as price pulls back into the 20-period EMA / Bollinger Mid-band.
* **`Institutional Volume Flow Accumulation`**: Employs **Chaikin Money Flow (CMF)** and **On-Balance Volume (OBV)** slope to detect smart money accumulating ahead of price breakouts.
* **`Composite Strategy Score` (0 to 100)**: Continuously scores market opportunity quality across all technical dimensions.

---

### 5. Multi-Slot Portfolio Allocation & Risk Parity
* **3 Concurrent Slots**: Configured for 3 simultaneous diversified positions with dynamic risk parity sizing.
* **Binance $5 minNotional Guard**: Guarantees all allocated orders satisfy Binance exchange minimum order requirements ($5.00+ USDT).
* **Cross-Asset Correlation Filter**: Computes rolling 30-day Pearson return correlation between candidate coins and existing open positions. If correlation exceeds **0.75**, the trade is marked `CORR BLOCKED` to enforce genuine sector diversification.

---

### 6. Chandelier Dynamic Volatility Trailing Stops & Visual Progress Gauges
* **TP1 ($1.2\times ATR$)**: Automatically banks partial profits.
* **Breakeven Stop Ratchet**: Immediately raises Stop-Loss to Entry Price + exchange fee buffer, making the runner completely risk-free.
* **Chandelier Trailing Exit**: Dynamic trailing stop ratchets behind peak highs:
  $$\text{Trailing Stop} = \text{Highest Peak} - 2.2 \times ATR$$
* **Visual Trade Progress Gauges in Discord**:
  ```text
  • BNB/USDT: Now $790.38 | Entry $789.68 | PnL: +$0.01 (+0.09%)
    🎯 Target: `[██████░░░░]` 60% to TP1 ($793.38) | SL: $785.36
  ```

---

## 🔬 Model Training & Out-of-Sample Performance

Trained across **174,960 samples (365 days across all 20 Binance crypto assets)** using the **Dual-Model Stacking Ensemble (`LightGBM` + `HistGradientBoosting`)** with exponential time-decay sample weighting:

```text
===========================================================================
  DUAL-MODEL STACKING ENSEMBLE OUT-OF-SAMPLE TEST RESULTS (Unseen Forward 20% Data)
===========================================================================
  Base Estimator 1:              HistGradientBoostingClassifier
  Base Estimator 2:              LightGBM (LGBMClassifier)
  Total Clean Samples:           174,960 (Train: 139,968 | Test: 34,992)
  ROC-AUC Score:                 0.638
  Confidence Threshold:          60%
  High-Conviction Trade Signals: 2,466 trades
  High-Conviction Win Rate:      58.8%   (with asymmetric target R:R)
  Profit Factor (after fees):    2.38
  Avg Return per Trade:          +1.08%
  Cumulative Sample PnL:         +2,659.0%

  Top Mathematical Alpha Indicators:
    1. dist_sma200                  (Importance: +0.0473)
    2. mtf_4h_slope                 (Importance: +0.0377)
    3. mtf_4h_rsi                   (Importance: +0.0327)
    4. ret_skew_30                  (Importance: +0.0290)
    5. mtf_4h_dist_ema50            (Importance: +0.0210)
    6. roc_24                       (Importance: +0.0173)
    7. vol_trend                    (Importance: +0.0153)
    8. macd_hist_norm               (Importance: +0.0133)
===========================================================================
```

---

## 🧪 Automated Test Suite (44 Tests, 100% Pass Rate)

Every cloud run and code modification is validated through a **44-point unit and stress test suite**:

```powershell
python -m unittest discover -s tests -p "*.py"
```

* **`tests/hard_test.py`** (20 tests): Microstructure sanitization, MTF confluence, order calculations, fee modeling.
* **`tests/deep_stress_test.py`** (14 tests): Chandelier trailing ratchets, Pearson correlation matrices, circuit breaker recovery.
* **`tests/extreme_level_test.py`** (10 tests): 20-symbol universe integrity, parallel multi-threaded scanner fault tolerance, progress gauge scaling, RS-BTC decoupling, 3-slot risk parity bounds.

---

## 💻 CLI Commands & Workflows

### 1. Download 1-Year Historical Market Data (All 20 Symbols)
```powershell
.\venv\Scripts\python.exe download_historical_data.py
```

### 2. Train AI Model Across All 20 Assets
```powershell
# Retrain model on locally cached 365-day dataset:
.\venv\Scripts\python.exe train.py --offline

# Or fetch fresh live data from Binance Vision API:
.\venv\Scripts\python.exe train.py --days 365 --timeframe 1h
```

### 3. Run Automated 44-Point Test Suite
```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -p "*.py"
```

### 4. Run Live Market Scan & Execute Pending Signals
```powershell
.\venv\Scripts\python.exe bot.py --scan
```

### 5. Send Real-Time Test Alert to Discord
```powershell
.\venv\Scripts\python.exe bot.py --test-discord
```

### 6. Send Daily Portfolio Performance Digest
```powershell
.\venv\Scripts\python.exe bot.py --digest
```

### 7. Run Continuously in Loop Mode (Local Machine / Dedicated VPS)
```powershell
.\venv\Scripts\python.exe bot.py --loop
```

---

## ⚙️ Configuration (`config.json`)

```json
{
  "trading_mode": "paper",
  "binance": {
    "api_key": "",
    "api_secret": "",
    "testnet": false
  },
  "symbols": [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "LINK/USDT", "NEAR/USDT",
    "SUI/USDT", "APT/USDT", "INJ/USDT", "TIA/USDT", "RENDER/USDT",
    "FET/USDT", "SEI/USDT", "ARB/USDT", "OP/USDT", "DOT/USDT"
  ],
  "timeframe": "1h",
  "ai_model": {
    "confidence_threshold": 0.60,
    "tp1_atr_mult": 1.2,
    "tp2_atr_mult": 2.4,
    "sl_atr_mult": 1.4,
    "partial_tp_ratio": 0.50,
    "breakeven_lock_enabled": true
  },
  "correlation_filter": {
    "enabled": true,
    "max_correlation": 0.75
  },
  "trailing_stop": {
    "enabled": true,
    "type": "chandelier",
    "atr_mult": 2.2
  },
  "mtf_confluence": {
    "enabled": true,
    "macro_timeframe": "4h",
    "macro_ema_period": 50,
    "macro_rsi_min": 48.0
  },
  "risk_management": {
    "capital_usdt": 30.0,
    "risk_per_trade_pct": 2.0,
    "max_open_trades": 3,
    "daily_max_drawdown_pct": 8.0,
    "circuit_breaker_hours": 24,
    "max_holding_hours": 48,
    "dynamic_alpha_sizing": true,
    "fee_rate": 0.00075,
    "slippage_rate": 0.0005
  },
  "discord": {
    "enabled": true,
    "webhook_url": "",
    "hourly_radar_alert": true
  }
}
```

---

## 📁 Repository Structure

```text
├── .github/workflows/
│   └── trade_bot.yml           # 24/7 GitHub Actions cloud cron workflow & CI runner
├── data/                       # 365-day 1h historical market candle cache (20 assets, 174,960 bars)
├── models/
│   └── binance_ai_model.joblib # Calibrated Dual-Tree Stacking Ensemble (HistGradientBoosting + LightGBM)
├── logs/
│   └── trade_history.csv       # Persistent trade execution ledger
├── tests/
│   ├── hard_test.py            # 20-point core unit and execution tests
│   ├── deep_stress_test.py     # 14-point deep stress, correlation & volatility trailing tests
│   └── extreme_level_test.py   # 10-point extreme universe, parallel scanning & RS alpha tests
├── binance_client.py           # CCXT Binance exchange client with microstructure sanitizer
├── bot.py                      # Core bot execution, Alpha Ranking, Correlation Filter & Discord alerts
├── features.py                 # 49 quantitative indicators, RS-BTC Alpha & StackingEnsembleModel
├── train.py                    # Multi-Model Stacking Ensemble training pipeline with time-decay weighting
├── backtest.py                 # Vectorized event-driven backtesting engine
├── download_historical_data.py # 1-year historical dataset downloader for 20 assets
├── bot_state.json              # Real-time state journal (cash, active trades, PnL)
├── config.json                 # Central configuration
├── requirements.txt            # Python dependencies (includes LightGBM)
└── README.md                   # System documentation
```
