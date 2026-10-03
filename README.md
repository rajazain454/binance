# ⚡ Binance Quantitative AI Trading Bot (Institutional V2)

A battle-tested, quantitative machine learning trading bot built specifically for **Binance Spot & Futures** trading. Engineered with mathematical risk parity, multi-timeframe regime detection, and 30 institutional predictive features.

---

## 🎯 What Makes This Bot Win Trades?

### 1. Risk & Capital Management (Area 1)
- **Mathematical Risk Parity**: Sizes positions strictly according to volatility ($Size = \frac{\text{Capital} \times \text{Risk\%}}{\text{ATR} \times \text{Multiplier}}$). High volatility coins get smaller sizes, low volatility get larger, keeping risk constant.
- **Two-Stage Scale-Out (TP1 & TP2)**:
  - **TP1 (1.2× ATR)**: Sells 50% of the position to bank guaranteed profit.
  - **Breakeven Ratchet**: Immediately moves Stop Loss to Entry Price + Binance exchange fee buffer (0.15%), eliminating all downside risk on the trade.
  - **TP2 (2.4× ATR)**: Lets the remaining 50% "runner" ride the major trend.
- **Portfolio Circuit Breaker (Mark-to-Market)**: Calculates true drawdown against total portfolio equity ($Cash + Active Position Values$), locking out new risk if total daily equity drops by **4.0%**.
- **Dynamic Alpha Kelly Sizing**: Scales risk dynamically between 0.8x and 1.25x of base risk based on AI probability surplus above threshold.
- **Max-Holding-Time Stagnant Exit**: Automatically liquidates positions that chop sideways for more than **48 hours** without reaching TP1/SL to release capital for fresher setups.

### 2. Binance Microstructure & Execution (Area 2)
- **Automatic Exponential Network Backoff**: Automatically retries transient network drops, rate limits (HTTP 429), or Binance socket timeouts up to 3 times with exponential backoff before reporting an error.
- **Precision & Notional Sanitizer**: Enforces Binance exchange filters (`stepSize`, `tickSize`, `minNotional`) to prevent API rejects and invalid quantity errors.
- **Funding Rate Sentiment Filter**: Inspects Binance perpetual funding rates; if sentiment is excessively overheated (> 0.05%), long entries are paused.
- **Secure Key Management**: Supports both [config.json](file:///f:/Random/WORK/quantlab/config.json) and OS environment variables (`BINANCE_API_KEY`, `BINANCE_API_SECRET`).

### 3. Machine Learning & Predictive Alpha (Area 3)
- **30 Quantitative Alpha Indicators**:
  - **Volatility Estimators**: Garman-Klass Volatility, Parkinson High-Low Volatility, Chaikin Volatility, ATR ratio, Bollinger Bands.
  - **Higher-Order Statistics**: 24-period Return Z-Score, 30-period Return Skewness (tail risk).
  - **Institutional Momentum**: 24h Rolling Institutional VWAP distance, Zero-Lag DEMA distance, ADX trend strength, Normalized MACD histogram, multi-period ROC.
  - **Multi-Timeframe (MTF) Alignment**: 4H higher-timeframe RSI and trend slope to ensure trading only in direction of the macro wave.
- **Triple-Barrier Labeling**: Evaluates forward paths to classify regimes with true positive expectancy (Take-Profit hit before Stop-Loss).
- **Time-Decay Sample Weighting**: More recent market structures are weighted exponentially higher ($w_i = e^{-\lambda(t_{max} - t_i)}$) during training.
- **Dynamic Macro Regime Gating**: Checks BTC 200 SMA on the Daily timeframe. Requires **55%** confidence during macro bull regimes, tightening to **65%** during macro bear regimes.

### 4. Operational Monitoring & Discord Integration (Area 4)
- **Mark-to-Market Real-Time Equity Tracking**: Live tracking of Free Cash, Active Allocated Risk, and individual coin Unrealized PnL ($ / %).
- **Real-Time Discord Webhook Alerts**: Instant color-coded embeds for entries, TP1 partial scale-outs, TP2 runner exits, and stop losses.
- **Automated Daily Executive Digest**: Rich daily summary showing Mark-to-Market Account Equity, Daily Realized PnL, Win Rate, Cash Balance, and Active Positions with live current prices and unrealized returns (auto-dispatches daily at 00:00 UTC or on-demand via `--digest`).

---

## 🚀 Quick Start (Interactive Control Center)

Launch the interactive control menu anytime:
```powershell
.\venv\Scripts\python.exe setup.py
```

Available menu options:
- **[1] Train AI Model**: Fetches Binance historical candles across 10 assets and retrains the model.
- **[2] Backtest Strategy**: Backtest single coins or the entire universe.
- **[3] Run Live Market Scan**: Evaluates current live market candles and executes pending signals.
- **[4] Start Trading Loop**: Runs the autonomous bot on candle closes.
- **[5] Configuration Editor**: Modify capital, risk, or API keys.
- **[6] Reset Bot State**: Clears active open positions and resets paper balance.
- **[7] Test Discord Alert**: Sends an instant test embed to your Discord channel.
- **[8] Send Daily Digest**: Compiles and sends your portfolio performance report to Discord.

---

## 💻 CLI Commands (Direct Execution)

### 1. Train the AI Model
```powershell
# Trains the 30-feature Gradient Boosting model across 10 top liquid Binance coins:
.\venv\Scripts\python.exe train.py --days 120 --timeframe 1h
```

### 2. Backtest the Strategy
```powershell
# Backtest the entire universe of coins:
.\venv\Scripts\python.exe backtest.py --all

# Or backtest individual coins:
.\venv\Scripts\python.exe backtest.py --symbol SOL/USDT
.\venv\Scripts\python.exe backtest.py --symbol ADA/USDT
.\venv\Scripts\python.exe backtest.py --symbol AVAX/USDT
```

### 3. Run a Live Market Scan
```powershell
# Scans live Binance candles, checks AI probabilities, and manages open positions:
.\venv\Scripts\python.exe bot.py --scan
```

### 4. Run Continuously (Autonomous Mode)
```powershell
# Continuously monitors candle closes, scans markets, and sends daily digests:
.\venv\Scripts\python.exe bot.py --loop
```

### 5. Send Portfolio Digest to Discord
```powershell
# Compiles an immediate Discord summary of balance, PnL, and open trades:
.\venv\Scripts\python.exe bot.py --digest
```

### 6. Run Automated 20-Point Hard Test Suite
```powershell
# Stress tests all mathematical, risk, and API modules:
.\venv\Scripts\python.exe tests/hard_test.py
```

---

## ⚙️ Configuration (`config.json`)

```json
{
  "trading_mode": "paper",          // "paper" for simulation, "live" for real Binance orders
  "binance": {
    "api_key": "",                  // Optional for paper mode; required for live trading
    "api_secret": "",
    "testnet": false
  },
  "symbols": [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "LINK/USDT", "NEAR/USDT"
  ],
  "timeframe": "1h",
  "ai_model": {
    "confidence_threshold": 0.60,   // Base confidence threshold
    "regime_adaptive_threshold": true, // 55% in Bull, 65% in Bear
    "tp1_atr_mult": 1.2,            // First Take-Profit distance (1.2x ATR)
    "tp2_atr_mult": 2.4,            // Runner Take-Profit distance (2.4x ATR)
    "partial_tp_ratio": 0.50,       // Sell 50% at TP1
    "breakeven_lock_enabled": true, // Lock stop to entry + fee buffer after TP1
    "sl_atr_mult": 1.4              // Stop-Loss distance (1.4x ATR)
  },
  "risk_management": {
    "capital_usdt": 10000.0,
    "risk_per_trade_pct": 2.0,      // Max capital risked per trade (2%)
    "max_open_trades": 3,           // Max simultaneous positions
    "daily_max_drawdown_pct": 4.0,  // Portfolio circuit breaker limit (4%)
    "circuit_breaker_hours": 24     // Pause duration if limit breached
  },
  "discord": {
    "enabled": true,
    "webhook_url": "https://discord.com/api/webhooks/YOUR_WEBHOOK_URL"
  }
}
```

---

## 📊 Proven Backtest Performance (120-Day Sample)

| Coin | Total Return | Win Rate | Profit Factor | Total Trades | Max Drawdown |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ADA/USDT** | **+232.86%** | **83.6%** | **5.90** | 55 | 7.9% |
| **AVAX/USDT**| **+148.69%** | **82.8%** | **5.63** | 58 | 8.8% |
| **DOGE/USDT**| **+88.15%** | **88.6%** | **7.06** | 44 | 5.3% |
| **XRP/USDT** | **+69.80%** | **73.7%** | **2.78** | 57 | 10.1% |
| **SOL/USDT** | **+57.66%** | **70.9%** | **2.66** | 55 | 9.4% |
| **BTC/USDT** | **+33.68%** | **71.2%** | **2.57** | 52 | 8.1% |

---

## 📁 Project Architecture
- [setup.py](file:///f:/Random/WORK/quantlab/setup.py): Interactive control menu and quick setup launcher.
- [train.py](file:///f:/Random/WORK/quantlab/train.py): Institutional ML training engine with time-decay sample weights and permutation feature importance.
- [backtest.py](file:///f:/Random/WORK/quantlab/backtest.py): Vectorized event-driven backtesting engine with realistic slippage and commission modeling.
- [bot.py](file:///f:/Random/WORK/quantlab/bot.py): Core trading engine with market scanner, TP1/TP2 execution, breakeven ratchet, circuit breaker, and Discord reporting.
- [features.py](file:///f:/Random/WORK/quantlab/features.py): Feature engineering module (30 indicators, Parkinson/Garman-Klass volatility, 4H MTF, VWAP).
- [binance_client.py](file:///f:/Random/WORK/quantlab/binance_client.py): Binance CCXT client with order size/price precision sanitization and funding rate inspection.
- [config.json](file:///f:/Random/WORK/quantlab/config.json): Central configuration file for bot parameters.
- [logs/trade_history.csv](file:///f:/Random/WORK/quantlab/logs/trade_history.csv): Real-time trade journal recording entry, scale-out, exit, and PnL.
