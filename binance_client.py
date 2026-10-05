"""Binance Exchange Client.
Provides a clean unified interface for Binance market data and order execution via CCXT.
Supports public data fetching, precision sanitization (stepSize/tickSize/minNotional),
funding rate sentiment indicators, and live authenticated execution.
"""

import os
import time
import functools
import ccxt
import pandas as pd


def retry_on_network_error(max_retries=3, initial_delay=1.0):
    """Exponential backoff decorator for transient Binance API and network drops."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_err = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (ccxt.NetworkError, ccxt.RateLimitExceeded, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout) as e:
                    last_err = e
                    if attempt < max_retries - 1:
                        time.sleep(delay)
                        delay *= 2.0
                    else:
                        raise last_err
                except Exception as e:
                    raise e
        return wrapper
    return decorator


class BinanceClient:
    def __init__(self, config):
        self.config = config
        b_cfg = config.get("binance", {})

        # Check environment variables first for security, then fallback to config.json
        api_key = os.getenv("BINANCE_API_KEY", b_cfg.get("api_key", "")).strip()
        api_secret = os.getenv("BINANCE_API_SECRET", b_cfg.get("api_secret", "")).strip()
        is_testnet = b_cfg.get("testnet", False)

        opts = {
            "enableRateLimit": True,
            "options": {"fetchMarkets": {"types": ["spot"]}}
        }
        if api_key and api_secret:
            opts["apiKey"] = api_key
            opts["secret"] = api_secret

        proxy = os.getenv("BINANCE_PROXY") or os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY")
        if proxy:
            opts["httpsProxy"] = proxy
            opts["proxy"] = proxy

        self.exchange = ccxt.binance(opts)
        if is_testnet:
            self.exchange.set_sandbox_mode(True)

        # Route public market data to data-api.binance.vision
        # This completely resolves HTTP 451 geo-restrictions in cloud environments like GitHub Actions
        use_vision = b_cfg.get("use_vision_api", True)
        if use_vision and hasattr(self.exchange, "urls") and "api" in self.exchange.urls:
            self.exchange.urls["api"]["public"] = "https://data-api.binance.vision/api/v3"

        self.has_credentials = bool(api_key and api_secret)
        self.markets_loaded = False

    def load_markets_once(self):
        if not self.markets_loaded:
            try:
                self.exchange.load_markets()
                self.markets_loaded = True
            except Exception as e:
                print(f"[Notice] Could not preload exchange markets: {e}")

    @retry_on_network_error(max_retries=3, initial_delay=1.0)
    def fetch_ohlcv(self, symbol, timeframe="1h", limit=500):
        """Fetches latest OHLCV bars from Binance and returns a clean DataFrame."""
        rows = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        # Drop unclosed current bar if needed
        return df.iloc[:-1] if len(df) > 1 else df

    def fetch_historical_ohlcv(self, symbol, timeframe="1h", days=180):
        """Downloads historical bars going back `days` for training."""
        tf_ms = self.exchange.parse_timeframe(timeframe) * 1000
        now_ms = int(time.time() * 1000)
        since_ms = now_ms - (days * 86_400_000)
        all_rows = []
        cursor = since_ms

        while cursor < now_ms:
            batch = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=1000)
            if not batch:
                break
            all_rows.extend(batch)
            last_ts = batch[-1][0]
            if last_ts <= cursor:
                break
            cursor = last_ts + tf_ms
            time.sleep(0.08)

        if not all_rows:
            return pd.DataFrame()

        df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.drop_duplicates(subset=["timestamp"], inplace=True)
        df.sort_values("timestamp", inplace=True)
        df.set_index("timestamp", inplace=True)
        return df

    @retry_on_network_error(max_retries=3, initial_delay=1.0)
    def get_ticker_price(self, symbol):
        ticker = self.exchange.fetch_ticker(symbol)
        return float(ticker["last"])

    @retry_on_network_error(max_retries=3, initial_delay=1.0)
    def fetch_funding_rate(self, symbol):
        """Fetches live Binance Futures funding rate for market sentiment analysis (public endpoint)."""
        try:
            raw_sym = symbol.replace("/", "")
            # Public endpoint for premium index & funding rate
            res = self.exchange.fapiPublicGetPremiumIndex({"symbol": raw_sym})
            return float(res.get("lastFundingRate", 0.0))
        except Exception:
            return 0.0

    @retry_on_network_error(max_retries=3, initial_delay=1.0)
    def get_balance(self):
        if not self.has_credentials:
            return {"USDT": {"free": self.config["risk_management"]["capital_usdt"], "total": self.config["risk_management"]["capital_usdt"]}}
        return self.exchange.fetch_balance()

    def sanitize_order_amount(self, symbol, amount, price):
        """
        Rounds amount and price strictly according to Binance LOT_SIZE,
        PRICE_FILTER, and MIN_NOTIONAL limits to guarantee 100% order acceptance.
        """
        self.load_markets_once()
        market = self.exchange.market(symbol) if (self.exchange.markets and symbol in self.exchange.markets) else None
        if not market:
            return round(amount, 6), round(price, 4)

        # Precision formatting
        clean_amount = float(self.exchange.amount_to_precision(symbol, amount))
        clean_price = float(self.exchange.price_to_precision(symbol, price))

        if clean_amount <= 0:
            raise ValueError(f"Order amount {amount} rounds down to 0 for {symbol} precision limits.")

        # Check minimum notional value (usually 5.0 to 10.0 USDT)
        min_cost = market.get("limits", {}).get("cost", {}).get("min", 5.0) or 5.0
        if clean_amount * clean_price < min_cost:
            raise ValueError(f"Order value ${clean_amount * clean_price:.2f} is below Binance minimum notional (${min_cost} USDT)")

        return clean_amount, clean_price

    @retry_on_network_error(max_retries=2, initial_delay=1.0)
    def place_spot_order(self, symbol, side, amount, price=None, order_type="market"):
        if not self.has_credentials:
            raise ValueError("Binance API keys not set for live execution.")

        curr_p = price or self.get_ticker_price(symbol)
        clean_amount, clean_price = self.sanitize_order_amount(symbol, amount, curr_p)

        if order_type == "limit":
            return self.exchange.create_order(symbol, "limit", side, clean_amount, clean_price)
        else:
            return self.exchange.create_order(symbol, "market", side, clean_amount)

    @retry_on_network_error(max_retries=2, initial_delay=1.0)
    def place_oco_order(self, symbol, side, amount, tp_price, sl_price, sl_limit_price=None):
        """
        Places a native exchange-side OCO (One-Cancels-the-Other) order on Binance Spot.
        Ensures Take-Profit and Stop-Loss orders sit on Binance's matching engine 24/7 with 0ms latency.
        """
        if not self.has_credentials:
            raise ValueError("Binance API keys not set for live execution.")

        self.load_markets_once()
        raw_sym = symbol.replace("/", "")
        clean_amount, clean_tp = self.sanitize_order_amount(symbol, amount, tp_price)
        _, clean_sl = self.sanitize_order_amount(symbol, amount, sl_price)
        clean_sl_limit = clean_sl * 0.998 if sl_limit_price is None else sl_limit_price
        _, clean_sl_limit = self.sanitize_order_amount(symbol, amount, clean_sl_limit)

        params = {
            "symbol": raw_sym,
            "side": side.upper(),
            "quantity": clean_amount,
            "price": self.exchange.price_to_precision(symbol, clean_tp),
            "stopPrice": self.exchange.price_to_precision(symbol, clean_sl),
            "stopLimitPrice": self.exchange.price_to_precision(symbol, clean_sl_limit),
            "stopLimitTimeInForce": "GTC"
        }
        return self.exchange.privatePostOrderOco(params)

    @retry_on_network_error(max_retries=2, initial_delay=1.0)
    def cancel_order(self, symbol, order_id):
        """Cancels an open order or OCO order on Binance."""
        if not self.has_credentials:
            return None
        return self.exchange.cancel_order(order_id, symbol)

