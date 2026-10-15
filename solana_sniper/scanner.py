"""Real-time Solana & Pump.fun Meme Token Discovery Scanner."""

import logging
import time
from typing import List, Dict, Any
import requests

logger = logging.getLogger("solana_sniper")


def scan_solana_momentum_tokens(trading_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Scans DexScreener for top Solana momentum tokens matching velocity,
    liquidity, and buyer accumulation thresholds.
    """
    candidates = []
    seen_addresses = set()

    headers = {"User-Agent": "SolanaSniper/1.0"}

    # 1. Fetch top boosted and trending Solana tokens
    urls = [
        "https://api.dexscreener.com/token-profiles/latest/v1",
        "https://api.dexscreener.com/token-boosts/top/v1",
        "https://api.dexscreener.com/token-boosts/latest/v1",
    ]

    token_addresses = []
    for u in urls:
        try:
            r = requests.get(u, headers=headers, timeout=5)
            if r.status_code == 200:
                data = r.json()
                for item in data:
                    if item.get("chainId") == "solana":
                        addr = item.get("tokenAddress")
                        if addr and addr not in seen_addresses:
                            seen_addresses.add(addr)
                            token_addresses.append(addr)
        except Exception as e:
            logger.warning("Error fetching token list from %s: %s", u, e)

    if not token_addresses:
        return []

    # 2. Batch inspect tokens (DexScreener allows up to 30 addresses comma-separated)
    batch_size = 30
    for i in range(0, min(len(token_addresses), 60), batch_size):
        batch = token_addresses[i : i + batch_size]
        query_str = ",".join(batch)
        detail_url = f"https://api.dexscreener.com/latest/dex/tokens/{query_str}"

        try:
            r = requests.get(detail_url, headers=headers, timeout=8)
            if r.status_code != 200:
                continue

            pairs = r.json().get("pairs", [])
            for p in pairs:
                if p.get("chainId") != "solana":
                    continue

                base_token = p.get("baseToken", {})
                mint = base_token.get("address", "")
                symbol = base_token.get("symbol", "UNKNOWN")
                pair_addr = p.get("pairAddress", "")

                price_usd = float(p.get("priceUsd", 0.0) or 0.0)
                liq_usd = float((p.get("liquidity", {}) or {}).get("usd", 0.0) or 0.0)
                vol_m5 = float((p.get("volume", {}) or {}).get("m5", 0.0) or 0.0)
                txns_m5 = (p.get("txns", {}) or {}).get("m5", {})
                buys_m5 = int(txns_m5.get("buys", 0) or 0)
                sells_m5 = int(txns_m5.get("sells", 0) or 0)

                created_at_ms = p.get("pairCreatedAt", 0)
                age_hours = (time.time() * 1000 - created_at_ms) / (1000 * 3600) if created_at_ms else 0.0

                price_change_m5 = float((p.get("priceChange", {}) or {}).get("m5", 0.0) or 0.0)

                fdv_usd = float(p.get("fdv", 0.0) or p.get("marketCap", 0.0) or 0.0)

                # Check Filters
                if liq_usd < trading_cfg.get("min_liquidity_usd", 15000.0):
                    continue
                if vol_m5 < trading_cfg.get("min_5m_volume_usd", 10000.0):
                    continue
                
                min_price_change = float(trading_cfg.get("min_5m_price_change_pct", 1.5))
                if price_change_m5 < min_price_change:
                    continue

                ratio = buys_m5 / max(1, sells_m5)
                if ratio < trading_cfg.get("min_buy_sell_ratio", 1.3):
                    continue

                if age_hours > trading_cfg.get("max_token_age_hours", 2.0):
                    continue

                max_fdv = float(trading_cfg.get("max_fdv_usd", 0.0))
                if max_fdv > 0 and fdv_usd > max_fdv:
                    continue

                is_pump_fun = mint.endswith("pump") or "pump" in p.get("dexId", "").lower()
                if trading_cfg.get("require_pump_fun", False) and not is_pump_fun:
                    continue

                candidates.append({
                    "symbol": symbol,
                    "mint": mint,
                    "pair_address": pair_addr,
                    "dex": p.get("dexId", "raydium"),
                    "price_usd": price_usd,
                    "liquidity_usd": liq_usd,
                    "volume_5m": vol_m5,
                    "buys_5m": buys_m5,
                    "sells_5m": sells_m5,
                    "buy_sell_ratio": ratio,
                    "age_hours": age_hours,
                    "is_pump_fun": is_pump_fun,
                    "price_change_5m": price_change_m5,
                    "url": p.get("url", f"https://dexscreener.com/solana/{pair_addr}"),
                })

        except Exception as e:
            logger.warning("Error fetching batch token details: %s", e)

    # Sort candidates by 5m volume and buy/sell ratio descending (highest velocity first)
    candidates.sort(key=lambda x: (x["volume_5m"] * x["buy_sell_ratio"]), reverse=True)
    return candidates
