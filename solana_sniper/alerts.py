"""Discord webhook alert notifications for Solana Sniper."""

import os
import json
import logging
from typing import List, Dict, Any
import requests

logger = logging.getLogger("solana_sniper")


def send_discord_sniper_alert(webhook_url: str, title: str, fields: List[Dict[str, Any]], color: int = 3066993, desc: str = ""):
    if not webhook_url or not webhook_url.startswith("http"):
        return

    payload = {
        "username": "Solana Quantum Sniper",
        "avatar_url": "https://raw.githubusercontent.com/solana-labs/token-list/main/assets/mainnet/So11111111111111111111111111111111111111112/logo.png",
        "embeds": [
            {
                "title": title,
                "description": desc,
                "color": color,
                "fields": fields,
                "footer": {"text": "Solana High-Velocity Moonbag Engine • Raydium & Pump.fun"},
            }
        ],
    }

    try:
        requests.post(webhook_url, json=payload, timeout=5)
    except Exception as e:
        logger.warning("Discord webhook alert failed: %s", e)
