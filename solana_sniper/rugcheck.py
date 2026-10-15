"""Automated Rug and Security Analyzer for Solana Tokens using RugCheck.xyz API."""

import logging
from typing import Dict, Any, Tuple
import requests

logger = logging.getLogger("solana_sniper")


def verify_token_security(token_address: str, safety_cfg: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Checks token security parameters against RugCheck API.
    Returns: (is_safe, reason_or_error, details)
    """
    url = f"https://api.rugcheck.xyz/v1/tokens/{token_address}/report/summary"
    headers = {"User-Agent": "SolanaSniper/1.0"}
    
    try:
        resp = requests.get(url, headers=headers, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            lp_locked = data.get("lpLockedPct", 0.0)
            score = data.get("score", 0)
            risks = data.get("risks", [])

            # 1. Check LP Lock Percentage
            min_lp = safety_cfg.get("min_lp_locked_pct", 80.0)
            if lp_locked < min_lp and lp_locked > 0:
                return False, f"LP Lock too low ({lp_locked:.1f}% < {min_lp}%)", data

            # 2. Check Rug Risks
            critical_risks = [
                r.get("name", "") for r in risks 
                if r.get("level") in ("danger", "critical") or "mint" in r.get("name", "").lower() or "freeze" in r.get("name", "").lower()
            ]
            if critical_risks:
                return False, f"Critical risk detected: {', '.join(critical_risks[:2])}", data

            # 3. Check Overall RugCheck Score
            max_score = safety_cfg.get("max_rugcheck_score", 1000)
            if score > max_score:
                return False, f"RugCheck risk score too high ({score} > {max_score})", data

            return True, f"Verified Safe (LP: {lp_locked:.1f}%, Score: {score})", data

        elif resp.status_code == 404:
            # Token is too new for RugCheck report
            return True, "Brand new token (Unindexed on RugCheck, proceed with caution)", {}
        else:
            return True, f"RugCheck API returned status {resp.status_code} (Bypass)", {}

    except Exception as e:
        logger.warning("RugCheck verification warning for %s: %s", token_address, e)
        # Non-blocking fallback so API hiccups do not choke the sniper
        return True, "RugCheck timeout (Passed via fallback)", {}
