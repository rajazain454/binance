"""Position tracking, Moonbag extraction, and trade execution engine."""

import os
import json
import logging
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple
import requests

logger = logging.getLogger("solana_sniper")


def load_sniper_state(state_path: str, initial_capital_sol: float = 1.0) -> Dict[str, Any]:
    if os.path.exists(state_path):
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Error loading sniper state '%s': %s", state_path, e)

    return {
        "balance_sol": initial_capital_sol,
        "peak_balance_sol": initial_capital_sol,
        "open_positions": {},
        "trade_count": 0,
        "win_count": 0,
        "loss_count": 0,
        "realized_pnl_sol": 0.0,
        "moonbags_harvested": 0,
        "last_update": None,
    }


def save_sniper_state(state: Dict[str, Any], state_path: str):
    tmp_path = f"{state_path}.tmp"
    os.makedirs(os.path.dirname(os.path.abspath(state_path)), exist_ok=True)
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp_path, state_path)


def log_sniper_trade(trade: Dict[str, Any], ledger_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(ledger_path)), exist_ok=True)
    file_exists = os.path.exists(ledger_path)
    with open(ledger_path, "a", encoding="utf-8") as f:
        if not file_exists:
            f.write("timestamp,symbol,mint,action,entry_price,exit_price,multiplier,pnl_sol,reason,outcome\n")
        f.write(
            f"{trade.get('timestamp')},{trade.get('symbol')},{trade.get('mint')},"
            f"{trade.get('action')},{trade.get('entry_price'):.8f},{trade.get('exit_price'):.8f},"
            f"{trade.get('multiplier', 1.0):.2f}x,{trade.get('pnl_sol', 0.0):.4f},"
            f"\"{trade.get('reason', '')}\",{trade.get('outcome', 'WIN')}\n"
        )


WSOL_MINT = "So11111111111111111111111111111111111111112"


def execute_live_jupiter_swap(
    input_mint: str,
    output_mint: str,
    amount_lamports: int,
    user_pubkey: str,
    private_key_b58: str,
    rpc_url: str,
    slippage_bps: int = 150
) -> Tuple[bool, str]:
    """Executes a real on-chain swap via Jupiter DEX aggregator and signs with Keypair."""
    import base64
    from solders.keypair import Keypair
    from solders.transaction import VersionedTransaction

    try:
        quote_url = f"https://api.jup.ag/swap/v1/quote?inputMint={input_mint}&outputMint={output_mint}&amount={amount_lamports}&slippageBps={slippage_bps}"
        q_res = requests.get(quote_url, timeout=5).json()
        if not q_res.get("outAmount"):
            return False, f"Jupiter quote failed: {q_res.get('error', 'No route')}"

        swap_url = "https://api.jup.ag/swap/v1/swap"
        swap_payload = {
            "quoteResponse": q_res,
            "userPublicKey": user_pubkey,
            "wrapAndUnwrapSol": True,
            "dynamicComputeUnitLimit": True,
            "prioritizationFeeLamports": "auto"
        }
        s_res = requests.post(swap_url, json=swap_payload, timeout=6).json()
        swap_tx_b64 = s_res.get("swapTransaction")
        if not swap_tx_b64:
            return False, f"Jupiter swap prep failed: {s_res.get('error', 'No tx')}"

        kp = Keypair.from_base58_string(private_key_b58)
        raw_tx = base64.b64decode(swap_tx_b64)
        tx = VersionedTransaction.from_bytes(raw_tx)
        sig = kp.sign_message(bytes(tx.message))
        signed_tx = VersionedTransaction.populate(tx.message, [sig])
        signed_bytes = bytes(signed_tx)

        rpc_payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "sendTransaction",
            "params": [
                base64.b64encode(signed_bytes).decode("utf-8"),
                {"encoding": "base64", "skipPreflight": True, "maxRetries": 3}
            ]
        }
        send_res = requests.post(rpc_url, json=rpc_payload, timeout=8).json()
        tx_sig = send_res.get("result")
        if tx_sig:
            return True, str(tx_sig)
        else:
            return False, str(send_res.get("error", "RPC broadcast failed"))
    except Exception as e:
        return False, f"Live swap error: {e}"


def get_onchain_token_balance(rpc_url: str, user_pubkey: str, token_mint: str) -> int:
    """Returns raw on-chain token balance (in base units) for the user's token account."""
    try:
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getTokenAccountsByOwner",
            "params": [
                user_pubkey,
                {"mint": token_mint},
                {"encoding": "jsonParsed"}
            ]
        }
        r = requests.post(rpc_url, json=payload, timeout=6).json()
        accounts = r.get("result", {}).get("value", [])
        if accounts:
            amount_str = accounts[0]["account"]["data"]["parsed"]["info"]["tokenAmount"]["amount"]
            return int(amount_str)
    except Exception as e:
        logger.warning("Error fetching on-chain token balance for %s: %s", token_mint, e)
    return 0


def fetch_live_token_price(mint: str) -> float:
    """Fetches real-time price in USD from DexScreener."""
    try:
        url = f"https://api.dexscreener.com/latest/dex/tokens/{mint}"
        r = requests.get(url, headers={"User-Agent": "SolanaSniper/1.0"}, timeout=4)
        if r.status_code == 200:
            pairs = r.json().get("pairs", [])
            if pairs:
                return float(pairs[0].get("priceUsd", 0.0) or 0.0)
    except Exception:
        pass
    return 0.0


def manage_active_positions(
    state: Dict[str, Any], 
    config: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Evaluates open positions for TP1 (Moonbag Extraction), Trailing Stops, and Stop Loss.
    """
    exit_cfg = config.get("exit_rules", {})
    tp1_mult = exit_cfg.get("tp1_multiplier", 2.0)
    tp1_ratio = exit_cfg.get("tp1_harvest_ratio", 0.5)
    stop_loss_pct = exit_cfg.get("stop_loss_pct", 0.25)
    trailing_cb = exit_cfg.get("trailing_stop_callback_pct", 0.15)
    stale_timeout_mins = exit_cfg.get("stale_timeout_minutes", 20)

    closed_trades = []
    partial_harvests = []
    open_positions = state.setdefault("open_positions", {})
    positions_to_remove = []
    is_live = config.get("mode", "paper").lower() == "live"
    sol_cfg = config.get("solana", {})

    for mint, pos in open_positions.items():
        symbol = pos["symbol"]
        entry_p = pos["entry_price_usd"]
        curr_p = fetch_live_token_price(mint)
        if curr_p <= 0:
            continue

        pos["current_price_usd"] = curr_p
        multiplier = curr_p / entry_p if entry_p > 0 else 1.0
        pos["multiplier"] = multiplier

        # Track Peak Price
        if curr_p > pos.get("peak_price_usd", curr_p):
            pos["peak_price_usd"] = curr_p

        peak_p = pos.get("peak_price_usd", curr_p)

        # 1. MOONBAG EXTRACTION (TP1 at 2.0x): Take out initial capital
        if not pos.get("tp1_taken", False) and multiplier >= tp1_mult:
            pos["tp1_taken"] = True
            pos["trailing_active"] = True
            rec_sol = pos["allocated_sol"] * tp1_ratio * multiplier
            state["balance_sol"] += rec_sol
            state["realized_pnl_sol"] += (rec_sol - (pos["allocated_sol"] * tp1_ratio))
            state["moonbags_harvested"] = state.get("moonbags_harvested", 0) + 1
            pos["allocated_sol"] *= (1.0 - tp1_ratio)

            if is_live and sol_cfg.get("private_key"):
                raw_bal = get_onchain_token_balance(sol_cfg.get("rpc_url"), sol_cfg.get("public_key"), mint)
                if raw_bal > 0:
                    sell_amount = int(raw_bal * tp1_ratio)
                    ok, tx_or_err = execute_live_jupiter_swap(
                        input_mint=mint,
                        output_mint=WSOL_MINT,
                        amount_lamports=sell_amount,
                        user_pubkey=sol_cfg.get("public_key"),
                        private_key_b58=sol_cfg.get("private_key"),
                        rpc_url=sol_cfg.get("rpc_url")
                    )
                    if ok:
                        logger.info("🎯 [LIVE TP1 SWAP SUCCESS] %s TX: https://solscan.io/tx/%s", symbol, tx_or_err)
                    else:
                        logger.error("❌ [LIVE TP1 SWAP FAILED] %s: %s", symbol, tx_or_err)

            harvest_ev = {
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                "symbol": symbol,
                "mint": mint,
                "action": "MOONBAG_EXTRACTED_2X",
                "entry_price": entry_p,
                "exit_price": curr_p,
                "multiplier": multiplier,
                "pnl_sol": rec_sol - (pos["allocated_sol"] * tp1_ratio),
                "reason": f"🎯 2X MOONBAG SECURED! 50% Capital Extracted ({multiplier:.2f}x). Remaining 50% Running Free!",
                "outcome": "WIN",
            }
            partial_harvests.append(harvest_ev)
            log_sniper_trade(harvest_ev, config["paths"]["trade_ledger"])
            logger.info("🎯 [MOONBAG SECURED] %s reached %.2fx! 50%% sold back to SOL. Remaining position is pure profit!", symbol, multiplier)

        # 2. TRAILING STOP ON MOONBAG / RUNNER
        should_exit = False
        outcome = "WIN"
        reason = ""

        if pos.get("trailing_active", False):
            trailing_stop = peak_p * (1.0 - trailing_cb)
            if curr_p <= trailing_stop:
                should_exit = True
                outcome = "WIN"
                reason = f"🚀 Moonbag Trailing Harvest (Peak: ${peak_p:.6f}, Drop: -{trailing_cb*100:.1f}%)"

        # 2.5 BREAKEVEN PROTECTION (If token peaked >= +20% but pulled back, protect capital)
        elif peak_p >= (entry_p * 1.20) and multiplier <= 1.02:
            should_exit = True
            outcome = "WIN" if multiplier >= 1.0 else "LOSS"
            reason = f"🛡️ Breakeven Guard (Peaked at +{(peak_p/entry_p - 1.0)*100:.1f}%, secured at entry)"

        # 3. HARD STOP LOSS (Only applies before TP1 moonbag is taken)
        elif multiplier <= (1.0 - stop_loss_pct):
            should_exit = True
            outcome = "LOSS"
            reason = f"🛡️ Risk Guard Stop Loss (-{stop_loss_pct*100:.1f}%)"

        # 4. STALE TIMEOUT EXIT
        if not should_exit:
            entry_time_ts = pos.get("entry_time_ts", time.time())
            mins_held = (time.time() - entry_time_ts) / 60.0
            if mins_held >= stale_timeout_mins and multiplier < 1.20:
                should_exit = True
                outcome = "WIN" if multiplier >= 1.0 else "LOSS"
                reason = f"⏳ Stale Momentum Cut (Held {mins_held:.1f}m without 2x breakout)"

        if should_exit:
            positions_to_remove.append(mint)
            state.setdefault("cooldown_mints", {})[mint] = time.time()

            if is_live and sol_cfg.get("private_key"):
                raw_bal = get_onchain_token_balance(sol_cfg.get("rpc_url"), sol_cfg.get("public_key"), mint)
                if raw_bal > 0:
                    ok, tx_or_err = execute_live_jupiter_swap(
                        input_mint=mint,
                        output_mint=WSOL_MINT,
                        amount_lamports=raw_bal,
                        user_pubkey=sol_cfg.get("public_key"),
                        private_key_b58=sol_cfg.get("private_key"),
                        rpc_url=sol_cfg.get("rpc_url")
                    )
                    if ok:
                        logger.info("🏁 [LIVE EXIT SWAP SUCCESS] %s TX: https://solscan.io/tx/%s", symbol, tx_or_err)
                    else:
                        logger.error("❌ [LIVE EXIT SWAP FAILED] %s: %s", symbol, tx_or_err)

            rem_sol = pos["allocated_sol"] * multiplier
            net_pnl = rem_sol - pos["allocated_sol"]
            state["balance_sol"] += rem_sol
            state["realized_pnl_sol"] += net_pnl
            state["trade_count"] = state.get("trade_count", 0) + 1
            if outcome == "WIN" and net_pnl > 0:
                state["win_count"] = state.get("win_count", 0) + 1
            elif outcome == "LOSS":
                state["loss_count"] = state.get("loss_count", 0) + 1

            close_ev = {
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                "symbol": symbol,
                "mint": mint,
                "action": "SNIPE_CLOSED",
                "entry_price": entry_p,
                "exit_price": curr_p,
                "multiplier": multiplier,
                "pnl_sol": net_pnl,
                "reason": reason,
                "outcome": outcome,
            }
            closed_trades.append(close_ev)
            log_sniper_trade(close_ev, config["paths"]["trade_ledger"])
            logger.info("🏁 [POSITION CLOSED] %s Sold at $%.6f (%.2fx). PnL: %+.4f SOL | %s", symbol, curr_p, multiplier, net_pnl, reason)

    for m in positions_to_remove:
        open_positions.pop(m, None)

    return partial_harvests, closed_trades


def execute_snipe(
    candidate: Dict[str, Any], 
    state: Dict[str, Any], 
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Executes a new token snipe entry (Paper or Live).
    """
    trading_cfg = config.get("trading", {})
    bullet_size = trading_cfg.get("bullet_size_sol", 0.1)
    
    if state["balance_sol"] < bullet_size:
        return None

    state["balance_sol"] -= bullet_size
    mint = candidate["mint"]
    symbol = candidate["symbol"]
    entry_p = float(candidate.get("price_usd", 0.0) or 0.0)
    is_live = config.get("mode", "paper").lower() == "live"
    sol_cfg = config.get("solana", {})
    tx_sig = None
    if is_live and sol_cfg.get("private_key"):
        amount_lamports = int(bullet_size * 1e9)
        ok, res = execute_live_jupiter_swap(
            input_mint=WSOL_MINT,
            output_mint=mint,
            amount_lamports=amount_lamports,
            user_pubkey=sol_cfg.get("public_key"),
            private_key_b58=sol_cfg.get("private_key"),
            rpc_url=sol_cfg.get("rpc_url")
        )
        if not ok:
            logger.error("❌ [LIVE SNIPE FAILED] %s: %s", symbol, res)
            state["balance_sol"] += bullet_size
            return None
        tx_sig = res
        logger.info("🚀 [LIVE SNIPE BROADCAST] %s TX: https://solscan.io/tx/%s", symbol, tx_sig)

    new_pos = {
        "mint": mint,
        "symbol": symbol,
        "entry_time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "entry_time_ts": time.time(),
        "entry_price_usd": entry_p,
        "current_price_usd": entry_p,
        "peak_price_usd": entry_p,
        "allocated_sol": bullet_size,
        "multiplier": 1.0,
        "tp1_taken": False,
        "trailing_active": False,
        "dex": candidate.get("dex", "raydium"),
        "url": candidate.get("url", ""),
        "tx_sig": tx_sig,
    }

    state.setdefault("open_positions", {})[mint] = new_pos
    logger.info("🟢 [SNIPER BUY] Sniped %s (%s) with %.2f SOL at $%.6f", symbol, mint[:8], bullet_size, entry_p)
    return new_pos
