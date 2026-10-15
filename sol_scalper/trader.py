"""Order execution and state persistence engine for SOL/USDC Scalper (Paper & Live)."""

import base64
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Dict, Any, Tuple, Optional
import requests

logger = logging.getLogger("sol_scalper")


def load_scalper_state(state_path: str, initial_capital_usd: float = 50.0) -> Dict[str, Any]:
    if os.path.exists(state_path):
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Error loading scalper state '%s': %s", state_path, e)

    return {
        "cash_usd": initial_capital_usd,
        "peak_equity_usd": initial_capital_usd,
        "active_position": None,
        "trade_count": 0,
        "win_count": 0,
        "loss_count": 0,
        "realized_pnl_usd": 0.0,
        "last_trade_time": 0,
        "last_update": None,
    }


def save_scalper_state(state: Dict[str, Any], state_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(state_path)), exist_ok=True)
    for _ in range(5):
        try:
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
            return
        except Exception:
            time.sleep(0.05)


def log_scalper_trade(trade: Dict[str, Any], ledger_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(ledger_path)), exist_ok=True)
    file_exists = os.path.exists(ledger_path)
    with open(ledger_path, "a", encoding="utf-8") as f:
        if not file_exists:
            f.write("timestamp,symbol,action,entry_price,exit_price,gain_pct,pnl_usd,reason,outcome\n")
        f.write(
            f"{trade.get('timestamp')},{trade.get('symbol','SOLUSDC')},"
            f"{trade.get('action')},{trade.get('entry_price'):.4f},{trade.get('exit_price'):.4f},"
            f"{trade.get('gain_pct', 0.0):+.2f}%,{trade.get('pnl_usd', 0.0):+.4f},"
            f"\"{trade.get('reason', '')}\",{trade.get('outcome', 'WIN')}\n"
        )


def execute_jupiter_live_swap(
    input_mint: str,
    output_mint: str,
    amount_units: int,
    user_pubkey: str,
    private_key_b58: str,
    rpc_url: str,
    slippage_bps: int = 50
) -> Tuple[bool, str]:
    """Executes a real on-chain swap between USDC and SOL via Jupiter DEX aggregator."""
    from solders.keypair import Keypair
    from solders.transaction import VersionedTransaction

    try:
        quote_url = f"https://api.jup.ag/swap/v1/quote?inputMint={input_mint}&outputMint={output_mint}&amount={amount_units}&slippageBps={slippage_bps}"
        q_res = requests.get(quote_url, timeout=5).json()
        if not q_res.get("outAmount"):
            return False, f"Jupiter quote error: {q_res.get('error', 'No route')}"

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
        return False, f"Live Jupiter swap error: {e}"


def execute_buy_order(
    snapshot: Dict[str, Any], 
    state: Dict[str, Any], 
    config: Dict[str, Any], 
    reason: str
) -> Optional[Dict[str, Any]]:
    """Opens a new SOL position (Paper or Live)."""
    trading_cfg = config.get("trading", {})
    cash = state.get("cash_usd", 50.0)

    # Dynamic Compounding Sizing (e.g. 50% of available cash per trade)
    if trading_cfg.get("dynamic_compounding", True):
        size_pct = float(trading_cfg.get("position_size_pct", 50.0)) / 100.0
        pos_size_usd = max(10.0, round(cash * size_pct, 2))
    else:
        pos_size_usd = float(trading_cfg.get("position_size_usd", 25.0))

    if cash < pos_size_usd or pos_size_usd < 5.0:
        logger.warning("Insufficient cash ($%.2f) to open $%.2f position", cash, pos_size_usd)
        return None

    entry_price = snapshot["price"]
    sol_amount = round(pos_size_usd / entry_price, 4)
    state["cash_usd"] -= pos_size_usd

    is_live = config.get("mode", "paper").lower() == "live"
    sol_cfg = config.get("solana", {})
    tx_sig = None

    if is_live and sol_cfg.get("private_key"):
        # USDC has 6 decimals
        usdc_amount_units = int(pos_size_usd * 1e6)
        ok, res = execute_jupiter_live_swap(
            input_mint=sol_cfg.get("usdc_mint", "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"),
            output_mint=sol_cfg.get("wsol_mint", "So11111111111111111111111111111111111111112"),
            amount_units=usdc_amount_units,
            user_pubkey=sol_cfg.get("public_key"),
            private_key_b58=sol_cfg.get("private_key"),
            rpc_url=sol_cfg.get("rpc_url")
        )
        if not ok:
            logger.error("❌ [LIVE BUY SWAP FAILED]: %s", res)
            state["cash_usd"] += pos_size_usd
            return None
        tx_sig = res
        logger.info("🚀 [LIVE BUY BROADCAST] TX: https://solscan.io/tx/%s", tx_sig)

    new_pos = {
        "symbol": "SOLUSDC",
        "entry_time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "entry_time_ts": time.time(),
        "entry_price_usd": entry_price,
        "peak_price_usd": entry_price,
        "allocated_usd": pos_size_usd,
        "sol_amount": sol_amount,
        "target_tp_price": round(entry_price * (1.0 + float(trading_cfg.get("take_profit_pct", 1.8)) / 100.0), 4),
        "target_sl_price": round(entry_price * (1.0 - float(trading_cfg.get("stop_loss_pct", 0.9)) / 100.0), 4),
        "reason": reason,
        "tx_sig": tx_sig,
    }

    state["active_position"] = new_pos
    state["last_trade_time"] = time.time()
    logger.info("🟢 [SCALPER BUY] Bought %.4f SOL at $%.2f ($%.2f allocated)", sol_amount, entry_price, pos_size_usd)
    return new_pos


def execute_sell_order(
    snapshot: Dict[str, Any], 
    state: Dict[str, Any], 
    config: Dict[str, Any], 
    outcome: str, 
    reason: str
) -> Dict[str, Any]:
    """Closes the active SOL position, returning proceeds to cash."""
    pos = state["active_position"]
    exit_price = snapshot["price"]
    entry_price = pos["entry_price_usd"]
    allocated = pos["allocated_usd"]

    gross_return = (exit_price / entry_price) * allocated
    net_pnl = gross_return - allocated
    gain_pct = (exit_price - entry_price) / entry_price * 100.0

    state["cash_usd"] += gross_return
    state["realized_pnl_usd"] += net_pnl
    state["trade_count"] = state.get("trade_count", 0) + 1
    if outcome == "WIN":
        state["win_count"] = state.get("win_count", 0) + 1
    else:
        state["loss_count"] = state.get("loss_count", 0) + 1

    is_live = config.get("mode", "paper").lower() == "live"
    sol_cfg = config.get("solana", {})
    tx_sig = None

    if is_live and sol_cfg.get("private_key"):
        # SOL has 9 decimals
        sol_amount_lamports = int(pos["sol_amount"] * 1e9)
        ok, res = execute_jupiter_live_swap(
            input_mint=sol_cfg.get("wsol_mint", "So11111111111111111111111111111111111111112"),
            output_mint=sol_cfg.get("usdc_mint", "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"),
            amount_units=sol_amount_lamports,
            user_pubkey=sol_cfg.get("public_key"),
            private_key_b58=sol_cfg.get("private_key"),
            rpc_url=sol_cfg.get("rpc_url")
        )
        if ok:
            tx_sig = res
            logger.info("🏁 [LIVE SELL BROADCAST] TX: https://solscan.io/tx/%s", tx_sig)
        else:
            logger.error("❌ [LIVE SELL SWAP FAILED]: %s", res)

    close_ev = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": "SOLUSDC",
        "action": "SCALP_CLOSED",
        "entry_price": entry_price,
        "exit_price": exit_price,
        "gain_pct": gain_pct,
        "pnl_usd": net_pnl,
        "reason": reason,
        "outcome": outcome,
        "tx_sig": tx_sig,
    }

    log_scalper_trade(close_ev, config["paths"]["trade_ledger"])
    state["active_position"] = None
    state["last_trade_time"] = time.time()
    logger.info("🏁 [SCALP CLOSED] Sold at $%.2f (%+.2f%%). Net PnL: %+.2f USD | %s", exit_price, gain_pct, net_pnl, reason)
    return close_ev
