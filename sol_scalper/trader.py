"""Multi-asset order execution, Jupiter DEX routing, and balance tracking (Paper & Live)."""

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
                st = json.load(f)
                if "open_positions" not in st:
                    st["open_positions"] = {}
                return st
        except Exception as e:
            logger.warning("Error loading scalper state '%s': %s", state_path, e)

    return {
        "cash_usd": initial_capital_usd,
        "peak_equity_usd": initial_capital_usd,
        "open_positions": {},
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
            f"{trade.get('timestamp')},{trade.get('symbol')},"
            f"{trade.get('action')},{trade.get('entry_price'):.6f},{trade.get('exit_price'):.6f},"
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
    """Executes a real on-chain swap via Jupiter DEX aggregator."""
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
    symbol: str,
    snapshot: Dict[str, Any], 
    state: Dict[str, Any], 
    config: Dict[str, Any], 
    reason: str
) -> Optional[Dict[str, Any]]:
    """Opens a new position in the specified basket asset (Paper or Live)."""
    trading_cfg = config.get("trading", {})
    cash = state.get("cash_usd", 50.0)

    # Position Sizing (e.g. 40% of cash or $20)
    if trading_cfg.get("dynamic_compounding", True):
        size_pct = float(trading_cfg.get("position_size_pct", 40.0)) / 100.0
        pos_size_usd = max(10.0, round(cash * size_pct, 2))
    else:
        pos_size_usd = float(trading_cfg.get("position_size_usd", 20.0))

    if cash < pos_size_usd or pos_size_usd < 5.0:
        return None

    entry_price = snapshot["price"]
    units = round(pos_size_usd / entry_price, 4)
    state["cash_usd"] -= pos_size_usd

    # Find token config in basket
    token_cfg = next((item for item in config.get("basket", []) if item["symbol"] == symbol), {})
    mint = token_cfg.get("mint", "")
    is_live = config.get("mode", "paper").lower() == "live"
    sol_cfg = config.get("solana", {})
    tx_sig = None

    if is_live and sol_cfg.get("private_key") and mint:
        usdc_mint = sol_cfg.get("usdc_mint", "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v")
        usdc_units = int(pos_size_usd * 1e6)
        ok, res = execute_jupiter_live_swap(
            input_mint=usdc_mint,
            output_mint=mint,
            amount_units=usdc_units,
            user_pubkey=sol_cfg.get("public_key"),
            private_key_b58=sol_cfg.get("private_key"),
            rpc_url=sol_cfg.get("rpc_url")
        )
        if not ok:
            logger.error("❌ [LIVE BUY FAILED] %s: %s", symbol, res)
            state["cash_usd"] += pos_size_usd
            return None
        tx_sig = res
        logger.info("🚀 [LIVE BUY BROADCAST] %s TX: https://solscan.io/tx/%s", symbol, tx_sig)

    tp_pct = float(trading_cfg.get("take_profit_pct", 1.20))
    sl_pct = float(trading_cfg.get("stop_loss_pct", 0.60))

    new_pos = {
        "symbol": symbol,
        "entry_time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "entry_time_ts": time.time(),
        "entry_price_usd": entry_price,
        "peak_price_usd": entry_price,
        "allocated_usd": pos_size_usd,
        "units": units,
        "target_tp_price": round(entry_price * (1.0 + tp_pct / 100.0), 6),
        "target_sl_price": round(entry_price * (1.0 - sl_pct / 100.0), 6),
        "reason": reason,
        "tx_sig": tx_sig,
    }

    state.setdefault("open_positions", {})[symbol] = new_pos
    state["last_trade_time"] = time.time()
    logger.info("🟢 [BUY] Bought %.4f %s at $%.4f ($%.2f allocated)", units, symbol, entry_price, pos_size_usd)
    return new_pos


def execute_sell_order(
    symbol: str,
    snapshot: Dict[str, Any], 
    state: Dict[str, Any], 
    config: Dict[str, Any], 
    outcome: str, 
    reason: str
) -> Dict[str, Any]:
    """Closes the active position for symbol, returning proceeds to cash."""
    pos = state["open_positions"].pop(symbol, None)
    if not pos:
        return {}

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

    token_cfg = next((item for item in config.get("basket", []) if item["symbol"] == symbol), {})
    mint = token_cfg.get("mint", "")
    decimals = int(token_cfg.get("decimals", 6))
    is_live = config.get("mode", "paper").lower() == "live"
    sol_cfg = config.get("solana", {})
    tx_sig = None

    if is_live and sol_cfg.get("private_key") and mint:
        token_units = int(pos["units"] * (10 ** decimals))
        usdc_mint = sol_cfg.get("usdc_mint", "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v")
        ok, res = execute_jupiter_live_swap(
            input_mint=mint,
            output_mint=usdc_mint,
            amount_units=token_units,
            user_pubkey=sol_cfg.get("public_key"),
            private_key_b58=sol_cfg.get("private_key"),
            rpc_url=sol_cfg.get("rpc_url")
        )
        if ok:
            tx_sig = res
            logger.info("🏁 [LIVE SELL BROADCAST] %s TX: https://solscan.io/tx/%s", symbol, tx_sig)
        else:
            logger.error("❌ [LIVE SELL FAILED] %s: %s", symbol, res)

    close_ev = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": symbol,
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
    state["last_trade_time"] = time.time()
    logger.info("🏁 [EXIT] %s Sold at $%.4f (%+.2f%%). Net PnL: %+.2f USD | %s", symbol, exit_price, gain_pct, net_pnl, reason)
    return close_ev
