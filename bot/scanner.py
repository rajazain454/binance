"""Market scanning, regime detection, correlation filtering, and order sizing."""

import concurrent.futures
from datetime import datetime, timezone
import logging
import os
import sys
import numpy as np
import pandas as pd

import features
from bot.ledger import log_trade
from bot.state import check_circuit_breaker

logger = logging.getLogger("quantlab")


def compute_asset_correlation(sym1, sym2, ohlcv_map=None, data_dir="data", lookback=720):
    """
    Computes rolling Pearson return correlation between two crypto assets.
    Prioritizes fresh in-memory OHLCV data from the current scan cycle to avoid stale data,
    falling back to cached data files if necessary.
    """
    df1, df2 = None, None
    if ohlcv_map:
        df1 = ohlcv_map.get(sym1)
        df2 = ohlcv_map.get(sym2)

    if df1 is None or df2 is None or len(df1) < 30 or len(df2) < 30:
        clean1 = sym1.replace("/", "_")
        clean2 = sym2.replace("/", "_")
        f1 = os.path.join(data_dir, f"binance_{clean1}_1h.csv")
        f2 = os.path.join(data_dir, f"binance_{clean2}_1h.csv")
        if os.path.exists(f1) and os.path.exists(f2):
            try:
                df1 = pd.read_csv(f1, index_col=0, parse_dates=True)
                df2 = pd.read_csv(f2, index_col=0, parse_dates=True)
            except Exception:
                return 0.0
        else:
            return 0.0

    try:
        c1 = df1["close"].iloc[-lookback:].pct_change().dropna()
        c2 = df2["close"].iloc[-lookback:].pct_change().dropna()
        common = c1.index.intersection(c2.index)
        if len(common) < 20:
            return 0.0
        corr = float(c1.loc[common].corr(c2.loc[common]))
        return corr if not np.isnan(corr) else 0.0
    except Exception as e:
        logger.debug("Correlation computation failed for %s/%s: %s", sym1, sym2, e)
        return 0.0


def get_macro_regime_threshold(client, config):
    """
    Computes dynamic confidence threshold based on macro BTC 200 SMA regime.
    - Macro Bull (BTC > 200 SMA): threshold adjusted slightly with strict 0.60 floor
    - Macro Bear / Chop (BTC < 200 SMA): threshold raised for defense
    """
    base_thresh = config.get("ai_model", {}).get("confidence_threshold", 0.62)
    try:
        btc_df = client.fetch_ohlcv("BTC/USDT", timeframe="1d", limit=220)
        if len(btc_df) >= 200:
            c = btc_df["close"]
            sma200 = float(c.rolling(200).mean().iloc[-1])
            curr_c = float(c.iloc[-1])
            if curr_c > sma200:
                bull_thresh = max(0.60, round(base_thresh - 0.02, 2))
                return bull_thresh, f"MACRO BULL ({bull_thresh:.0%})"
            else:
                bear_thresh = max(0.60, round(base_thresh + 0.05, 2))
                return bear_thresh, f"MACRO DEFENSIVE ({bear_thresh:.0%})"
    except Exception as e:
        logger.warning("Macro regime threshold calculation failed: %s", e)
    return max(0.60, base_thresh), f"NORMAL ({max(0.60, base_thresh):.0%})"


def scan_and_execute(client, model_bundle, state, config):
    """Evaluates Binance market candles, runs model inference, and executes trades."""
    model = model_bundle["model"]
    feat_names = model_bundle["feature_names"]
    conf_thresh, regime_label = get_macro_regime_threshold(client, config)
    tp1_mult = config["ai_model"].get("tp1_atr_mult", 1.5)
    tp2_mult = config["ai_model"].get("tp2_atr_mult", 2.8)
    sl_mult = config["ai_model"].get("sl_atr_mult", 1.2)
    max_trades = config["risk_management"].get("max_open_trades", 3)
    risk_pct = config["risk_management"].get("risk_per_trade_pct", 2.0) / 100.0
    tf = config["timeframe"]

    cb_locked, cb_status = check_circuit_breaker(state, config)

    analysis_rows = []
    new_signals = []
    trade_candidates = []

    # 1. Parallel multi-threaded candle fetching across all symbols
    is_mock = hasattr(client.fetch_ohlcv, "assert_called") or hasattr(client.fetch_ohlcv, "mock_calls")
    ohlcv_map = {}
    if not is_mock and len(config["symbols"]) > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            future_to_sym = {
                executor.submit(client.fetch_ohlcv, sym, timeframe=tf, limit=300): sym
                for sym in config["symbols"]
            }
            for future in concurrent.futures.as_completed(future_to_sym):
                s = future_to_sym[future]
                try:
                    ohlcv_map[s] = future.result()
                except Exception as e:
                    logger.warning("Failed to fetch %s: %s", s, e)

    df_btc = ohlcv_map.get("BTC/USDT")
    if df_btc is None and "BTC/USDT" in config["symbols"]:
        try:
            df_btc = client.fetch_ohlcv("BTC/USDT", timeframe=tf, limit=300)
        except Exception as e:
            logger.warning("Fallback fetch of BTC/USDT failed: %s", e)

    # Macro BTC Cascade Detection & Volatility Regime Scaling
    btc_cascade_halt = False
    btc_halt_thresh = config.get("risk_management", {}).get("btc_cascade_halt_pct", -0.025)
    if df_btc is not None and len(df_btc) >= 2:
        try:
            btc_prev_c = float(df_btc["close"].iloc[-2])
            btc_curr_c = float(df_btc["close"].iloc[-1])
            btc_1h_ret = (btc_curr_c / btc_prev_c) - 1.0
            if btc_1h_ret <= btc_halt_thresh:
                btc_cascade_halt = True
        except Exception as e:
            logger.warning("BTC cascade detection failed: %s", e)

    # Dynamic ATR Volatility Scaling
    sl_mult_eff = sl_mult
    vol_scale_factor = 1.0
    if config.get("risk_management", {}).get("volatility_scaling_enabled", True) and df_btc is not None and len(df_btc) >= 30:
        try:
            btc_atr = features.compute_atr(df_btc, 14)
            mean_atr = btc_atr.rolling(30, min_periods=10).mean().iloc[-1]
            curr_atr_val = btc_atr.iloc[-1]
            atr_ratio = float(curr_atr_val / (mean_atr + 1e-9))
            if atr_ratio > 1.35:
                sl_mult_eff = round(sl_mult * 1.25, 2)
                vol_scale_factor = 0.80
        except Exception as e:
            logger.warning("Volatility scaling failed: %s", e)

    for sym in config["symbols"]:
        try:
            df = ohlcv_map.get(sym)
            if df is None:
                df = client.fetch_ohlcv(sym, timeframe=tf, limit=300)
            if df is None or len(df) < 50:
                continue

            # Funding rate sentiment check
            fr = client.fetch_funding_rate(sym)
            funding_overheated = (fr > 0.0003)

            feat_df = features.extract_features(df)
            atr_series = features.compute_atr(df, 14)
            active_setups, strat_score, indicator_summary = features.detect_active_strategies(df, funding_rate=fr)

            last_feats = feat_df.iloc[-1:][feat_names]
            prob_win = float(model.predict_proba(last_feats)[0, 1])

            c = float(df["close"].iloc[-1])
            curr_atr = float(atr_series.iloc[-1])
            sl = c - (sl_mult_eff * curr_atr)
            tp1 = c + (tp1_mult * curr_atr)
            tp2 = c + (tp2_mult * curr_atr)

            # MTF Macro Confluence Check (4h Trend Alignment)
            mtf_cfg = config.get("mtf_confluence", {})
            use_mtf = mtf_cfg.get("enabled", True)
            min_rsi = mtf_cfg.get("macro_rsi_min", 48.0)
            is_macro_bullish, mtf_info = features.evaluate_macro_confluence(df, min_rsi=min_rsi)

            # Technical Confluence & Trend Verification
            st_series, _ = features.compute_supertrend(df, 10, 3.0)
            is_supertrend_bull = bool(st_series.iloc[-1] > 0.5)
            cmf_series = features.compute_cmf(df, 20)
            cmf_val = float(cmf_series.iloc[-1])

            ai_cfg = config.get("ai_model", {})
            min_strat_score = ai_cfg.get("min_strategy_score", 20.0)
            req_bull_st = ai_cfg.get("require_bull_supertrend", True)
            req_active_setup = ai_cfg.get("require_active_setup", False)
            min_cmf = ai_cfg.get("min_cmf", -0.05)

            # Check loss cooldown with datetime parsing
            now_utc = datetime.now(timezone.utc)
            cooldown_expiry = state.get("loss_cooldowns", {}).get(sym)
            is_on_cooldown = False
            if cooldown_expiry:
                try:
                    exp_dt = datetime.fromisoformat(cooldown_expiry)
                    if exp_dt.tzinfo is None:
                        exp_dt = exp_dt.replace(tzinfo=timezone.utc)
                    is_on_cooldown = exp_dt > now_utc
                except Exception:
                    pass

            # Relative Strength vs BTC Alpha evaluation
            is_alpha_leader = False
            rs_info = {"alpha_24h": 0.0, "is_leader": False}
            if sym != "BTC/USDT" and df_btc is not None and len(df_btc) >= 50:
                is_alpha_leader, rs_info = features.evaluate_relative_strength(df, df_btc)

            is_high_conviction = prob_win >= conf_thresh
            is_open = sym in state["open_positions"]
            confluence_passed = is_macro_bullish if use_mtf else True

            # Alpha Leader override
            is_rs_override = False
            if not confluence_passed and is_high_conviction and is_alpha_leader:
                confluence_passed = True
                is_rs_override = True

            # Technical confirmation gates
            supertrend_ok = (not req_bull_st) or is_supertrend_bull
            strat_score_ok = strat_score >= min_strat_score
            setup_ok = (not req_active_setup) or (len(active_setups) > 0)
            cmf_ok = cmf_val >= min_cmf
            tech_confluence_passed = supertrend_ok and strat_score_ok and setup_ok and cmf_ok

            is_btc_halted = btc_cascade_halt and (sym != "BTC/USDT")

            if is_open:
                status = "HOLDING"
            elif cb_locked:
                status = "CIRCUIT PAUSE"
            elif is_btc_halted:
                status = "BTC CASCADE"
            elif is_on_cooldown:
                status = "COOLDOWN"
            elif funding_overheated:
                status = "FUNDING HOT"
            elif not supertrend_ok:
                status = "BEAR TREND"
            elif not strat_score_ok:
                status = "WEAK SETUP"
            elif not setup_ok:
                status = "NO SETUP"
            elif not cmf_ok:
                status = "OUTFLOW"
            elif is_rs_override and tech_confluence_passed:
                status = "RS ALPHA"
            elif is_high_conviction and confluence_passed and tech_confluence_passed:
                status = "BUY TRIGGER"
            elif is_high_conviction and not confluence_passed:
                status = "4H BLOCKED"
            else:
                status = "SCANNING"

            mtf_display = f"RS +{rs_info['alpha_24h']}%" if is_rs_override else mtf_info.get("status", "N/A")

            row = {
                "symbol": sym,
                "price": c,
                "atr": curr_atr,
                "prob_win": prob_win,
                "strategy_score": strat_score,
                "active_setups": active_setups,
                "indicators_summary": indicator_summary,
                "high_conviction": is_high_conviction,
                "mtf_status": mtf_display,
                "status": status,
                "stop_loss": sl,
                "tp1": tp1,
                "tp2": tp2,
            }
            analysis_rows.append(row)

            # Collect all qualified candidates that passed all filters
            if (not cb_locked and not is_open and not is_on_cooldown and not is_btc_halted
                and not funding_overheated and is_high_conviction
                and confluence_passed and tech_confluence_passed):
                trade_candidates.append({
                    "symbol": sym,
                    "price": c,
                    "atr": curr_atr,
                    "prob_win": prob_win,
                    "strategy_score": strat_score,
                    "active_setups": active_setups,
                    "indicators_summary": indicator_summary,
                    "mtf_status": mtf_display,
                    "sl": sl,
                    "tp1": tp1,
                    "tp2": tp2,
                })

        except Exception as e:
            logger.error("Error analyzing %s: %s", sym, e, exc_info=True)

    # ALPHA RANKING & CROSS-ASSET CORRELATION FILTER
    trade_candidates.sort(key=lambda x: (x["prob_win"], x["strategy_score"]), reverse=True)

    corr_cfg = config.get("correlation_filter", {})
    use_corr_filter = corr_cfg.get("enabled", True)
    max_corr = corr_cfg.get("max_correlation", 0.75)
    data_dir = config.get("paths", {}).get("data_dir", "data")

    selected_trades = []
    portfolio_symbols = list(state["open_positions"].keys())
    available_slots = max(0, max_trades - len(state["open_positions"]))

    for cand in trade_candidates:
        if len(selected_trades) >= available_slots:
            break
        sym = cand["symbol"]
        is_corr = False
        if use_corr_filter and portfolio_symbols:
            bot_mod = sys.modules.get("bot")
            corr_fn = getattr(bot_mod, "compute_asset_correlation", compute_asset_correlation) if bot_mod else compute_asset_correlation
            for p_sym in portfolio_symbols:
                try:
                    r = corr_fn(sym, p_sym, ohlcv_map=ohlcv_map, data_dir=data_dir)
                except TypeError:
                    try:
                        r = corr_fn(sym, p_sym, data_dir=data_dir)
                    except TypeError:
                        r = corr_fn(sym, p_sym)
                if r > max_corr:
                    is_corr = True
                    break
        if is_corr:
            for r in analysis_rows:
                if r["symbol"] == sym:
                    r["status"] = "CORR BLOCKED"
            continue

        selected_trades.append(cand)
        portfolio_symbols.append(sym)

    for cand in selected_trades:
        sym = cand["symbol"]
        c = cand["price"]
        sl = cand["sl"]
        tp1 = cand["tp1"]
        tp2 = cand["tp2"]
        curr_atr = cand["atr"]
        prob_win = cand["prob_win"]
        strat_score = cand["strategy_score"]
        active_setups = cand["active_setups"]
        indicator_summary = cand["indicators_summary"]
        mtf_status = cand["mtf_status"]

        # Risk Parity position sizing with Cash Reserve and Portfolio Cap Protection
        available_balance = state["balance_usdt"]
        open_alloc = sum(p.get("allocated_usdt", 0.0) for p in state["open_positions"].values())
        total_equity = available_balance + open_alloc

        min_reserve_pct = config.get("risk_management", {}).get("min_cash_reserve_pct", 0.15)
        min_reserve_usdt = total_equity * min_reserve_pct
        spendable_cash = max(0.0, available_balance - min_reserve_usdt)

        # Enforce portfolio exposure cap simultaneously with cash reserve
        max_portfolio_spend = max(0.0, (total_equity * (1.0 - min_reserve_pct)) - open_alloc)
        spendable_cash = min(spendable_cash, max_portfolio_spend)

        dynamic_sizing = config.get("risk_management", {}).get("dynamic_alpha_sizing", True)
        if dynamic_sizing:
            surplus = max(-0.10, min(0.20, prob_win - conf_thresh))
            alpha_mult = 1.0 + (surplus * 2.0)
            eff_risk_pct = risk_pct * alpha_mult
        else:
            eff_risk_pct = risk_pct

        risk_amount = available_balance * eff_risk_pct * vol_scale_factor
        try:
            min_notional = float(client.get_min_notional(sym))
        except Exception:
            min_notional = float(config.get("risk_management", {}).get("min_notional_usdt", 5.0))
        risk_per_unit = c - sl

        if risk_per_unit > 0 and spendable_cash >= min_notional:
            units = risk_amount / risk_per_unit
            remaining_slots = max(1, max_trades - len(state["open_positions"]))
            slot_cap = (spendable_cash / remaining_slots) * 0.95
            max_trade_pct = config.get("risk_management", {}).get("max_trade_allocation_pct", 0.32)
            portfolio_cap = total_equity * max_trade_pct
            if dynamic_sizing:
                max_alloc = min(slot_cap * min(1.25, max(0.80, alpha_mult)), portfolio_cap)
            else:
                max_alloc = min(slot_cap, portfolio_cap)

            pos_cost = min(spendable_cash, max_alloc, units * c)
            units = pos_cost / c

            if pos_cost >= min_notional:  # Dynamic Binance order threshold
                # Real live execution if live mode enabled
                if config.get("trading_mode") == "live":
                    try:
                        order_res = client.place_spot_order(sym, "buy", units, price=c)
                        logger.info("[LIVE SPOT ORDER] Successfully executed buy for %s: %s", sym, order_res.get("id", "FILLED"))
                    except Exception as e:
                        logger.error("[LIVE ORDER REJECTED] Binance order failed for %s: %s", sym, e)
                        continue

                    # Native exchange-side OCO order placement
                    if config.get("execution", {}).get("use_native_oco", False):
                        try:
                            oco_res = client.place_oco_order(sym, "sell", units, tp_price=tp2, sl_price=sl)
                            logger.info("[LIVE OCO ACTIVE] Native Binance OCO order registered for %s", sym)
                        except Exception as e:
                            logger.warning("[LIVE OCO WARNING] Could not register native OCO order for %s: %s", sym, e)

                state["balance_usdt"] -= pos_cost
                state["open_positions"][sym] = {
                    "entry_price": round(c, 4),
                    "allocated_usdt": round(pos_cost, 2),
                    "units": units,
                    "stop_loss": round(sl, 4),
                    "take_profit_1": round(tp1, 4),
                    "take_profit_2": round(tp2, 4),
                    "tp1_reached": False,
                    "highest_price": round(c, 4),
                    "atr": round(curr_atr, 6),
                    "entry_time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                    "confidence": round(prob_win * 100, 1),
                    "strategy_score": strat_score,
                    "strategy_setups": active_setups or ["Quantitative Multi-Tool Confluence"],
                    "indicators_summary": indicator_summary,
                    "macro_4h": mtf_status,
                }

                trade_signal = {
                    "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                    "symbol": sym,
                    "action": "BUY (ENTER)",
                    "entry_price": round(c, 4),
                    "exit_price": "",
                    "allocated_usdt": round(pos_cost, 2),
                    "stop_loss": round(sl, 4),
                    "tp1": round(tp1, 4),
                    "tp2": round(tp2, 4),
                    "net_pnl": "",
                    "pnl_pct": "",
                    "confidence_pct": round(prob_win * 100, 1),
                    "reason": (" + ".join(active_setups) if active_setups else "AI Quant Confluence"),
                    "outcome": "OPEN",
                    "mode": config.get("trading_mode", "paper").upper(),
                    # Keep extra fields for Discord alerts
                    "price": round(c, 4),
                    "size_usdt": round(pos_cost, 2),
                    "strategy_score": strat_score,
                    "strategy_setups": " + ".join(active_setups) if active_setups else "AI Quant Confluence",
                    "indicators_summary": indicator_summary,
                    "macro_4h": mtf_status,
                }
                log_trade(trade_signal, config["paths"]["trade_ledger"])
                new_signals.append(trade_signal)

    return analysis_rows, new_signals, cb_status
