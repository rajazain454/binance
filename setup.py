"""Interactive Control Center for QuantLab Binance Memecoin Rocket Engine.
Provides an easy CLI menu to configure API keys, run scans, inspect statistics, and launch the bot.
"""

import os
import sys
import subprocess
import json

PYTHON = sys.executable


def clear():
    os.system("cls" if os.name == "nt" else "clear")


from utils import load_config


def save_config(cfg):
    with open("config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def run_bot_scan():
    clear()
    print("=" * 68)
    print("  EXECUTING LIVE BINANCE TOP-10 MEMECOIN RADAR SCAN")
    print("=" * 68)
    cmd = [PYTHON, "bot.py", "--scan"]
    subprocess.run(cmd)
    input("\nPress Enter to return to main menu...")


def run_bot_loop():
    clear()
    print("=" * 68)
    print("  STARTING CONTINUOUS BINANCE MEMECOIN 24/7 TRADING LOOP")
    print("=" * 68)
    print("Press Ctrl+C at any time to safely stop the loop.\n")
    cmd = [PYTHON, "bot.py", "--loop"]
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nBot stopped by user.")
    input("\nPress Enter to return to main menu...")


def show_statistics():
    clear()
    print("=" * 68)
    print("  LEDGER PERFORMANCE & STATISTICAL EXPECTANCY REPORT")
    print("=" * 68)
    cmd = [PYTHON, "bot.py", "--stats"]
    subprocess.run(cmd)
    input("\nPress Enter to return to main menu...")


def configure_menu():
    clear()
    cfg = load_config()
    grid_cfg = cfg.get("grid_trading", {})
    risk_cfg = cfg.get("risk_management", {})
    print("=" * 68)
    print("  CONFIGURATION WIZARD")
    print("=" * 68)
    print(f"1. Trading Mode: {cfg.get('trading_mode', 'paper').upper()}")
    print(f"2. Timeframe: {cfg.get('timeframe', '15m')}")
    print(f"3. Capital (USDT): ${risk_cfg.get('capital_usdt', 100.0):,.2f}")
    print(f"4. Order Size (USDT): ${grid_cfg.get('order_size_usdt', 45.0):,.2f}")
    print(f"5. Target Profit: +{grid_cfg.get('target_tp_pct', 0.045)*100:.1f}%")
    print(f"6. Binance API Keys: {'CONFIGURED' if cfg.get('binance', {}).get('api_key') else 'NOT SET (PUBLIC MARKET DATA)'}")
    print(f"7. Discord Alerts: {'ENABLED' if cfg.get('discord', {}).get('enabled') else 'DISABLED'}")
    print("8. Test Discord Webhook Alert")
    print("9. Send Instant 24h Performance Digest to Discord")
    print("0. Return to Main Menu")
    print("-" * 68)

    choice = input("Select option to modify (0-9): ").strip()
    if choice == "1":
        mode = input("Enter trading mode (paper / live): ").strip().lower()
        if mode in ["paper", "live"]:
            cfg["trading_mode"] = mode
            save_config(cfg)
            print(f"Trading mode updated to {mode.upper()}.")
    elif choice == "2":
        tf = input("Enter timeframe (15m, 1h, 4h) [default: 15m]: ").strip().lower() or "15m"
        cfg["timeframe"] = tf
        save_config(cfg)
    elif choice == "3":
        cap = input("Enter trading capital in USDT (e.g. 100.0): ").strip()
        try:
            cfg.setdefault("risk_management", {})["capital_usdt"] = float(cap)
            save_config(cfg)
        except ValueError:
            print("Invalid number.")
    elif choice == "4":
        size = input("Enter order size in USDT (e.g. 45.0): ").strip()
        try:
            cfg.setdefault("grid_trading", {})["order_size_usdt"] = float(size)
            save_config(cfg)
        except ValueError:
            print("Invalid number.")
    elif choice == "5":
        tp = input("Enter baseline target TP (e.g. 0.045 for +4.5%): ").strip()
        try:
            cfg.setdefault("grid_trading", {})["target_tp_pct"] = float(tp)
            save_config(cfg)
        except ValueError:
            print("Invalid number.")
    elif choice == "6":
        key = input("Enter Binance API Key: ").strip()
        sec = input("Enter Binance API Secret: ").strip()
        cfg.setdefault("binance", {})["api_key"] = key
        cfg["binance"]["api_secret"] = sec
        save_config(cfg)
        print("Binance credentials updated.")
    elif choice == "7":
        en = input("Enable Discord webhook alerts? (y/n): ").strip().lower() == "y"
        cfg.setdefault("discord", {})["enabled"] = en
        if en:
            url = input("Paste your Discord Webhook URL: ").strip()
            cfg["discord"]["webhook_url"] = url
        save_config(cfg)
        print("Discord settings updated.")
    elif choice == "8":
        print("\nSending test alert to Discord Webhook...")
        subprocess.run([PYTHON, "bot.py", "--test-discord"])
    elif choice == "9":
        print("\nSending 24h Performance Digest to Discord...")
        subprocess.run([PYTHON, "bot.py", "--digest"])
    input("\nPress Enter to continue...")


def reset_state():
    state_file = "bot_state.json"
    clean_state = {
        "balance_usdt": 100.0,
        "peak_balance": 100.0,
        "daily_peak_balance": 100.0,
        "daily_reset_time": "2026-10-14T00:00:00.000000+00:00",
        "circuit_breaker_until": None,
        "open_positions": {},
        "loss_cooldowns": {},
        "trade_count": 0,
        "win_count": 0,
        "loss_count": 0,
        "realized_pnl_usdt": 0.0,
        "last_update": "2026-10-14 00:00:00",
        "last_digest_date": "2026-10-14",
        "last_hourly_alert": None,
        "macro_regime": "MACRO BULL TREND (BTC > 200d SMA)",
        "is_bear_defense": False,
        "grid_state": {
            "active_symbol": "PEPE/USDT",
            "active_symbols": [],
            "total_grid_profit_usdt": 0.0,
            "cycles_completed": 0,
            "grid_trades": 0,
        },
    }
    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(clean_state, f, indent=2)
    print(f"\nSuccessfully reset {state_file} to fresh $100.00 cash balance.")
    input("\nPress Enter to return...")


def run_unit_tests():
    clear()
    print("=" * 68)
    print("  RUNNING 50-POINT COMPREHENSIVE TEST SUITE")
    print("=" * 68)
    cmd = [PYTHON, "-m", "pytest"]
    subprocess.run(cmd)
    input("\nPress Enter to return to main menu...")


def main():
    while True:
        clear()
        print("=" * 68)
        print("   BINANCE TOP-10 MEMECOIN ROCKET ENGINE - CONTROL CENTER")
        print("=" * 68)
        print("  1. Run Live Market Radar Scan (Scan & Execute Signals)")
        print("  2. Launch Continuous 24/7 Trading Loop")
        print("  3. View Closed Trade Ledger & Statistical Expectancy")
        print("  4. Settings (Capital, $45 Sizing, Discord, API Keys)")
        print("  5. Reset Active Bot State & Cash Balance ($100)")
        print("  6. Run Full 50-Point Automated Test Suite")
        print("  7. Exit")
        print("=" * 68)
        choice = input("\nEnter choice [1-7]: ").strip()

        if choice == "1":
            run_bot_scan()
        elif choice == "2":
            run_bot_loop()
        elif choice == "3":
            show_statistics()
        elif choice == "4":
            configure_menu()
        elif choice == "5":
            reset_state()
        elif choice == "6":
            run_unit_tests()
        elif choice == "7":
            print("\nExiting. May your scalps be profitable!\n")
            break


if __name__ == "__main__":
    main()
