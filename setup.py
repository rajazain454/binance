"""Interactive Setup & Training Wizard for Binance AI Trading Bot.
Provides an easy menu to train models, run backtests, configure API keys, and launch the bot.
"""

import os
import sys
import subprocess
import json

PYTHON = sys.executable


def clear():
    os.system("cls" if os.name == "nt" else "clear")


def load_config():
    with open("config.json", "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(cfg):
    with open("config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def train_menu():
    clear()
    print("=" * 65)
    print("  TRAIN AI MODEL ON BINANCE MARKET DATA")
    print("=" * 65)
    days = input("Enter days of historical Binance data to fetch (default: 120): ").strip() or "120"
    tf = input("Enter candle timeframe [1h, 4h, 1d] (default: 1h): ").strip() or "1h"
    cmd = [PYTHON, "train.py", "--days", days, "--timeframe", tf]
    print(f"\nExecuting: {' '.join(cmd)}\n")
    subprocess.run(cmd)
    input("\nPress Enter to return to main menu...")


def backtest_menu():
    clear()
    print("=" * 65)
    print("  WALK-FORWARD STRATEGY BACKTEST")
    print("=" * 65)
    sym = input("Enter symbol to backtest (e.g. BTC/USDT, ETH/USDT, SOL/USDT) [default: BTC/USDT]: ").strip() or "BTC/USDT"
    cmd = [PYTHON, "backtest.py", "--symbol", sym]
    print(f"\nExecuting: {' '.join(cmd)}\n")
    subprocess.run(cmd)
    input("\nPress Enter to return to main menu...")


def run_bot_scan():
    clear()
    print("=" * 65)
    print("  EXECUTING BINANCE LIVE MARKET SCAN")
    print("=" * 65)
    cmd = [PYTHON, "bot.py", "--scan"]
    subprocess.run(cmd)
    input("\nPress Enter to return to main menu...")


def run_bot_loop():
    clear()
    print("=" * 65)
    print("  STARTING CONTINUOUS BINANCE AI TRADING LOOP")
    print("=" * 65)
    print("Press Ctrl+C at any time to stop the bot.\n")
    cmd = [PYTHON, "bot.py", "--loop"]
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nBot stopped by user.")
    input("\nPress Enter to return to main menu...")


def configure_menu():
    clear()
    cfg = load_config()
    print("=" * 65)
    print("  CONFIGURATION WIZARD")
    print("=" * 65)
    print(f"1. Trading Mode: {cfg.get('trading_mode', 'paper').upper()}")
    print(f"2. Timeframe: {cfg.get('timeframe', '1h')}")
    print(f"3. Capital (USDT): ${cfg['risk_management']['capital_usdt']:,.2f}")
    print(f"4. AI Confidence Threshold: {cfg['ai_model']['confidence_threshold']:.0%}")
    print(f"5. Binance API Keys: {'CONFIGURED' if cfg['binance']['api_key'] else 'NOT SET (PUBLIC DATA ONLY)'}")
    print(f"6. Discord Alerts: {'ENABLED' if cfg.get('discord', {}).get('enabled') else 'DISABLED'}")
    print("7. Test Discord Webhook Alert")
    print("8. Send Instant 24h Performance Digest to Discord")
    print("9. Return to Main Menu")
    print("-" * 65)

    choice = input("Select option to modify (1-9): ").strip()
    if choice == "1":
        mode = input("Enter trading mode (paper / live): ").strip().lower()
        if mode in ["paper", "live"]:
            cfg["trading_mode"] = mode
            save_config(cfg)
            print(f"Trading mode updated to {mode.upper()}.")
    elif choice == "2":
        tf = input("Enter timeframe (1h, 4h, 1d): ").strip().lower()
        if tf in ["1h", "4h", "1d"]:
            cfg["timeframe"] = tf
            save_config(cfg)
    elif choice == "3":
        cap = input("Enter trading capital in USDT (e.g. 5000): ").strip()
        try:
            cfg["risk_management"]["capital_usdt"] = float(cap)
            save_config(cfg)
        except ValueError:
            print("Invalid number.")
    elif choice == "4":
        thresh = input("Enter confidence threshold (0.50 to 0.80) [e.g. 0.65]: ").strip()
        try:
            cfg["ai_model"]["confidence_threshold"] = float(thresh)
            save_config(cfg)
        except ValueError:
            print("Invalid number.")
    elif choice == "5":
        key = input("Enter Binance API Key: ").strip()
        sec = input("Enter Binance API Secret: ").strip()
        cfg["binance"]["api_key"] = key
        cfg["binance"]["api_secret"] = sec
        save_config(cfg)
        print("Binance credentials updated.")
    elif choice == "6":
        en = input("Enable Discord webhook alerts? (y/n): ").strip().lower() == "y"
        if "discord" not in cfg:
            cfg["discord"] = {}
        cfg["discord"]["enabled"] = en
        if en:
            url = input("Paste your Discord Webhook URL: ").strip()
            cfg["discord"]["webhook_url"] = url
        save_config(cfg)
        print("Discord settings updated.")
    elif choice == "7":
        print("\nSending test message to Discord Webhook...")
        subprocess.run([PYTHON, "bot.py", "--test-discord"])
    elif choice == "8":
        print("\nSending 24h Performance Digest to Discord...")
        subprocess.run([PYTHON, "bot.py", "--digest"])
    input("\nPress Enter to continue...")


def reset_state():
    state_file = "bot_state.json"
    if os.path.exists(state_file):
        os.remove(state_file)
        print(f"Cleared {state_file}. All simulated positions reset.")
    else:
        print("No active state file found.")
    input("\nPress Enter to return...")


def main():
    while True:
        clear()
        print("=" * 65)
        print("      BINANCE QUANTITATIVE AI TRADING BOT - CONTROL CENTER")
        print("=" * 65)
        print("  1. Train AI Model on Binance Historical Data")
        print("  2. Run Backtest & Verify Strategy Win Rate")
        print("  3. Run Live Market Scan (Execute Trades / Signals)")
        print("  4. Launch Continuous Auto-Trading Loop")
        print("  5. Settings (Capital, Risk, API Keys, Discord)")
        print("  6. Reset Active Bot State & Positions")
        print("  7. Run 20-Point Hard Stress-Test Suite")
        print("  8. Exit")
        print("=" * 65)
        choice = input("\nEnter choice [1-8]: ").strip()

        if choice == "1":
            train_menu()
        elif choice == "2":
            backtest_menu()
        elif choice == "3":
            run_bot_scan()
        elif choice == "4":
            run_bot_loop()
        elif choice == "5":
            configure_menu()
        elif choice == "6":
            reset_state()
        elif choice == "7":
            clear()
            print("=" * 65)
            print("  EXECUTING 20-POINT HARD STRESS-TEST SUITE")
            print("=" * 65)
            subprocess.run([PYTHON, "tests/hard_test.py"])
            input("\nPress Enter to return to main menu...")
        elif choice == "8":
            print("\nExiting. Good luck trading!\n")
            break


if __name__ == "__main__":
    main()
