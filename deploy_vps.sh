#!/usr/bin/env bash
set -e

echo "=================================================="
echo "⚡ SOL/USDC Systematic Scalper 24/7 Cloud Installer"
echo "=================================================="

# Update and install Python prerequisites
sudo apt update && sudo apt install -y python3-pip python3-venv git curl

# Create virtual environment
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Create systemd 24/7 background service
SERVICE_PATH="/etc/systemd/system/sol-scalper.service"
CURRENT_DIR=$(pwd)

echo "Setting up 24/7 systemd background daemon..."
sudo bash -c "cat <<EOF > $SERVICE_PATH
[Unit]
Description=SOL/USDC Quantitative Institutional Scalper
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$CURRENT_DIR
ExecStart=$CURRENT_DIR/venv/bin/python3 -m sol_scalper.bot_scalper --loop
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF"

sudo systemctl daemon-reload
sudo systemctl enable sol-scalper
sudo systemctl restart sol-scalper

echo "=================================================="
echo "✅ SOL/USDC Scalper is now running 24/7 in the cloud!"
echo "Check status anytime with: sudo systemctl status sol-scalper"
echo "Check live logs with: journalctl -u sol-scalper -f"
echo "=================================================="
