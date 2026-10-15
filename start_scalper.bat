@echo off
title Multi-Asset (JUP+SOL) Quantitative Scalper
cd /d "%~dp0"
.\venv\Scripts\python.exe -m sol_scalper.bot_scalper --loop
pause
