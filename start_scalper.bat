@echo off
title 15M Halal Quantitative Scalper (PYTH + RAY + SOL)
cd /d "%~dp0"
.\venv\Scripts\python.exe -m sol_scalper.bot_scalper --loop
pause
