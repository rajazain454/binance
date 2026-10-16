@echo off
title Deploy Bot to Oracle Cloud VPS (129.151.212.235)
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\deploy_remote.ps1"
pause
