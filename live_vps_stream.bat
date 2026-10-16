@echo off
title 📡 Live 24/7 Cloud Bot Stream (129.151.212.235)
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\stream_remote.ps1"
pause
