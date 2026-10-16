@echo off
title Check Oracle Cloud VPS Bot Status (129.151.212.235)
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\status_remote.ps1"
pause
