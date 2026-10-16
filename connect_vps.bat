@echo off
title Connect to Oracle Cloud VPS (129.151.212.235)
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$key = Get-ChildItem -Path $HOME\Downloads\*.key | Sort-Object LastWriteTime -Descending | Select-Object -First 1; ^
   if (-not $key) { Write-Host 'No .key file found in Downloads. Please drag and drop your .key file here: ' -ForegroundColor Yellow; $keyPath = Read-Host } else { $keyPath = $key.FullName }; ^
   Write-Host \"Using SSH Key: $keyPath\" -ForegroundColor Cyan; ^
   Write-Host 'Connecting to ubuntu@129.151.212.235...' -ForegroundColor Green; ^
   icacls $keyPath /inheritance:r /grant:r \"$($env:USERNAME):(R)\" > $null 2>&1; ^
   ssh -i $keyPath -o StrictHostKeyChecking=accept-new ubuntu@129.151.212.235"
pause
