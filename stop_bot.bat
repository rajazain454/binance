@echo off
echo Stopping Solana Sniper background process...
powershell -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*solana_sniper*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force; Write-Host 'Stopped PID:' $_.ProcessId }"
echo Sniper safely stopped.
pause
