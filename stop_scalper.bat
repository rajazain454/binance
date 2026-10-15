@echo off
echo Stopping SOL/USDC Scalper background process...
powershell -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*sol_scalper*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force; Write-Host 'Stopped PID:' $_.ProcessId }"
echo Scalper safely stopped.
pause
