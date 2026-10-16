$HostIP = '129.151.212.235'
$User = 'ubuntu'
$key = Get-ChildItem -Path "$HOME\Downloads\*.key" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$keyPath = $key.FullName
$target = "$User@$HostIP"
$cmd = 'journalctl -u sol-scalper -f -n 25'

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "STREAMING LIVE 24/7 CLOUD BOT EXECUTION (Ctrl+C to stop)" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

ssh -i $keyPath -o StrictHostKeyChecking=accept-new -t $target $cmd
