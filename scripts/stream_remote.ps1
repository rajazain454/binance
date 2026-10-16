$HostIP = "129.151.212.235"
$User = "ubuntu"
$key = Get-ChildItem -Path "$HOME\Downloads\*.key" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$keyPath = $key.FullName
$sshArgs = @("-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=10")

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "📡 STREAMING LIVE 24/7 CLOUD BOT EXECUTION (Ctrl+C to stop)" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

ssh -i $keyPath @sshArgs -t "$User@$HostIP" "journalctl -u sol-scalper -f -n 25"
