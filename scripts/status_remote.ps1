$HostIP = "129.151.212.235"
$User = "ubuntu"
$key = Get-ChildItem -Path "$HOME\Downloads\*.key" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$keyPath = $key.FullName
$sshArgs = @("-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=10")

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "⚡ CHECKING LIVE STATUS ON ORACLE CLOUD VPS" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

ssh -i $keyPath @sshArgs "$User@$HostIP" "sudo systemctl status sol-scalper --no-pager && echo '--- RECENT LOGS ---' && journalctl -u sol-scalper -n 12 --no-pager"
