$HostIP = "129.151.212.235"
$User = "ubuntu"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "🚀 AUTOMATED 24/7 CLOUD DEPLOYER (Oracle VPS: $HostIP)" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Locate SSH Key
$key = Get-ChildItem -Path "$HOME\Downloads\*.key" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $key) {
    Write-Host "No .key file found in Downloads." -ForegroundColor Yellow
    $keyPath = Read-Host "Please drag and drop your downloaded .key file here and press Enter"
    $keyPath = $keyPath.Trim('"').Trim("'")
} else {
    $keyPath = $key.FullName
    Write-Host "Found SSH Key: $keyPath" -ForegroundColor Green
}

# 2. Fix Key Permissions for Windows OpenSSH
Write-Host "Setting secure key permissions..." -ForegroundColor Gray
icacls $keyPath /inheritance:r /grant:r "$($env:USERNAME):(R)" > $null 2>&1

# 3. Test Connection
Write-Host "Testing SSH connection to $User@$HostIP..." -ForegroundColor Cyan
$sshOpt = "-o StrictHostKeyChecking=accept-new -o ConnectTimeout=10"
$testConn = ssh -i "$keyPath" $sshOpt "$User@$HostIP" "echo 'CONNECTED_OK'" 2>&1
if ($testConn -notcontains "CONNECTED_OK") {
    Write-Host "❌ Could not connect via SSH yet. The server might still be booting up." -ForegroundColor Red
    Write-Host "Error details: $testConn" -ForegroundColor Yellow
    exit 1
}
Write-Host "✅ Connected successfully to Ubuntu server!" -ForegroundColor Green

# 4. Prepare remote directories
Write-Host "Preparing remote directories on server..." -ForegroundColor Cyan
ssh -i "$keyPath" $sshOpt "$User@$HostIP" "mkdir -p ~/quantlab/sol_scalper ~/quantlab/logs"

# 5. Sync project files (including config.json and scripts)
Write-Host "Uploading codebase and configurations..." -ForegroundColor Cyan
$root = (Get-Item -Path "$PSScriptRoot\..").FullName

scp -i "$keyPath" $sshOpt -r "$root\sol_scalper\*" "$User@${HostIP}:~/quantlab/sol_scalper/"
scp -i "$keyPath" $sshOpt "$root\requirements.txt" "$User@${HostIP}:~/quantlab/"
scp -i "$keyPath" $sshOpt "$root\deploy_vps.sh" "$User@${HostIP}:~/quantlab/"

# 6. Run the 24/7 deployment script remotely
Write-Host "Running 24/7 cloud installer on VPS (installing Python, systemd service)..." -ForegroundColor Cyan
ssh -i "$keyPath" $sshOpt -t "$User@$HostIP" "cd ~/quantlab && chmod +x deploy_vps.sh && ./deploy_vps.sh"

Write-Host "==========================================================" -ForegroundColor Green
Write-Host "🎉 DEPLOYMENT COMPLETE! Bot is now running 24/7 on VPS!" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
