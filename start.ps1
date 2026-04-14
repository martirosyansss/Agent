#Requires -Version 5.1
# Sentinel v2 Launcher - Advanced startup with health checks and configuration validation

$ErrorActionPreference = "SilentlyContinue"

# CONFIG
$PidFile  = "$PSScriptRoot\sentinel\sentinel.pid"
$EnvFile  = "$PSScriptRoot\sentinel\.env"
$Python   = "C:/Users/marti/AppData/Local/Programs/Python/Python312/python.exe"
$BackPort = 8080
$LogDir   = "$PSScriptRoot\sentinel\logs"

# Try to read port from .env
if (Test-Path $EnvFile) {
    $line = Get-Content $EnvFile | Where-Object { $_ -match '^\s*DASHBOARD_PORT\s*=\s*\d+\s*$' } | Select-Object -First 1
    if ($line -match '(\d+)') {
        $BackPort = [int]$Matches[1]
    }
}

# Create log directory
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir -Force | Out-Null }

# Kill old process by PID file
if (Test-Path $PidFile) {
    $oldPid = Get-Content $PidFile -Raw
    if ($oldPid -match '^\d+$') {
        Write-Host "Stopping previous process PID=$($oldPid.Trim())..." -ForegroundColor Yellow
        Stop-Process -Id ([int]$oldPid.Trim()) -Force -ErrorAction SilentlyContinue
    }
    Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
}

# Kill processes on ports
$ports = @($BackPort, 8080, 8888) | Select-Object -Unique
foreach ($port in $ports) {
    $proc = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    if ($proc) {
        Write-Host "Closing process on port $port (PID=$($proc.OwningProcess))..." -ForegroundColor Yellow
        Stop-Process -Id $proc.OwningProcess -Force -ErrorAction SilentlyContinue
    }
}

Start-Sleep -Seconds 1

# Start backend
Write-Host "Starting Sentinel Backend..." -ForegroundColor Cyan
$backendLog = "$LogDir\backend.log"
$backendProc = Start-Process `
    -FilePath $Python `
    -ArgumentList "main.py" `
    -WorkingDirectory "$PSScriptRoot\sentinel" `
    -RedirectStandardOutput $backendLog `
    -RedirectStandardError "$LogDir\backend_err.log" `
    -PassThru `
    -WindowStyle Hidden

Write-Host "Backend PID: $($backendProc.Id)" -ForegroundColor Green

# Save PID
$backendProc.Id | Out-File -FilePath $PidFile -Force -NoNewline

# Health check with exponential backoff
Write-Host "Waiting for backend on port $BackPort..." -ForegroundColor Cyan
$ready = $false
$attempt = 0
$delay = 500

while ($attempt -lt 20) {
    Start-Sleep -Milliseconds $delay
    $attempt++
    
    try {
        $resp = Invoke-WebRequest -Uri "http://localhost:$BackPort/api/status" -TimeoutSec 2 -UseBasicParsing -ErrorAction Stop
        if ($resp.StatusCode -eq 200) {
            $ready = $true
            break
        }
    }
    catch { }
    
    $delaySeconds = [math]::Round($delay / 1000, 2)
    Write-Host "  Attempt $attempt - retry in ${delaySeconds}s..." -ForegroundColor Yellow
    $delay = [math]::Min($delay * 1.3, 3000)
}

if ($ready) {
    Write-Host "Backend is healthy!" -ForegroundColor Green
}
else {
    Write-Host "WARNING: Backend may not be ready. Check logs:" -ForegroundColor Yellow
    Write-Host "  $backendLog" -ForegroundColor Yellow
    Write-Host "  $LogDir\backend_err.log" -ForegroundColor Yellow
}

# Summary
Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "  Sentinel Started Successfully!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host "  Dashboard: http://localhost:$BackPort" -ForegroundColor Green
Write-Host "  Backend PID: $($backendProc.Id)" -ForegroundColor Green
Write-Host "  Logs: $LogDir" -ForegroundColor Green
Write-Host "  Status: $(if ($ready) { 'READY' } else { 'RUNNING' })" -ForegroundColor $(if ($ready) { 'Green' } else { 'Yellow' })
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
