#Requires -Version 5.1
# Sentinel v2 Launcher - Advanced startup with health checks and configuration validation

$ErrorActionPreference = "Stop"

# CONFIG
$PidFile  = "$PSScriptRoot\sentinel\sentinel.pid"
$EnvFile  = "$PSScriptRoot\sentinel\.env"
$Python   = "C:/Users/marti/AppData/Local/Programs/Python/Python312/python.exe"
$BackPort = 8888
$LogDir   = "$PSScriptRoot\sentinel\logs"
$MainPy   = "$PSScriptRoot\sentinel\main.py"

# Verify python exists
if (-not (Test-Path $Python)) {
    Write-Host "ERROR: Python not found at $Python" -ForegroundColor Red
    $fallback = (Get-Command python -ErrorAction SilentlyContinue).Source
    if ($fallback) {
        Write-Host "Using fallback: $fallback" -ForegroundColor Yellow
        $Python = $fallback
    } else {
        exit 1
    }
}

# Verify main.py exists
if (-not (Test-Path $MainPy)) {
    Write-Host "ERROR: main.py not found at $MainPy" -ForegroundColor Red
    exit 1
}

# Try to read port from .env
if (Test-Path $EnvFile) {
    $line = Get-Content $EnvFile -ErrorAction SilentlyContinue | Where-Object { $_ -match '^\s*DASHBOARD_PORT\s*=\s*(\d+)' } | Select-Object -First 1
    if ($line -match 'DASHBOARD_PORT\s*=\s*(\d+)') {
        $BackPort = [int]$Matches[1]
    }
}

# Create log directory
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir -Force | Out-Null }

# --- CLEANUP PHASE ------------------------------------------------------------

function Stop-SentinelPid {
    param([int]$ProcessId, [string]$Reason)
    try {
        $p = Get-Process -Id $ProcessId -ErrorAction Stop
        Write-Host "  Killing PID=$ProcessId ($($p.ProcessName)) - $Reason" -ForegroundColor Yellow
        Stop-Process -Id $ProcessId -Force -ErrorAction Stop
        return $true
    } catch {
        return $false
    }
}

# 1) Kill process from PID file
if (Test-Path $PidFile) {
    $oldPidRaw = Get-Content $PidFile -Raw -ErrorAction SilentlyContinue
    if ($oldPidRaw) {
        $oldPidClean = ($oldPidRaw -replace '[^\d]', '').Trim()
        if ($oldPidClean -match '^\d+$') {
            Stop-SentinelPid -ProcessId ([int]$oldPidClean) -Reason "from PID file" | Out-Null
        }
    }
    Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
}

# 2) Kill zombie python processes running sentinel/main.py (command-line match)
try {
    $sentinelProcs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -and ($_.CommandLine -match 'sentinel[\\/]+main\.py' -or $_.CommandLine -match 'main\.py') -and $_.CommandLine -match 'sentinel' }
    foreach ($sp in $sentinelProcs) {
        Stop-SentinelPid -ProcessId ([int]$sp.ProcessId) -Reason "zombie sentinel python" | Out-Null
    }
} catch {
    Write-Host "  (CIM scan skipped: $($_.Exception.Message))" -ForegroundColor DarkGray
}

# 3) Free the dashboard port
$ports = @($BackPort, 8080, 8888) | Select-Object -Unique
foreach ($port in $ports) {
    $conns = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    foreach ($c in $conns) {
        Stop-SentinelPid -ProcessId ([int]$c.OwningProcess) -Reason "holds port $port" | Out-Null
    }
}

Start-Sleep -Seconds 2

# Verify port is free
$still = Get-NetTCPConnection -LocalPort $BackPort -State Listen -ErrorAction SilentlyContinue
if ($still) {
    Write-Host "WARNING: port $BackPort still occupied after cleanup (PID=$($still.OwningProcess))" -ForegroundColor Red
}

# --- STARTUP PHASE ------------------------------------------------------------

Write-Host "Starting Sentinel Backend (port $BackPort)..." -ForegroundColor Cyan
$backendLog = "$LogDir\backend.log"
$backendErr = "$LogDir\backend_err.log"

try {
    $backendProc = Start-Process `
        -FilePath $Python `
        -ArgumentList "main.py" `
        -WorkingDirectory "$PSScriptRoot\sentinel" `
        -RedirectStandardOutput $backendLog `
        -RedirectStandardError $backendErr `
        -PassThru `
        -WindowStyle Hidden `
        -ErrorAction Stop
} catch {
    Write-Host "ERROR: Failed to start python: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

Write-Host "Backend PID: $($backendProc.Id)" -ForegroundColor Green
# NOTE: do NOT write $PidFile here. main.py writes it itself at pre-flight step 10,
# and if we pre-populate the file python sees its own PID as a "live other instance" and aborts.

# --- HEALTH CHECK -------------------------------------------------------------

Write-Host "Waiting for backend on port $BackPort..." -ForegroundColor Cyan
$ready = $false
$attempt = 0
$delay = 500

while ($attempt -lt 30) {
    Start-Sleep -Milliseconds $delay
    $attempt++

    # Early-fail: if the python process already died, stop waiting
    if ($backendProc.HasExited) {
        Write-Host "ERROR: backend process exited with code $($backendProc.ExitCode) before becoming ready" -ForegroundColor Red
        break
    }

    try {
        $resp = Invoke-WebRequest -Uri "http://localhost:$BackPort/api/status" -TimeoutSec 2 -UseBasicParsing -ErrorAction Stop
        if ($resp.StatusCode -eq 200) {
            $ready = $true
            break
        }
    } catch { }

    $delaySeconds = [math]::Round($delay / 1000, 2)
    Write-Host "  Attempt $attempt - retry in ${delaySeconds}s..." -ForegroundColor Yellow
    $delay = [math]::Min($delay * 1.3, 3000)
}

if ($ready) {
    Write-Host "Backend is healthy!" -ForegroundColor Green
} else {
    Write-Host "WARNING: Backend may not be ready. Last 15 lines of logs:" -ForegroundColor Yellow
    if (Test-Path $backendErr) {
        Write-Host "--- backend_err.log ---" -ForegroundColor DarkYellow
        Get-Content $backendErr -Tail 15 -ErrorAction SilentlyContinue | ForEach-Object { Write-Host "  $_" -ForegroundColor DarkGray }
    }
    if (Test-Path $backendLog) {
        Write-Host "--- backend.log ---" -ForegroundColor DarkYellow
        Get-Content $backendLog -Tail 15 -ErrorAction SilentlyContinue | ForEach-Object { Write-Host "  $_" -ForegroundColor DarkGray }
    }
}

# --- SUMMARY ------------------------------------------------------------------

Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "  Sentinel Started!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host "  Dashboard:   http://localhost:$BackPort" -ForegroundColor Green
Write-Host "  Backend PID: $($backendProc.Id)" -ForegroundColor Green
Write-Host "  Logs:        $LogDir" -ForegroundColor Green
Write-Host "  Status:      $(if ($ready) { 'READY' } else { 'NOT READY - check logs' })" -ForegroundColor $(if ($ready) { 'Green' } else { 'Yellow' })
Write-Host "============================================" -ForegroundColor Green
Write-Host ""

if (-not $ready) { exit 2 }
