#Requires -Version 5.1
<#
.SYNOPSIS
    Запуск Sentinel: убивает предыдущие процессы, затем стартует backend.
#>

$ErrorActionPreference = "SilentlyContinue"
$PidFile  = "$PSScriptRoot\sentinel\sentinel.pid"
$EnvFile  = "$PSScriptRoot\sentinel\.env"
$Python   = "C:/Users/marti/AppData/Local/Programs/Python/Python312/python.exe"
$BackPort = 8080
$LogDir   = "$PSScriptRoot\sentinel\logs"

# Пытаемся взять порт из sentinel/.env (DASHBOARD_PORT=...)
if (Test-Path $EnvFile) {
    $line = Get-Content $EnvFile | Where-Object { $_ -match '^\s*DASHBOARD_PORT\s*=\s*\d+\s*$' } | Select-Object -First 1
    if ($line -match '(\d+)') {
        $BackPort = [int]$Matches[1]
    }
}

# Создать папку логов если нет
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }

# --- 1. Убить процесс по PID-файлу ---
if (Test-Path $PidFile) {
    $oldPid = Get-Content $PidFile -Raw
    if ($oldPid -match '^\d+$') {
        Write-Host "Останавливаем предыдущий процесс PID=$($oldPid.Trim())..."
        Stop-Process -Id ([int]$oldPid.Trim()) -Force -ErrorAction SilentlyContinue
    }
    Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
}

# --- 2. Убить всё, что занимает порт backend ---
$portsToClean = @($BackPort, 8080, 8888) | Select-Object -Unique
foreach ($port in $portsToClean) {
    $occupied = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    if ($occupied) {
        foreach ($conn in $occupied) {
            Write-Host "Закрываем процесс на порту $port (PID=$($conn.OwningProcess))..."
            Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue
        }
    }
}

# Небольшая пауза, чтобы ОС освободила порт
Start-Sleep -Seconds 2

# --- 3. Запуск Backend ---
Write-Host "Запускаем Sentinel Backend..."
$backendLog = "$LogDir\backend.log"
$backendProc = Start-Process `
    -FilePath $Python `
    -ArgumentList "main.py" `
    -WorkingDirectory "$PSScriptRoot\sentinel" `
    -RedirectStandardOutput $backendLog `
    -RedirectStandardError "$LogDir\backend_err.log" `
    -PassThru `
    -WindowStyle Hidden

Write-Host "Backend PID: $($backendProc.Id)"

# --- 4. Ждём чтобы backend поднялся (max 15 сек) ---
Write-Host "Ждём запуска backend (порт $BackPort)..."
$tries = 0
$ready = $false
while ($tries -lt 15) {
    Start-Sleep -Seconds 1
    $tries++
    try {
        $resp = Invoke-WebRequest -Uri "http://localhost:$BackPort/api/status" -TimeoutSec 2 -UseBasicParsing -ErrorAction Stop
        if ($resp.StatusCode -eq 200) { $ready = $true; break }
    } catch {}
}

if ($ready) {
    Write-Host "Backend запущен успешно."
} else {
    Write-Host "ПРЕДУПРЕЖДЕНИЕ: Backend не ответил за 15 сек. Проверьте $backendLog и $LogDir\backend_err.log"
}

Write-Host ""
Write-Host "============================================"
Write-Host " Sentinel запущен!"
Write-Host " Dashboard: http://localhost:$BackPort"
Write-Host "============================================"
