@echo off
setlocal

set BACKEND_PORT=8080
if exist "sentinel\.env" (
    for /f "tokens=1,2 delims==" %%a in ('findstr /R /I "^DASHBOARD_PORT=" "sentinel\.env"') do (
        set BACKEND_PORT=%%b
    )
)

REM === Закрыть предыдущие процессы по PID-файлу ===
set PID_FILE=sentinel\sentinel.pid
if exist "%PID_FILE%" (
    set /p OLD_PID=<"%PID_FILE%"
    if defined OLD_PID (
        echo Останавливаем предыдущий процесс PID=%OLD_PID%...
        taskkill /PID %OLD_PID% /F >nul 2>&1
        del /F "%PID_FILE%" >nul 2>&1
    )
)

REM === Убить всё, что слушает порт backend ===
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":%BACKEND_PORT% "') do (
    echo Закрываем процесс на порту %BACKEND_PORT%: PID=%%a
    taskkill /PID %%a /F >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":8080 "') do (
    echo Закрываем процесс на fallback-порту 8080: PID=%%a
    taskkill /PID %%a /F >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":8888 "') do (
    echo Закрываем процесс на fallback-порту 8888: PID=%%a
    taskkill /PID %%a /F >nul 2>&1
)

REM === Пауза, чтобы ОС освободила порт ===
timeout /t 2 /nobreak >nul

REM === Запуск Backend (sentinel/main.py) ===
echo Запускаем Sentinel Backend...
cd /d "%~dp0sentinel"
start "Sentinel Backend" /min cmd /c "C:/Users/marti/AppData/Local/Programs/Python/Python312/python.exe main.py > ..\logs\backend.log 2>&1"

REM === Ждём пока backend поднимется (max 10 сек) ===
echo Ждём запуска backend...
set /a tries=0
:wait_backend
timeout /t 1 /nobreak >nul
set /a tries+=1
curl -s -o nul -w "%%{http_code}" http://localhost:%BACKEND_PORT%/api/status 2>nul | findstr "200" >nul
if %errorlevel%==0 goto backend_ready
if %tries% geq 10 (
    echo ПРЕДУПРЕЖДЕНИЕ: Backend не ответил за 10 сек, продолжаем...
    goto backend_ready
)
goto wait_backend

:backend_ready
echo Backend запущен.
echo.
echo ============================================
echo  Sentinel запущен!
echo  Dashboard: http://localhost:%BACKEND_PORT%
echo ============================================

endlocal
