@echo off
chcp 65001 >nul
title LAAMBA GOVERNOR — Launching...
color 0B

echo.
echo  LAAMBA GOVERNOR v1.0
.
cd /d "%~dp0"

set "EXE=src-tauri\target\release\laamba-governor.exe"
set "LOG=logs\launch.log"

if not exist "logs" mkdir logs

if not exist "%EXE%" (
    echo [ERROR] Release binary not found at %EXE%
    echo [INFO] Run run.bat first to build.
    pause
    exit /b 1
)

echo [OK] Launching LAAMBA Governor... > "%LOG%" 2>&1
echo [OK] Working dir: %CD% >> "%LOG%" 2>&1
echo [OK] Binary: %EXE% >> "%LOG%" 2>&1
echo. >> "%LOG%" 2>&1

REM Run exe directly (NOT with start) so errors stay visible and get logged
"%EXE%" >> "%LOG%" 2>&1

set "EXITCODE=%ERRORLEVEL%"
echo. >> "%LOG%" 2>&1
echo [EXIT] Code: %EXITCODE% >> "%LOG%" 2>&1

if %EXITCODE% neq 0 (
    echo.
    echo [CRASH] App exited with code %EXITCODE%
    echo [LOG]   See logs\launch.log for details
    echo.
    type "%LOG%"
    echo.
    pause
)
