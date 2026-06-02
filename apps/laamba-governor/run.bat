@echo off
chcp 65001 >nul
title LAAMBA GOVERNOR - Native Aether Topology Workstation
color 0B

echo.
echo ================================================================
echo   LAAMBA GOVERNOR
echo   Native Rust + Aether topology workstation
echo ================================================================
echo.

set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"

echo [1/4] Checking Node.js...
node --version >nul 2>&1
if errorlevel 1 (
    echo     Node.js not found. Install from https://nodejs.org
    pause
    exit /b 1
)
for /f "tokens=*" %%a in ('node --version') do echo     Node.js %%a

echo [2/4] Checking Rust...
cargo --version >nul 2>&1
if errorlevel 1 (
    echo     Rust not found. Install from https://rustup.rs
    pause
    exit /b 1
)
for /f "tokens=*" %%a in ('cargo --version') do echo     %%a

echo [3/4] Checking npm dependencies...
if not exist "node_modules" (
    echo     Installing npm packages...
    call npm install
    if errorlevel 1 (
        echo     npm install failed
        pause
        exit /b 1
    )
) else (
    echo     node_modules present
)

echo [4/4] Checking seed datasets...
if not exist "data\index.json" (
    echo     data\index.json missing. Use the checked-in seed data bundle or restore data files.
    pause
    exit /b 1
) else (
    echo     datasets present
)

echo.
echo Launching native LAAMBA GOVERNOR...
echo Press Ctrl+C here to stop.
echo.

npm run tauri dev

if errorlevel 1 (
    echo.
    echo [ERROR] App exited with error.
    pause
)
