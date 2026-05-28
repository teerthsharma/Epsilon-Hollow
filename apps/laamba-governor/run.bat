@echo off
chcp 65001 >nul
title LAAMBA GOVERNOR — Topology Research Workstation
color 0B

echo.
echo  ╔══════════════════════════════════════════════════════════════╗
echo  ║                                                              ║
echo  ║           LAAMBA GOVERNOR  v1.0                              ║
echo  ║           Topology · Physics · AI · Research                 ║
echo  ║                                                              ║
echo  ╚══════════════════════════════════════════════════════════════╝
echo.

set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"

:: Check Node.js
echo [1/5] Checking Node.js...
node --version >nul 2>&1
if errorlevel 1 (
    echo     ✗ Node.js not found. Install from https://nodejs.org
    pause
    exit /b 1
)
for /f "tokens=*" %%a in ('node --version') do echo     ✓ Node.js %%a

:: Check Rust
echo [2/5] Checking Rust...
cargo --version >nul 2>&1
if errorlevel 1 (
    echo     ✗ Rust not found. Install from https://rustup.rs
    pause
    exit /b 1
)
for /f "tokens=*" %%a in ('cargo --version') do echo     ✓ %%a

:: Check Python
echo [3/5] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo     ✗ Python not found. Install from https://python.org
    pause
    exit /b 1
)
for /f "tokens=*" %%a in ('python --version') do echo     ✓ %%a

:: Install npm deps
echo [4/5] Checking npm dependencies...
if not exist "node_modules" (
    echo     → Installing npm packages... this may take a minute
    call npm install
    if errorlevel 1 (
        echo     ✗ npm install failed
        pause
        exit /b 1
    )
) else (
    echo     ✓ node_modules present
)

:: Seed data
echo [5/5] Checking seed datasets...
if not exist "data\index.json" (
    echo     → Generating seed datasets...
    python cli\generate_seed_data.py
) else (
    echo     ✓ Datasets present
)

:: Check Python deps
echo.
echo Checking Python packages...
python -c "import numpy, sklearn" >nul 2>&1
if errorlevel 1 (
    echo     → Installing numpy + scikit-learn...
    pip install numpy scikit-learn
) else (
    echo     ✓ numpy, scikit-learn OK
)

echo.
echo ══════════════════════════════════════════════════════════════
echo   Launching LAAMBA GOVERNOR...
echo   Wait 10-30 seconds for first compile.
echo   Press Ctrl+C here to stop.
echo ══════════════════════════════════════════════════════════════
echo.

npm run tauri dev

if errorlevel 1 (
    echo.
    echo [ERROR] App exited with error.
    pause
)
