# Build Rust DSP to WebAssembly
# ===============================
# Compiles sanctuary-dsp to WASM for browser execution

Write-Host "🦀 Building Rust DSP to WebAssembly..." -ForegroundColor Cyan

$rustDspPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$frontendPath = Join-Path (Split-Path -Parent $rustDspPath) "frontend"
$wasmOutputPath = Join-Path $frontendPath "public\wasm"

# Check if wasm-pack is installed
if (-not (Get-Command wasm-pack -ErrorAction SilentlyContinue)) {
    Write-Host "📦 Installing wasm-pack..." -ForegroundColor Yellow
    cargo install wasm-pack
}

# Build WASM
Push-Location $rustDspPath
try {
    Write-Host "⚙️ Compiling to WebAssembly (release + SIMD)..." -ForegroundColor Yellow
    wasm-pack build --target web --release --out-dir pkg
    
    if ($LASTEXITCODE -ne 0) {
        throw "wasm-pack build failed"
    }
    
    # Create output directory if needed
    if (-not (Test-Path $wasmOutputPath)) {
        New-Item -ItemType Directory -Path $wasmOutputPath -Force | Out-Null
    }
    
    # Copy WASM files to frontend
    Write-Host "📁 Copying WASM files to frontend..." -ForegroundColor Yellow
    Copy-Item "pkg\sanctuary_dsp_bg.wasm" -Destination $wasmOutputPath -Force
    Copy-Item "pkg\sanctuary_dsp.js" -Destination $wasmOutputPath -Force
    Copy-Item "pkg\sanctuary_dsp.d.ts" -Destination $wasmOutputPath -Force -ErrorAction SilentlyContinue
    
    Write-Host "✅ WASM build complete!" -ForegroundColor Green
    Write-Host "   Output: $wasmOutputPath" -ForegroundColor Gray
    
}
finally {
    Pop-Location
}
