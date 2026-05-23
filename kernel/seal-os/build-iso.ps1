# Build Seal OS kernel and create bootable UEFI disk image
$ErrorActionPreference = "Stop"

Write-Host "=== Building Seal OS Kernel (UEFI) ==="
cargo +nightly build --release

Write-Host "=== Creating UEFI disk image ==="
Push-Location ../seal-mkimage
cargo +stable run --release
Pop-Location

Write-Host "=== Done ==="
Write-Host "Image: target/x86_64-unknown-uefi/release/seal-os.img"
Write-Host ""
Write-Host "Run with: .\run-qemu.ps1 (if you create one) or use Docker"
