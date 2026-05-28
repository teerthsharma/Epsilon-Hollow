# Run Seal OS in QEMU with UEFI firmware on Windows.
# Requires qemu-system-x86_64 on PATH and an OVMF/EDK2 firmware file.
$ErrorActionPreference = "Stop"

$Image = Join-Path $PSScriptRoot "target\x86_64-unknown-uefi\release\seal-os.img"

if (-not (Test-Path -LiteralPath $Image)) {
    Write-Host "No disk image found. Building..."
    Push-Location $PSScriptRoot
    cargo +nightly build --release
    Pop-Location

    Push-Location (Join-Path $PSScriptRoot "..\seal-mkimage")
    cargo +stable run --release
    Pop-Location
}

$OvmfCandidates = @(
    "$env:ProgramFiles\qemu\share\edk2-x86_64-code.fd",
    "$env:ProgramFiles\qemu\share\edk2-x86_64-code.fd.bak",
    "$env:ProgramFiles\qemu\share\OVMF_CODE.fd",
    "$env:ProgramFiles\qemu\OVMF_CODE.fd",
    "$env:ChocolateyInstall\lib\qemu\tools\qemu\share\edk2-x86_64-code.fd"
)

$Ovmf = $OvmfCandidates | Where-Object { $_ -and (Test-Path -LiteralPath $_) } | Select-Object -First 1

if (-not $Ovmf) {
    Write-Warning "OVMF/EDK2 firmware not found in common QEMU paths. Trying QEMU without pflash."
    qemu-system-x86_64 -drive "file=$Image,format=raw" -serial stdio -m 4G -vga std -no-reboot -no-shutdown
    exit $LASTEXITCODE
}

qemu-system-x86_64 `
    -machine q35 `
    -drive "if=pflash,format=raw,readonly=on,file=$Ovmf" `
    -drive "file=$Image,format=raw" `
    -serial stdio `
    -m 4G `
    -vga std `
    -no-reboot `
    -no-shutdown
