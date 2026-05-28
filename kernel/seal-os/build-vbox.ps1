# Build Seal OS and create an Oracle VM VirtualBox-friendly VDI image.
# Requires Rust nightly/stable and Oracle VM VirtualBox.
$ErrorActionPreference = "Stop"

$Image = Join-Path $PSScriptRoot "target\x86_64-unknown-uefi\release\seal-os.img"
$Vdi = Join-Path $PSScriptRoot "target\x86_64-unknown-uefi\release\seal-os.vdi"

function Find-VBoxManage {
    $cmd = Get-Command VBoxManage -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }

    $candidates = @(
        "$env:ProgramFiles\Oracle\VirtualBox\VBoxManage.exe",
        "${env:ProgramFiles(x86)}\Oracle\VirtualBox\VBoxManage.exe"
    )

    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path -LiteralPath $candidate)) {
            return $candidate
        }
    }

    throw "VBoxManage.exe not found. Install Oracle VM VirtualBox or add VBoxManage.exe to PATH."
}

Push-Location $PSScriptRoot
cargo +nightly build --release
Pop-Location

Push-Location (Join-Path $PSScriptRoot "..\seal-mkimage")
cargo +stable run --release
Pop-Location

$VBoxManage = Find-VBoxManage

if (Test-Path -LiteralPath $Vdi) {
    Remove-Item -LiteralPath $Vdi -Force
}

& $VBoxManage convertfromraw --format VDI $Image $Vdi
Write-Host "VirtualBox image ready: $Vdi"
Write-Host "Create VM with EFI enabled, SATA storage, VMSVGA display, and attach this VDI."
