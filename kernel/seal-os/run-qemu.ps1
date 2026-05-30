param(
    [switch]$HeadlessProof,
    [int]$ProofSeconds = 120
)

# Run Seal OS in QEMU with UEFI firmware on Windows.
# Prefer native qemu-system-x86_64 when present; fall back to WSL Ubuntu QEMU.
$ErrorActionPreference = "Stop"

$Image = Join-Path $PSScriptRoot "target\x86_64-unknown-uefi\release\seal-os.img"
$EfiImage = Join-Path $PSScriptRoot "target\x86_64-unknown-uefi\release\seal-os.efi"
$ProofDir = Join-Path $PSScriptRoot "target\x86_64-unknown-uefi\release\qemu-proof"
$SealMkimageManifest = Join-Path $PSScriptRoot "..\seal-mkimage\Cargo.toml"

function Build-SealImage {
    Push-Location $PSScriptRoot
    try {
        cargo +nightly build --release
        if ($LASTEXITCODE -ne 0) {
            throw "kernel build failed"
        }
    } finally {
        Pop-Location
    }

    Push-Location (Join-Path $PSScriptRoot "..\seal-mkimage")
    try {
        cargo +stable run --release
        if ($LASTEXITCODE -ne 0) {
            throw "seal-mkimage build failed"
        }
    } finally {
        Pop-Location
    }
}

if ($HeadlessProof) {
    Write-Host "Headless proof requested. Rebuilding kernel and disk image..."
    Build-SealImage
} elseif (-not (Test-Path -LiteralPath $Image)) {
    Write-Host "No disk image found. Building..."
    Build-SealImage
}

function Find-Qemu {
    $cmd = Get-Command qemu-system-x86_64 -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }

    $candidates = @(
        "$env:ProgramFiles\qemu\qemu-system-x86_64.exe",
        "${env:ProgramFiles(x86)}\qemu\qemu-system-x86_64.exe",
        "$env:ChocolateyInstall\lib\qemu\tools\qemu\qemu-system-x86_64.exe"
    )

    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path -LiteralPath $candidate)) {
            return $candidate
        }
    }

    return $null
}

function Test-WslQemu {
    $wsl = Get-Command wsl.exe -ErrorAction SilentlyContinue
    if (-not $wsl) {
        return $false
    }

    & wsl.exe -- bash -lc 'command -v qemu-system-x86_64 >/dev/null && test -f /usr/share/OVMF/OVMF_CODE_4M.fd && test -f /usr/share/OVMF/OVMF_VARS_4M.fd'
    return ($LASTEXITCODE -eq 0)
}

function ConvertTo-WslPath {
    param([string]$Path)

    $resolved = (Resolve-Path -LiteralPath $Path).Path
    if ($resolved -match '^([A-Za-z]):\\(.*)$') {
        $drive = $Matches[1].ToLowerInvariant()
        $rest = $Matches[2] -replace '\\', '/'
        return "/mnt/$drive/$rest"
    }

    throw "Cannot convert path to WSL form: $Path"
}

function Join-CommandArguments {
    param([string[]]$Arguments)

    $quoted = foreach ($arg in $Arguments) {
        if ($null -eq $arg) {
            '""'
        } elseif ($arg -match '[\s"]') {
            '"' + ($arg -replace '"', '\"') + '"'
        } else {
            $arg
        }
    }

    return ($quoted -join " ")
}

function Send-QemuMonitorCommand {
    param(
        [int]$Port,
        [string]$Command
    )

    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $client.Connect("127.0.0.1", $Port)
        $stream = $client.GetStream()
        $bytes = [Text.Encoding]::ASCII.GetBytes($Command)
        $stream.Write($bytes, 0, $bytes.Length)
    } finally {
        $client.Close()
    }
}

function New-ProofRunDirectory {
    param([string]$CanonicalPath)

    $parent = Split-Path -Parent $CanonicalPath
    $runsRoot = Join-Path $parent "qemu-proof-runs"
    New-Item -ItemType Directory -Force -Path $runsRoot | Out-Null

    do {
        $stamp = Get-Date -Format "yyyyMMdd-HHmmss-fff"
        $suffix = [Guid]::NewGuid().ToString("N").Substring(0, 8)
        $runPath = Join-Path $runsRoot "$stamp-$suffix"
    } while (Test-Path -LiteralPath $runPath)

    New-Item -ItemType Directory -Force -Path $runPath | Out-Null
    return $runPath
}

function Test-NonEmptyFile {
    param([string]$Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return $false
    }

    return ((Get-Item -LiteralPath $Path).Length -gt 0)
}

function Assert-NonEmptyFile {
    param(
        [string]$Path,
        [string]$Description
    )

    if (-not (Test-NonEmptyFile -Path $Path)) {
        throw "$Description missing or empty: $Path"
    }
}

function Get-FileCrc32 {
    param([string]$Path)

    if (-not $script:Crc32Table) {
        $script:Crc32Table = New-Object 'uint32[]' 256
        for ($i = 0; $i -lt 256; $i++) {
            [uint32]$crc = $i
            for ($bit = 0; $bit -lt 8; $bit++) {
                if (($crc -band 1) -ne 0) {
                    $crc = (($crc -shr 1) -bxor [uint32]3988292384)
                } else {
                    $crc = ($crc -shr 1)
                }
            }
            $script:Crc32Table[$i] = $crc
        }
    }

    [uint32]$running = 4294967295
    $buffer = New-Object byte[] 65536
    $stream = [System.IO.File]::OpenRead($Path)
    try {
        while (($read = $stream.Read($buffer, 0, $buffer.Length)) -gt 0) {
            for ($i = 0; $i -lt $read; $i++) {
                $idx = (($running -bxor [uint32]$buffer[$i]) -band 0xFF)
                $running = (($running -shr 8) -bxor $script:Crc32Table[$idx])
            }
        }
    } finally {
        $stream.Dispose()
    }

    [uint32]$final = ($running -bxor [uint32]4294967295)
    return ("{0:x8}" -f $final)
}

function Get-FileSha256 {
    param([string]$Path)

    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-GitValue {
    param(
        [string]$RepoRoot,
        [string[]]$Arguments,
        [string]$Fallback
    )

    try {
        $output = & git -C $RepoRoot @Arguments 2>$null
        if ($LASTEXITCODE -eq 0 -and $output) {
            return (($output | Select-Object -First 1) -as [string]).Trim()
        }
    } catch {
    }

    return $Fallback
}

function Add-ManifestArtifact {
    param(
        [System.Collections.Generic.List[string]]$Lines,
        [string]$Prefix,
        [string]$Path
    )

    $item = Get-Item -LiteralPath $Path
    $Lines.Add("${Prefix}_path=$Path")
    $Lines.Add("${Prefix}_bytes=$($item.Length)")
    $Lines.Add("${Prefix}_crc32=$(Get-FileCrc32 -Path $Path)")
    $Lines.Add("${Prefix}_sha256=$(Get-FileSha256 -Path $Path)")
}

function Invoke-SealMkimageGate {
    param([string[]]$Arguments)

    $cargoArgs = @("+stable", "run", "--manifest-path", $SealMkimageManifest, "--release", "--") + $Arguments
    $gateCwd = Split-Path -Parent $SealMkimageManifest
    Push-Location $gateCwd
    try {
        & cargo @cargoArgs
        if ($LASTEXITCODE -ne 0) {
            throw "seal-mkimage gate failed: $($Arguments -join ' ')"
        }
    } finally {
        Pop-Location
    }
}

function Invoke-ProofGates {
    param(
        [string]$ImagePath,
        [string]$ProofPath
    )

    $serialLog = Join-Path $ProofPath "serial.log"
    $ppm = Join-Path $ProofPath "screen.ppm"

    Assert-NonEmptyFile -Path $serialLog -Description "QEMU proof serial.log"
    Assert-NonEmptyFile -Path $ppm -Description "QEMU proof screen.ppm"

    if (Test-Path -LiteralPath $EfiImage -PathType Leaf) {
        Invoke-SealMkimageGate -Arguments @("--verify", $ImagePath, $EfiImage)
    } else {
        Invoke-SealMkimageGate -Arguments @("--verify", $ImagePath)
    }
    Invoke-SealMkimageGate -Arguments @("--check-vm-proof", $serialLog)
    Invoke-SealMkimageGate -Arguments @("--check-theorem-log", $serialLog)
    Invoke-SealMkimageGate -Arguments @("--check-aether-runtime", $serialLog)
    Invoke-SealMkimageGate -Arguments @("--check-desktop-soak", $serialLog)
    Invoke-SealMkimageGate -Arguments @("--check-benchmark-log", $serialLog)
    Invoke-SealMkimageGate -Arguments @("--check-proof-screen", $ppm)
}

function Write-ProofManifest {
    param(
        [string]$RunPath,
        [string]$ImagePath,
        [string]$Backend,
        [int]$Seconds
    )

    $efiSnapshot = Join-Path $RunPath "seal-os.efi"
    if (Test-Path -LiteralPath $EfiImage -PathType Leaf) {
        Copy-Item -LiteralPath $EfiImage -Destination $efiSnapshot -Force
    }

    $repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")).Path
    $manifest = Join-Path $RunPath "proof-manifest.txt"
    Invoke-SealMkimageGate -Arguments @("--write-qemu-proof-manifest", $RunPath, $ImagePath, $Backend, "$Seconds", $repoRoot)
    Invoke-SealMkimageGate -Arguments @("--check-current-proof-manifest", $manifest, $repoRoot)
}

function Publish-ProofArtifacts {
    param(
        [string]$SourcePath,
        [string]$CanonicalPath
    )

    New-Item -ItemType Directory -Force -Path $CanonicalPath | Out-Null

    foreach ($name in @("serial.log", "screen.ppm", "proof-manifest.txt", "seal-os.img", "seal-os.efi")) {
        $source = Join-Path $SourcePath $name
        if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
            continue
        }
        $destination = Join-Path $CanonicalPath $name
        Copy-Item -LiteralPath $source -Destination $destination -Force
    }

    $pngSource = Join-Path $SourcePath "screen.png"
    $pngDestination = Join-Path $CanonicalPath "screen.png"
    if (Test-Path -LiteralPath $pngSource -PathType Leaf) {
        Copy-Item -LiteralPath $pngSource -Destination $pngDestination -Force
    } else {
        Remove-Item -LiteralPath $pngDestination -Force -ErrorAction SilentlyContinue
    }
}

function Complete-ProofRun {
    param(
        [string]$RunPath,
        [string]$CanonicalPath,
        [string]$ImagePath,
        [string]$Backend,
        [int]$Seconds
    )

    try {
        Invoke-ProofGates -ImagePath $ImagePath -ProofPath $RunPath
        Write-ProofManifest -RunPath $RunPath -ImagePath $ImagePath -Backend $Backend -Seconds $Seconds
        Publish-ProofArtifacts -SourcePath $RunPath -CanonicalPath $CanonicalPath
    } catch {
        Write-Host "QEMU proof failed run kept: $RunPath"
        Write-Error $_.Exception.Message -ErrorAction Continue
        exit 1
    }

    Write-Host "QEMU proof artifacts published: $CanonicalPath"
}

function Invoke-NativeQemuProof {
    param(
        [string]$QemuPath,
        [string]$OvmfPath,
        [string]$ImagePath,
        [string]$ProofPath,
        [int]$Seconds
    )

    New-Item -ItemType Directory -Force -Path $ProofPath | Out-Null
    $serialLog = Join-Path $ProofPath "serial.log"
    $ppm = Join-Path $ProofPath "screen.ppm"
    $png = Join-Path $ProofPath "screen.png"
    Remove-Item -LiteralPath $serialLog, $ppm, $png -Force -ErrorAction SilentlyContinue

    $monitorPort = Get-Random -Minimum 49152 -Maximum 60999
    $arguments = @(
        "-machine", "q35",
        "-m", "4096",
        "-cpu", "qemu64",
        "-smp", "2"
    )
    if ($OvmfPath) {
        $arguments += @("-drive", "if=pflash,format=raw,readonly=on,file=$OvmfPath")
    }
    $arguments += @(
        "-device", "ahci,id=seal_sata",
        "-drive", "if=none,id=seal_disk,file=$ImagePath,format=raw,media=disk",
        "-device", "ide-hd,drive=seal_disk,bus=seal_sata.0,unit=0",
        "-serial", "file:$serialLog",
        "-monitor", "tcp:127.0.0.1:$monitorPort,server,nowait",
        "-no-reboot",
        "-display", "none",
        "-device", "VGA"
    )

    $start = New-Object System.Diagnostics.ProcessStartInfo
    $start.FileName = $QemuPath
    $start.Arguments = Join-CommandArguments $arguments
    $start.UseShellExecute = $false
    $start.CreateNoWindow = $true

    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $start
    [void]$process.Start()

    $deadline = (Get-Date).AddSeconds($Seconds)
    $verdict = "TIMEOUT"
    while ((Get-Date) -lt $deadline -and -not $process.HasExited) {
        $serial = ""
        if (Test-Path -LiteralPath $serialLog) {
            $serial = Get-Content -LiteralPath $serialLog -Raw -ErrorAction SilentlyContinue
        }

        if ($serial -match '\[BOOT\] Desktop proof frame blit done' -and
            $serial -match '\[BOOT\] Seal OS desktop ready\.' -and
            $serial -match '\[EVENT\] Entering real event loop') {
            $verdict = "PASS"
            break
        }
        if ($serial -match '!!! SEAL OS KERNEL PANIC !!!|\[FAULT\]|\[WATCHDOG\]|gurumeditation') {
            $verdict = "FAIL"
            break
        }

        Start-Sleep -Seconds 1
    }

    if ($verdict -eq "PASS") {
        Start-Sleep -Seconds 2
        try {
            Send-QemuMonitorCommand -Port $monitorPort -Command "screendump $ppm`nquit`n"
        } catch {
            Write-Warning "QEMU monitor screenshot capture failed: $($_.Exception.Message)"
            if (-not $process.HasExited) {
                $process.Kill()
            }
        }
    } else {
        if (-not $process.HasExited) {
            $process.Kill()
        }
    }

    $process.WaitForExit(10000) | Out-Null
    if (-not $process.HasExited) {
        $process.Kill()
    }

    Add-Type -AssemblyName System.Drawing -ErrorAction SilentlyContinue
    if ((Test-Path -LiteralPath $ppm) -and ("System.Drawing.Bitmap" -as [type])) {
        try {
            $bitmap = [System.Drawing.Bitmap]::FromFile($ppm)
            $bitmap.Save($png, [System.Drawing.Imaging.ImageFormat]::Png)
            $bitmap.Dispose()
        } catch {
            Write-Warning "PPM to PNG conversion skipped: $($_.Exception.Message)"
        }
    }

    if ($verdict -eq "PASS" -and -not (Test-NonEmptyFile -Path $ppm)) {
        $verdict = "FAIL"
        Write-Host "QEMU proof verdict: FAIL (screen.ppm missing or empty)"
    }

    Write-Host "Serial log: $serialLog"
    if (Test-Path -LiteralPath $png) {
        Write-Host "Screenshot: $png"
    }
    if ($verdict -ne "PASS") {
        Write-Host "QEMU proof verdict: $verdict"
        if (Test-Path -LiteralPath $serialLog) {
            Get-Content -LiteralPath $serialLog -Tail 120
        }
        return $false
    }

    return $true
}

function Invoke-WslQemu {
    param(
        [string]$ImagePath,
        [string]$ProofPath,
        [bool]$Proof,
        [int]$Seconds
    )

    $wslImage = ConvertTo-WslPath $ImagePath
    if ($Proof) {
        New-Item -ItemType Directory -Force -Path $ProofPath | Out-Null
        $wslProof = ConvertTo-WslPath $ProofPath
    } else {
        $wslProof = ""
    }
    $proofFlag = if ($Proof) { "1" } else { "0" }

    $bash = @'
set -euo pipefail
IMG="@@IMG@@"
PROOF_DIR="@@PROOF_DIR@@"
PROOF="@@PROOF@@"
SECONDS_LIMIT="@@SECONDS@@"
CODE="/usr/share/OVMF/OVMF_CODE_4M.fd"
VARS="/tmp/seal_OVMF_VARS_4M.fd"
LOG="$PROOF_DIR/serial.log"
PPM="$PROOF_DIR/screen.ppm"
PNG="$PROOF_DIR/screen.png"
MON="/tmp/seal-qemu-monitor-$$.sock"
cp /usr/share/OVMF/OVMF_VARS_4M.fd "$VARS"

if [ "$PROOF" = "1" ]; then
  mkdir -p "$PROOF_DIR"
  rm -f "$LOG" "$PPM" "$PNG"
  rm -f "$MON"
  qemu-system-x86_64 \
    -machine q35 \
    -m 4096 \
    -cpu qemu64 \
    -smp 2 \
    -drive if=pflash,format=raw,readonly=on,file="$CODE" \
    -drive if=pflash,format=raw,file="$VARS" \
    -device ahci,id=seal_sata \
    -drive if=none,id=seal_disk,file="$IMG",format=raw,media=disk \
    -device ide-hd,drive=seal_disk,bus=seal_sata.0,unit=0 \
    -serial file:"$LOG" \
    -monitor unix:"$MON",server,nowait \
    -no-reboot \
    -display none \
    -device VGA &
  QEMU_PID=$!

  deadline=$((SECONDS_LIMIT + $(date +%s)))
  verdict="TIMEOUT"
  while kill -0 "$QEMU_PID" 2>/dev/null; do
    if [ -f "$LOG" ] && grep -F "[BOOT] Desktop proof frame blit done" "$LOG" >/dev/null && grep -F "[BOOT] Seal OS desktop ready." "$LOG" >/dev/null && grep -F "[EVENT] Entering real event loop" "$LOG" >/dev/null; then
      verdict="PASS"
      break
    fi
    if [ -f "$LOG" ] && grep -E "!!! SEAL OS KERNEL PANIC !!!|\[FAULT\]|\[WATCHDOG\]|gurumeditation" "$LOG" >/dev/null; then
      verdict="FAIL"
      break
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
      break
    fi
    sleep 1
  done

  if [ "$verdict" = "PASS" ]; then
    sleep 2
    for _ in 1 2 3 4 5; do
      [ -S "$MON" ] && break
      sleep 1
    done
    if [ -S "$MON" ] && command -v nc >/dev/null; then
      if ! printf 'screendump %s\nquit\n' "$PPM" | nc -U "$MON" >/dev/null 2>&1; then
        kill "$QEMU_PID" 2>/dev/null || true
      fi
    else
      kill "$QEMU_PID" 2>/dev/null || true
    fi
  else
    kill "$QEMU_PID" 2>/dev/null || true
  fi
  wait "$QEMU_PID" 2>/dev/null || true

  if [ -s "$PPM" ] && command -v ffmpeg >/dev/null; then
    ffmpeg -y -hide_banner -loglevel error -i "$PPM" "$PNG" || true
  fi

  if [ "$verdict" = "PASS" ] && [ ! -s "$PPM" ]; then
    verdict="FAIL"
    echo "QEMU proof verdict: FAIL (screen.ppm missing or empty)"
  fi

  echo "Serial log: $LOG"
  [ -s "$PNG" ] && echo "Screenshot: $PNG"
  if [ "$verdict" != "PASS" ]; then
    echo "QEMU proof verdict: $verdict"
    tail -n 120 "$LOG" 2>/dev/null || true
    exit 1
  fi
  grep -F "[BOOT] Seal OS desktop ready." "$LOG" >/dev/null
else
  qemu-system-x86_64 \
    -machine q35 \
    -m 4096 \
    -cpu qemu64 \
    -smp 2 \
    -drive if=pflash,format=raw,readonly=on,file="$CODE" \
    -drive if=pflash,format=raw,file="$VARS" \
    -device ahci,id=seal_sata \
    -drive if=none,id=seal_disk,file="$IMG",format=raw,media=disk \
    -device ide-hd,drive=seal_disk,bus=seal_sata.0,unit=0 \
    -serial stdio \
    -monitor none \
    -no-reboot \
    -display gtk \
    -device VGA
fi
'@

    $bash = $bash.Replace("@@IMG@@", $wslImage)
    $bash = $bash.Replace("@@PROOF_DIR@@", $wslProof)
    $bash = $bash.Replace("@@PROOF@@", $proofFlag)
    $bash = $bash.Replace("@@SECONDS@@", [string]$Seconds)
    $bash = $bash -replace "`r", ""

    $encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($bash))
    & wsl.exe -- bash -lc "printf '%s' '$encoded' | base64 -d | bash"
}

function Invoke-NativeHeadlessProof {
    param(
        [string]$QemuPath,
        [string]$OvmfPath,
        [string]$ImagePath,
        [string]$CanonicalPath,
        [int]$Seconds
    )

    $runPath = New-ProofRunDirectory -CanonicalPath $CanonicalPath
    Write-Host "QEMU proof run directory: $runPath"

    $passed = Invoke-NativeQemuProof -QemuPath $QemuPath -OvmfPath $OvmfPath -ImagePath $ImagePath -ProofPath $runPath -Seconds $Seconds
    if (-not $passed) {
        Write-Host "QEMU proof failed run kept: $runPath"
        exit 1
    }

    Complete-ProofRun -RunPath $runPath -CanonicalPath $CanonicalPath -ImagePath $ImagePath -Backend "native" -Seconds $Seconds
}

function Invoke-WslHeadlessProof {
    param(
        [string]$ImagePath,
        [string]$CanonicalPath,
        [int]$Seconds
    )

    $runPath = New-ProofRunDirectory -CanonicalPath $CanonicalPath
    Write-Host "QEMU proof run directory: $runPath"

    Invoke-WslQemu -ImagePath $ImagePath -ProofPath $runPath -Proof $true -Seconds $Seconds
    if ($LASTEXITCODE -ne 0) {
        Write-Host "QEMU proof failed run kept: $runPath"
        exit $LASTEXITCODE
    }

    Complete-ProofRun -RunPath $runPath -CanonicalPath $CanonicalPath -ImagePath $ImagePath -Backend "wsl" -Seconds $Seconds
}

$OvmfCandidates = @(
    "$env:ProgramFiles\qemu\share\edk2-x86_64-code.fd",
    "$env:ProgramFiles\qemu\share\edk2-x86_64-code.fd.bak",
    "$env:ProgramFiles\qemu\share\OVMF_CODE.fd",
    "$env:ProgramFiles\qemu\OVMF_CODE.fd",
    "$env:ChocolateyInstall\lib\qemu\tools\qemu\share\edk2-x86_64-code.fd"
)

$Qemu = Find-Qemu
$Ovmf = $OvmfCandidates | Where-Object { $_ -and (Test-Path -LiteralPath $_) } | Select-Object -First 1

if (-not $Qemu) {
    if (Test-WslQemu) {
        Write-Host "Native Windows QEMU not found; using WSL Ubuntu QEMU."
        if ($HeadlessProof) {
            Invoke-WslHeadlessProof -ImagePath $Image -CanonicalPath $ProofDir -Seconds $ProofSeconds
            exit 0
        }
        Invoke-WslQemu -ImagePath $Image -ProofPath $ProofDir -Proof $false -Seconds $ProofSeconds
        exit $LASTEXITCODE
    }

    throw "qemu-system-x86_64 not found on Windows PATH and WSL QEMU is unavailable."
}

if (-not $Ovmf) {
    if (Test-WslQemu) {
        Write-Host "Windows OVMF not found; using WSL Ubuntu QEMU + OVMF."
        if ($HeadlessProof) {
            Invoke-WslHeadlessProof -ImagePath $Image -CanonicalPath $ProofDir -Seconds $ProofSeconds
            exit 0
        }
        Invoke-WslQemu -ImagePath $Image -ProofPath $ProofDir -Proof $false -Seconds $ProofSeconds
        exit $LASTEXITCODE
    }

    Write-Warning "OVMF/EDK2 firmware not found in common QEMU paths. Trying native QEMU without pflash."
    if ($HeadlessProof) {
        Invoke-NativeHeadlessProof -QemuPath $Qemu -OvmfPath "" -ImagePath $Image -CanonicalPath $ProofDir -Seconds $ProofSeconds
        exit 0
    }
    & $Qemu `
        -device "ahci,id=seal_sata" `
        -drive "if=none,id=seal_disk,file=$Image,format=raw,media=disk" `
        -device "ide-hd,drive=seal_disk,bus=seal_sata.0,unit=0" `
        -serial stdio `
        -m 4G `
        -vga std `
        -no-reboot `
        -no-shutdown
    exit $LASTEXITCODE
}

if ($HeadlessProof) {
    Invoke-NativeHeadlessProof -QemuPath $Qemu -OvmfPath $Ovmf -ImagePath $Image -CanonicalPath $ProofDir -Seconds $ProofSeconds
    exit 0
}

& $Qemu `
    -machine q35 `
    -drive "if=pflash,format=raw,readonly=on,file=$Ovmf" `
    -device "ahci,id=seal_sata" `
    -drive "if=none,id=seal_disk,file=$Image,format=raw,media=disk" `
    -device "ide-hd,drive=seal_disk,bus=seal_sata.0,unit=0" `
    -serial stdio `
    -m 4G `
    -vga std `
    -no-reboot `
    -no-shutdown
