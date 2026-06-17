param(
    [int]$Seconds = 45,
    [switch]$SkipBuild,
    [string[]]$SuccessPattern = @(
        '\[ALLOC\] O\(1\) proof:',
        '\[BENCH\] toporam-alloc',
        '\[BENCH\] alloc-frame',
        '\[BENCH\] slab-alloc',
        '\[BENCH\] manifold-teleport',
        '\[BENCH\] manifold-lookup',
        '\[BENCH\] scheduler-select-next',
        '\[BENCH\] tcp-packet-demux',
        '\[BENCH\] tcp-roundtrip',
        '\[BENCH\] tls-encrypt',
        '\[BENCH\] topo-render-3d',
        '\[BENCH\] tensor-render',
        '\[GPU-BENCH\] suite',
        '\[Aether-Lang\] runtime proof:',
        '\[LAAMBA\] app proof:',
        '\[THEOREM\] T1/TSS VERIFIED',
        '\[THEOREM\] T2/SCM VERIFIED',
        '\[THEOREM\] T3/GMC VERIFIED',
        '\[THEOREM\] T4/AGCR VERIFIED',
        '\[THEOREM\] T5/HCS VERIFIED',
        '\[THEOREM\] T6/RGCS VERIFIED',
        '\[THEOREM\] T7/PHKP VERIFIED',
        '\[THEOREM\] T8/TEB VERIFIED',
        '\[THEOREM\] T9/CMA VERIFIED',
        '\[THEOREM\] T10/WPHB VERIFIED',
        '\[AHCI\] Device model: VBOX HARDDISK',
        '\[AHCI\] Registered as block device 0x800',
        '\[disk::ahci\] First disk readable \(sector 0 OK\)',
        '\[VFS\] ManifoldFS mounted from disk',
        '\[GFX\] desktop-proof',
        '\[GFX\] desktop-live-proof',
        '\[BOOT\] Desktop proof frame blit done',
        '\[GFX\] desktop-soak',
        '\[BOOT\] Seal OS desktop ready\.',
        '\[EVENT\] Entering real event loop'
    ),
    [string[]]$FailurePattern = @(
        '!!! SEAL OS KERNEL PANIC !!!',
        '\[FAULT\]',
        '\[WATCHDOG\]',
        '\[VFS\] No persistent disk found',
        'Falling back to ramfs',
        '\[AHCI\] No SATA device found',
        '\[disk::ahci\] No AHCI device present',
        'gurumeditation'
    ),
    [int]$PollSeconds = 1,
    [int]$VBoxCommandTimeoutSeconds = 30,
    [switch]$UseGlobalVBoxHome
)

# Build a VDI, boot Seal OS in a temporary VirtualBox VM, and watch serial output.
$ErrorActionPreference = "Stop"

$ReleaseDir = Join-Path $PSScriptRoot "target\x86_64-unknown-uefi\release"
$Image = Join-Path $ReleaseDir "seal-os.img"
$EfiImage = Join-Path $ReleaseDir "seal-os.efi"
$Vdi = Join-Path $ReleaseDir "seal-os.vdi"
$SmokeDir = Join-Path $ReleaseDir "vbox-smoke"
$SmokeVdi = Join-Path $SmokeDir "seal-os-smoke.vdi"
$ProofVdi = Join-Path $SmokeDir "seal-os-smoke-proof.vdi"
$SerialLog = Join-Path $SmokeDir "serial.log"
$Screenshot = Join-Path $SmokeDir "screenshot.png"
$ProofManifest = Join-Path $SmokeDir "proof-manifest.txt"
$VBoxCommandLogDir = Join-Path $ReleaseDir "vbox-command-logs"
$RunStamp = Get-Date -Format "yyyyMMdd-HHmmss"
$VBoxHome = Join-Path $SmokeDir "vbox-home"
$VmName = "SealOS-Codex-Smoke-$RunStamp"
$SealMkimageManifest = Join-Path $PSScriptRoot "..\seal-mkimage\Cargo.toml"

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

function Invoke-VBoxManage {
    param(
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Arguments
    )

    New-Item -ItemType Directory -Force -Path $VBoxCommandLogDir | Out-Null
    $commandId = [Guid]::NewGuid().ToString("N")
    $stdoutPath = Join-Path $VBoxCommandLogDir "$commandId.out"
    $stderrPath = Join-Path $VBoxCommandLogDir "$commandId.err"
    $argLine = Join-Arguments $Arguments

    $start = New-Object System.Diagnostics.ProcessStartInfo
    $start.FileName = $VBoxManage
    $start.Arguments = $argLine
    $start.UseShellExecute = $false
    $start.RedirectStandardOutput = $true
    $start.RedirectStandardError = $true
    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $start

    [void]$process.Start()
    $stdoutTask = $process.StandardOutput.ReadToEndAsync()
    $stderrTask = $process.StandardError.ReadToEndAsync()

    if (-not $process.WaitForExit($VBoxCommandTimeoutSeconds * 1000)) {
        try {
            $process.Kill()
        } catch {
            Write-Warning "Failed to kill timed-out VBoxManage process: $($_.Exception.Message)"
        }
        throw "VBoxManage $($Arguments -join ' ') timed out after $VBoxCommandTimeoutSeconds seconds"
    }

    $stdout = $stdoutTask.Result
    $stderr = $stderrTask.Result
    Set-Content -LiteralPath $stdoutPath -Value $stdout -NoNewline
    Set-Content -LiteralPath $stderrPath -Value $stderr -NoNewline

    $output = @()
    if ($stdout) {
        $output += ($stdout -split "`r?`n" | Where-Object { $_ -ne "" })
    }
    if ($stderr) {
        $output += ($stderr -split "`r?`n" | Where-Object { $_ -ne "" })
    }

    if ($process.ExitCode -ne 0) {
        throw "VBoxManage $($Arguments -join ' ') failed with exit code $($process.ExitCode): $($output -join [Environment]::NewLine)"
    }
    return $output
}

function Join-Arguments {
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

function Write-VBoxProofManifest {
    param([int]$Seconds)

    $efiSnapshot = Join-Path $SmokeDir "seal-os.efi"
    Copy-Item -LiteralPath $EfiImage -Destination $efiSnapshot -Force

    Assert-NonEmptyFile -Path $ProofVdi -Description "VirtualBox proof seal-os-smoke-proof.vdi"
    Assert-NonEmptyFile -Path $efiSnapshot -Description "VirtualBox proof seal-os.efi"
    Assert-NonEmptyFile -Path $SerialLog -Description "VirtualBox proof serial.log"
    Assert-NonEmptyFile -Path $Screenshot -Description "VirtualBox proof screenshot.png"

    $repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")).Path
    $gitCommit = Get-GitValue -RepoRoot $repoRoot -Arguments @("rev-parse", "HEAD") -Fallback "unknown"
    $dirty = Get-GitValue -RepoRoot $repoRoot -Arguments @("status", "--porcelain") -Fallback ""
    $gitDirty = if ($dirty) { "true" } else { "false" }

    $lines = [System.Collections.Generic.List[string]]::new()
    $lines.Add("seal_proof_manifest_version=1")
    $lines.Add("vm_target=vbox")
    $lines.Add("virtualbox_backend=headless")
    $lines.Add("proof_verdict=PASS")
    $lines.Add("proof_seconds=$Seconds")
    $lines.Add("created_utc=$((Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ"))")
    $lines.Add("git_commit=$gitCommit")
    $lines.Add("git_dirty=$gitDirty")
    $lines.Add("gate_verify=ok")
    $lines.Add("gate_vbox_proof=ok")
    $lines.Add("gate_theorem_log=ok")
    $lines.Add("gate_aether_runtime=ok")
    $lines.Add("gate_desktop_soak=ok")
    $lines.Add("gate_benchmark_log=ok")

    Add-ManifestArtifact -Lines $lines -Prefix "image" -Path $ProofVdi
    Add-ManifestArtifact -Lines $lines -Prefix "efi" -Path $efiSnapshot
    Add-ManifestArtifact -Lines $lines -Prefix "serial_log" -Path $SerialLog
    Add-ManifestArtifact -Lines $lines -Prefix "screenshot_png" -Path $Screenshot

    Set-Content -LiteralPath $ProofManifest -Value $lines -Encoding ascii
    Invoke-SealMkimageGate -Arguments @("--check-proof-manifest", $ProofManifest)
    Invoke-SealMkimageGate -Arguments @("--check-current-proof-manifest", $ProofManifest, $repoRoot)
}

function Invoke-VBoxManageWarn {
    param(
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Arguments
    )

    try {
        Invoke-VBoxManage @Arguments
        return $true
    } catch {
        Write-Warning $_.Exception.Message
        return $false
    }
}

function Test-VmExists {
    param([string]$Name)

    try {
        Invoke-VBoxManage showvminfo $Name --machinereadable | Out-Null
        return $true
    } catch {
        $message = $_.Exception.Message
        if ($message -match "Could not find a registered machine named" -or $message -match "Could not find a registered machine") {
            return $false
        }
        Write-Warning "Unable to query VM '$Name': $message"
        return $false
    }
}

function Get-VmState {
    param([string]$Name)

    try {
        $stateLine = Invoke-VBoxManage showvminfo $Name --machinereadable | Select-String '^VMState=' | Select-Object -First 1
        if ($stateLine) {
            return ($stateLine.ToString() -replace '^VMState="?(.*?)"?$', '$1')
        }
    } catch {
        return $null
    }

    return $null
}

function Stop-SmokeVm {
    param([string]$Name)

    if (-not (Test-VmExists $Name)) {
        return
    }

    $state = Get-VmState $Name
    if ($state -in @("running", "paused", "stuck", "guru_meditation", "guru meditation")) {
        Write-Host "Stopping $Name (state: $state)..."
        Invoke-VBoxManageWarn controlvm $Name poweroff | Out-Null
        Start-Sleep -Seconds 2
    }

    if (Test-VmExists $Name) {
        Write-Host "Unregistering $Name..."
        Invoke-VBoxManageWarn unregistervm $Name --delete | Out-Null
    }
}

function Read-SerialLog {
    param([string]$Path)

    if (Test-Path -LiteralPath $Path) {
        $content = Get-Content -LiteralPath $Path -Raw -ErrorAction SilentlyContinue
        if ($null -eq $content) {
            return ""
        }
        return [string]$content
    }

    return ""
}

function Wait-ForSmokeVerdict {
    param(
        [string]$VmName,
        [string]$SerialLog,
        [int]$TimeoutSeconds,
        [string[]]$SuccessPattern,
        [string[]]$FailurePattern,
        [int]$PollSeconds
    )

    if ($SuccessPattern.Count -eq 0) {
        return @{
            Verdict = "FAIL"
            Reason = "no success sentinels configured"
        }
    }

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        $serial = [string](Read-SerialLog $SerialLog)
        if ($serial.Length -eq 0) {
            Start-Sleep -Seconds $PollSeconds
            continue
        }

        $missingSuccess = @($SuccessPattern | Where-Object { $serial -notmatch $_ })
        if ($missingSuccess.Count -eq 0) {
            return @{
                Verdict = "PASS"
                Reason = "all success sentinels matched"
            }
        }

        foreach ($pattern in $FailurePattern) {
            if ($serial -match $pattern) {
                return @{
                    Verdict = "FAIL"
                    Reason = "failure sentinel matched: $pattern"
                }
            }
        }

        $state = Get-VmState $VmName
        if ($state -and ($state -notin @("running", "starting", "paused"))) {
            return @{
                Verdict = "FAIL"
                Reason = "VM stopped before success sentinel (state: $state)"
            }
        }

        Start-Sleep -Seconds $PollSeconds
    }

    return @{
        Verdict = "TIMEOUT"
        Reason = "no sentinel matched within $TimeoutSeconds seconds"
    }
}

$VBoxManage = Find-VBoxManage
$exitCode = 1
$result = @{
    Verdict = "FAIL"
    Reason = "smoke script did not complete"
}

if ($PollSeconds -lt 1) {
    throw "-PollSeconds must be at least 1."
}

if (-not $SkipBuild -or -not (Test-Path -LiteralPath $Vdi)) {
    Push-Location $PSScriptRoot
    try {
        powershell -ExecutionPolicy Bypass -File .\build-vbox.ps1
    } finally {
        Pop-Location
    }
}

if (Test-Path -LiteralPath $SmokeDir) {
    Remove-Item -LiteralPath $SmokeDir -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $SmokeDir | Out-Null
if (-not $UseGlobalVBoxHome) {
    New-Item -ItemType Directory -Force -Path $VBoxHome | Out-Null
    $env:VBOX_USER_HOME = $VBoxHome
    Write-Host "Using isolated VBOX_USER_HOME: $VBoxHome"
}
Copy-Item -LiteralPath $Vdi -Destination $SmokeVdi -Force
Copy-Item -LiteralPath $Vdi -Destination $ProofVdi -Force

try {
    Invoke-VBoxManage createvm --name $VmName --basefolder $SmokeDir --register | Out-Null
    Invoke-VBoxManage modifyvm $VmName --ostype Other_64 --firmware efi --memory 4096 --cpus 2 --vram 128 --graphicscontroller vmsvga --boot1 disk --audio-driver none --nic1 none | Out-Null
    Invoke-VBoxManage storagectl $VmName --name "SATA" --add sata --controller IntelAhci --portcount 1 --bootable on | Out-Null
    Invoke-VBoxManage storageattach $VmName --storagectl "SATA" --port 0 --device 0 --type hdd --medium $SmokeVdi | Out-Null
    Invoke-VBoxManage modifyvm $VmName --uart1 0x3F8 4 --uartmode1 file $SerialLog | Out-Null

    Write-Host "Starting $VmName; watching serial for up to $Seconds seconds..."
    Invoke-VBoxManage startvm $VmName --type headless | Out-Null
    $result = Wait-ForSmokeVerdict -VmName $VmName -SerialLog $SerialLog -TimeoutSeconds $Seconds -SuccessPattern $SuccessPattern -FailurePattern $FailurePattern -PollSeconds $PollSeconds

    Invoke-VBoxManageWarn controlvm $VmName screenshotpng $Screenshot | Out-Null
} finally {
    Stop-SmokeVm $VmName
}

$serialText = Read-SerialLog $SerialLog
if ($result.Verdict -eq "PASS") {
    $missingAfterRun = @($SuccessPattern | Where-Object { $serialText -notmatch $_ })
    if ($serialText.Length -eq 0) {
        $result = @{
            Verdict = "FAIL"
            Reason = "serial log is empty after reported PASS"
        }
    } elseif ($missingAfterRun.Count -gt 0) {
        $result = @{
            Verdict = "FAIL"
            Reason = "post-run serial proof missing: $($missingAfterRun -join ', ')"
        }
    } else {
        try {
            Invoke-SealMkimageGate -Arguments @("--verify", $Image, $EfiImage)
            Invoke-SealMkimageGate -Arguments @("--check-vbox-proof", $SerialLog)
            Invoke-SealMkimageGate -Arguments @("--check-theorem-log", $SerialLog)
            Invoke-SealMkimageGate -Arguments @("--check-aether-runtime", $SerialLog)
            Invoke-SealMkimageGate -Arguments @("--check-desktop-soak", $SerialLog)
            Invoke-SealMkimageGate -Arguments @("--check-benchmark-log", $SerialLog)
            Write-VBoxProofManifest -Seconds $Seconds
        } catch {
            $result = @{
                Verdict = "FAIL"
                Reason = $_.Exception.Message
            }
        }
    }
}

if (Test-Path -LiteralPath $SerialLog) {
    $serialBytes = (Get-Item -LiteralPath $SerialLog).Length
    Write-Host "Smoke verdict: $($result.Verdict) - $($result.Reason)"
    Write-Host "Serial log: $SerialLog ($serialBytes bytes)"
    Write-Host "Last serial lines:"
    Get-Content -LiteralPath $SerialLog -Tail 120
} else {
    Write-Host "Smoke verdict: $($result.Verdict) - $($result.Reason)"
    Write-Warning "No serial log was produced."
}

if (Test-Path -LiteralPath $Screenshot) {
    Write-Host "Screenshot: $Screenshot"
}
if (Test-Path -LiteralPath $ProofManifest) {
    Write-Host "Proof manifest: $ProofManifest"
}

if ($result.Verdict -eq "PASS") {
    $exitCode = 0
}

exit $exitCode
