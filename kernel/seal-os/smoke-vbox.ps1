param(
    [int]$Seconds = 45,
    [switch]$SkipBuild,
    [string[]]$SuccessPattern = @(
        '\[ALLOC\] O\(1\) proof:',
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
        '\[BOOT\] Desktop proof frame blit done',
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
$Vdi = Join-Path $ReleaseDir "seal-os.vdi"
$SmokeDir = Join-Path $ReleaseDir "vbox-smoke"
$SmokeVdi = Join-Path $SmokeDir "seal-os-smoke.vdi"
$SerialLog = Join-Path $SmokeDir "serial.log"
$Screenshot = Join-Path $SmokeDir "screenshot.png"
$VBoxCommandLogDir = Join-Path $ReleaseDir "vbox-command-logs"
$RunStamp = Get-Date -Format "yyyyMMdd-HHmmss"
$VBoxHome = Join-Path $SmokeDir "vbox-home"
$VmName = "SealOS-Codex-Smoke-$RunStamp"

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
        Push-Location (Join-Path $PSScriptRoot "..\seal-mkimage")
        try {
            cargo +stable run --release -- --check-vbox-proof $SerialLog
            if ($LASTEXITCODE -ne 0) {
                $result = @{
                    Verdict = "FAIL"
                    Reason = "seal-mkimage --check-vbox-proof failed"
                }
            }
        } finally {
            Pop-Location
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

if ($result.Verdict -eq "PASS") {
    $exitCode = 0
}

exit $exitCode
