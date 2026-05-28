param(
    [int]$Seconds = 45,
    [switch]$SkipBuild,
    [string]$SuccessPattern = '\[BOOT\] Seal OS desktop ready\.',
    [string[]]$FailurePattern = @(
        '!!! SEAL OS KERNEL PANIC !!!',
        '\[FAULT\]',
        '\[WATCHDOG\]',
        'gurumeditation'
    ),
    [int]$PollSeconds = 1
)

# Build a VDI, boot Seal OS in a temporary VirtualBox VM, and watch serial output.
$ErrorActionPreference = "Stop"

$ReleaseDir = Join-Path $PSScriptRoot "target\x86_64-unknown-uefi\release"
$Vdi = Join-Path $ReleaseDir "seal-os.vdi"
$SmokeDir = Join-Path $ReleaseDir "vbox-smoke"
$SmokeVdi = Join-Path $SmokeDir "seal-os-smoke.vdi"
$SerialLog = Join-Path $SmokeDir "serial.log"
$Screenshot = Join-Path $SmokeDir "screenshot.png"
$VmName = "SealOS-Codex-Smoke"

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

    $oldErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = & $VBoxManage @Arguments 2>&1 | ForEach-Object { $_.ToString() }
    } finally {
        $ErrorActionPreference = $oldErrorActionPreference
    }
    if ($LASTEXITCODE -ne 0) {
        throw "VBoxManage $($Arguments -join ' ') failed with exit code $LASTEXITCODE`: $($output -join [Environment]::NewLine)"
    }
    return $output
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

    $existing = & $VBoxManage list vms 2>$null | Select-String "`"$Name`""
    return [bool]$existing
}

function Get-VmState {
    param([string]$Name)

    try {
        $stateLine = & $VBoxManage showvminfo $Name --machinereadable 2>$null | Select-String '^VMState=' | Select-Object -First 1
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
        return Get-Content -LiteralPath $Path -Raw -ErrorAction SilentlyContinue
    }

    return ""
}

function Wait-ForSmokeVerdict {
    param(
        [string]$VmName,
        [string]$SerialLog,
        [int]$TimeoutSeconds,
        [string]$SuccessPattern,
        [string[]]$FailurePattern,
        [int]$PollSeconds
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        $serial = Read-SerialLog $SerialLog

        if ($serial -match $SuccessPattern) {
            return @{
                Verdict = "PASS"
                Reason = "success sentinel matched: $SuccessPattern"
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

Stop-SmokeVm $VmName

if (Test-Path -LiteralPath $SmokeDir) {
    Remove-Item -LiteralPath $SmokeDir -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $SmokeDir | Out-Null
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
