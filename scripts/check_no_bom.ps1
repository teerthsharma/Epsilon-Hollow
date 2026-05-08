# PowerShell equivalent of scripts/check_no_bom.sh.
# Fails (exit 1) if any tracked text file under kernel/, infrastructure/, or scripts/
# contains a UTF-8 BOM (0xEF 0xBB 0xBF) anywhere in the file.

$ErrorActionPreference = 'Stop'

$Roots = @('kernel', 'infrastructure', 'scripts')
$BinaryExt = @('.png','.jpg','.jpeg','.gif','.ico','.pdf','.zip','.gz','.tar','.bin','.exe','.dll','.so','.dylib','.wasm','.o','.a','.rlib')

# Prefer git ls-files for tracked-file enumeration; fall back to Get-ChildItem.
$files = @()
try {
    $null = git rev-parse --is-inside-work-tree 2>$null
    if ($LASTEXITCODE -eq 0) {
        $files = git ls-files -- $Roots 2>$null
    }
} catch { }
if (-not $files -or $files.Count -eq 0) {
    foreach ($r in $Roots) {
        if (Test-Path $r) {
            $files += (Get-ChildItem -Path $r -File -Recurse | ForEach-Object { $_.FullName })
        }
    }
}

$found = 0
$bom = [byte[]](0xEF, 0xBB, 0xBF)

foreach ($f in $files) {
    if (-not (Test-Path -LiteralPath $f)) { continue }
    $ext = [System.IO.Path]::GetExtension($f).ToLower()
    if ($BinaryExt -contains $ext) { continue }

    try {
        $bytes = [System.IO.File]::ReadAllBytes($f)
    } catch { continue }
    if ($bytes.Length -lt 3) { continue }

    # Search for BOM byte sequence anywhere.
    $offset = -1
    for ($i = 0; $i -le $bytes.Length - 3; $i++) {
        if ($bytes[$i] -eq 0xEF -and $bytes[$i+1] -eq 0xBB -and $bytes[$i+2] -eq 0xBF) {
            $offset = $i
            break
        }
    }
    if ($offset -ge 0) {
        Write-Error "BOM found: $f (byte offset $offset)" -ErrorAction Continue
        $found = 1
    }
}

if ($found -ne 0) {
    Write-Error "ERROR: UTF-8 BOM detected in tracked files. Strip BOMs and recommit." -ErrorAction Continue
    exit 1
}
exit 0
