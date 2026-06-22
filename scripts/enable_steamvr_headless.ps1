<#
.SYNOPSIS
  Enable SteamVR "headless" mode so Vive Trackers track without an HMD connected.

.DESCRIPTION
  SteamVR normally won't fully start without a headset. For tracker-only data
  collection that's in the way. The fix is two keys in steamvr.vrsettings:

    "steamvr" : {
        "requireHmd" : false,            # don't require a headset
        "activateMultipleDrivers" : true # let tracker + (null/other) drivers coexist
    }

  This script: locates Steam via the registry, backs up steamvr.vrsettings once
  to .bak, patches the two keys (idempotent), writes it back as UTF-8 (no BOM).
  If SteamVR is running it aborts — SteamVR rewrites steamvr.vrsettings on exit
  and would clobber the change.

  SteamVR must already be installed (run this after installing it from Steam).
  Self-elevates: run from a normal shell and approve the UAC prompt.

.EXAMPLE
  .\scripts\enable_steamvr_headless.ps1
#>

#Requires -Version 5.1
$ErrorActionPreference = 'Stop'

# --- Self-elevate -----------------------------------------------------------
$currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
$isAdmin = ([Security.Principal.WindowsPrincipal]$currentUser).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator
)
if (-not $isAdmin) {
    Write-Host "Requesting administrator privileges..." -ForegroundColor Yellow
    Start-Process -FilePath "powershell.exe" `
        -Verb RunAs `
        -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$PSCommandPath`""
    exit
}

# --- Locate SteamVR config --------------------------------------------------
$steamReg = Get-ItemProperty -Path 'HKCU:\Software\Valve\Steam' -ErrorAction SilentlyContinue
$steamPath = $steamReg.SteamPath
if (-not $steamPath) { $steamPath = 'C:\Program Files (x86)\Steam' }
$cfgPath = Join-Path $steamPath 'config\steamvr.vrsettings'

if (-not (Test-Path $cfgPath)) {
    Write-Host "[FAIL] steamvr.vrsettings not found at:" -ForegroundColor Red
    Write-Host "       $cfgPath"
    Write-Host "       Install SteamVR first (Steam -> Library -> Tools -> SteamVR), then re-run." -ForegroundColor Yellow
    Read-Host "`nPress Enter to close" | Out-Null
    exit 1
}
Write-Host "[OK]   Steam path:  $steamPath" -ForegroundColor Green
Write-Host "[OK]   Config file: $cfgPath" -ForegroundColor Green

# --- Abort if SteamVR is running --------------------------------------------
$vrProcs = Get-Process -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -in 'vrserver', 'vrmonitor', 'vrcompositor' }
if ($vrProcs) {
    Write-Host "[WARN] SteamVR is running ($($vrProcs.Name -join ', '))." -ForegroundColor Yellow
    Write-Host "       It overwrites steamvr.vrsettings on exit and would undo this change." -ForegroundColor Yellow
    Write-Host "       Fully quit SteamVR, then re-run this script." -ForegroundColor Yellow
    Read-Host "`nPress Enter to close" | Out-Null
    exit 1
}

# --- Backup once -------------------------------------------------------------
$bak = "$cfgPath.bak"
if (Test-Path $bak) {
    Write-Host "[SKIP] Backup already exists: $bak" -ForegroundColor DarkGray
} else {
    Copy-Item $cfgPath $bak
    Write-Host "[OK]   Backup created: $bak" -ForegroundColor Green
}

# --- Patch JSON --------------------------------------------------------------
$json = Get-Content $cfgPath -Raw | ConvertFrom-Json
if (-not ($json.PSObject.Properties.Name -contains 'steamvr')) {
    $json | Add-Member -NotePropertyName 'steamvr' -NotePropertyValue ([PSCustomObject]@{})
}

$patched = $false
if ($json.steamvr.requireHmd -ne $false) {
    $json.steamvr.requireHmd = $false; $patched = $true
}
if ($json.steamvr.activateMultipleDrivers -ne $true) {
    $json.steamvr.activateMultipleDrivers = $true; $patched = $true
}

if (-not $patched) {
    Write-Host "[SKIP] Already headless (requireHmd=false, activateMultipleDrivers=true)." -ForegroundColor DarkGray
} else {
    $out = $json | ConvertTo-Json -Depth 100
    # Write UTF-8 without BOM (SteamVR's parser is happiest with plain UTF-8).
    [IO.File]::WriteAllText($cfgPath, $out, (New-Object System.Text.UTF8Encoding $false))
    Write-Host "[OK]   Patched: steamvr.requireHmd=false, steamvr.activateMultipleDrivers=true." -ForegroundColor Green
}

# --- Summary -----------------------------------------------------------------
Write-Host ""
Write-Host "Done. Start SteamVR (Steam -> Library -> Tools -> SteamVR -> Play)." -ForegroundColor Cyan
Write-Host "With a tracker dongle plugged in it should track green with no headset." -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Enter to close this window..." -ForegroundColor DarkGray
Read-Host | Out-Null
