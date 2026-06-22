param(
  [string]$Repo = "$env:USERPROFILE\Desktop\yam_umi",
  [string]$OutputDir = "data/realsense_record_process_check",
  [double]$Duration = 5,
  [int]$MaxCameras = 3,
  [int]$Width = 640,
  [int]$Height = 480,
  [int]$Fps = 30
)

$ErrorActionPreference = "Stop"
Set-Location $Repo

$Session = Join-Path $OutputDir ("session_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
$ReadyFile = Join-Path $Session "realsense_ready.json"
New-Item -ItemType Directory -Force -Path $Session | Out-Null

Write-Host "[REMOTE-RS-REC] repo=$Repo"
Write-Host "[REMOTE-RS-REC] output=$Session duration=$Duration"

& ".\.venv\Scripts\python.exe" -m sim_teleop.data_collection.realsense_rgb_record `
  --output-dir $Session `
  --duration $Duration `
  --width $Width `
  --height $Height `
  --fps $Fps `
  --max-cameras $MaxCameras `
  --ready-file $ReadyFile

if (-not (Test-Path $ReadyFile)) {
  throw "ready file was not written: $ReadyFile"
}

Write-Host "[REMOTE-RS-REC] ready_file=$ReadyFile"
Write-Host "[REMOTE-RS-REC] session=$Session"
