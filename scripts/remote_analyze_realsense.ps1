param(
  [string]$Repo = "$env:USERPROFILE\Desktop\yam_umi",
  [string]$InputPath,
  [string]$Kind = "color"
)

$ErrorActionPreference = "Stop"
Set-Location $Repo

if (-not $InputPath) {
  throw "Provide -InputPath, e.g. data/realsense_record_process_check/session_YYYYmmdd_HHMMSS"
}

& ".\.venv\Scripts\python.exe" -m sim_teleop.data_collection.analyze_realsense_sync `
  $InputPath `
  --kind $Kind
