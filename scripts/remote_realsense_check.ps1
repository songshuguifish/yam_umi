param(
  [string]$Repo = "$env:USERPROFILE\Desktop\yam_umi",
  [string]$OutputDir = "data/realsense_remote_check",
  [int]$Frames = 5,
  [int]$MaxCameras = 3,
  [int]$Width = 640,
  [int]$Height = 480,
  [int]$Fps = 30
)

$ErrorActionPreference = "Stop"
Set-Location $Repo

Write-Host "[REMOTE-RS] repo=$Repo"
Write-Host "[REMOTE-RS] output=$OutputDir frames=$Frames"

& ".\.venv\Scripts\python.exe" -m sim_teleop.data_collection.realsense_rgb_check `
  --output-dir $OutputDir `
  --frames $Frames `
  --max-cameras $MaxCameras `
  --width $Width `
  --height $Height `
  --fps $Fps

exit $LASTEXITCODE
