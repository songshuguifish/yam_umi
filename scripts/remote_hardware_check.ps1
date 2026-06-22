param(
  [string]$Repo = "$env:USERPROFILE\Desktop\yam_umi"
)

$ErrorActionPreference = "Stop"
Set-Location $Repo

Write-Host "[REMOTE] repo=$Repo"
& ".\.venv\Scripts\python.exe" --version
& ".\.venv\Scripts\python.exe" -m pip show pyrealsense2 opencv-python openvr minimalmodbus pyserial numpy
& ".\.venv\Scripts\python.exe" -m sim_teleop.data_collection.collect_smoke --help | Select-Object -First 25
