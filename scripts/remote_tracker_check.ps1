param(
  [string]$Repo = "$env:USERPROFILE\Desktop\yam_umi",
  [double]$Duration = 5,
  [double]$Frequency = 10
)

$ErrorActionPreference = "Stop"
Set-Location $Repo

Write-Host "[REMOTE-TRACKER] repo=$Repo"
Write-Host "[REMOTE-TRACKER] duration=$Duration frequency=$Frequency"

& ".\.venv\Scripts\python.exe" -m sim_teleop.data_collection.tracker_process `
  --duration $Duration `
  --frequency $Frequency `
  --serial-prefix=

exit $LASTEXITCODE
