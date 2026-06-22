param(
  [string]$Repo = "$env:USERPROFILE\Desktop\yam_umi",
  [string]$Port,
  [double]$Duration = 3,
  [double]$Frequency = 10
)

$ErrorActionPreference = "Stop"
Set-Location $Repo

if (-not $Port) {
  throw "Provide -Port, e.g. COM4"
}

Write-Host "[REMOTE-ENCODER] repo=$Repo"
Write-Host "[REMOTE-ENCODER] port=$Port duration=$Duration frequency=$Frequency"

& ".\.venv\Scripts\python.exe" -m sim_teleop.data_collection.encoder_process `
  --port $Port `
  --duration $Duration `
  --frequency $Frequency

exit $LASTEXITCODE
