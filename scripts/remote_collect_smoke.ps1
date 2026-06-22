param(
  [string]$Repo = "$env:USERPROFILE\Desktop\yam_umi",
  [string]$OutputDir = "data/raw_smoke",
  [double]$Duration = 5,
  [double]$EncoderFrequency = 30,
  [double]$TrackerFrequency = 60,
  [int]$CameraWidth = 640,
  [int]$CameraHeight = 480,
  [int]$CameraFps = 30,
  [int]$MaxCameras = 3,
  [string[]]$EncoderPort = @()
)

$ErrorActionPreference = "Stop"
Set-Location $Repo

Write-Host "[REMOTE-COLLECT] repo=$Repo"
Write-Host "[REMOTE-COLLECT] output=$OutputDir duration=$Duration"
$EncoderPorts = @()
foreach ($Entry in $EncoderPort) {
  if ($Entry) {
    $EncoderPorts += ($Entry -split "," | Where-Object { $_ })
  }
}
if ($EncoderPorts.Count -eq 0) {
  $ResolvedPorts = & ".\.venv\Scripts\python.exe" -m sim_teleop.data_collection.calibrate_encoders resolve --plain
  if ($LASTEXITCODE -eq 0) {
    foreach ($Port in $ResolvedPorts) {
      if ($Port) {
        $EncoderPorts += $Port.Trim()
      }
    }
  }
}
if ($EncoderPorts.Count -gt 0) {
  Write-Host "[REMOTE-COLLECT] encoder_ports=$($EncoderPorts -join ',')"
}

$cmd = @(
  "-m", "sim_teleop.data_collection.collect_smoke",
  "--output-dir", $OutputDir,
  "--duration", $Duration,
  "--encoder-frequency", $EncoderFrequency,
  "--tracker-frequency", $TrackerFrequency,
  "--camera-width", $CameraWidth,
  "--camera-height", $CameraHeight,
  "--camera-fps", $CameraFps,
  "--max-cameras", $MaxCameras
)
foreach ($Port in $EncoderPorts) {
  $cmd += @("--encoder-port", $Port)
}

& ".\.venv\Scripts\python.exe" @cmd

exit $LASTEXITCODE
