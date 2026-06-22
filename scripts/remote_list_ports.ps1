Write-Host "[REMOTE-PORTS] SerialPort names:"
[System.IO.Ports.SerialPort]::GetPortNames()

Write-Host "[REMOTE-PORTS] PnP Ports:"
Get-PnpDevice -Class Ports | Select-Object FriendlyName, Status
