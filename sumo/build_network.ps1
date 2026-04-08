# build_network.ps1 - Jaam Ctrl
# Run from project root: .\sumo\build_network.ps1

$sumoDir = $PSScriptRoot

Write-Host "Jaam Ctrl - Building network with netconvert" -ForegroundColor Cyan

# Delete old hand-written network.net.xml if present
$netFile = "$sumoDir\network.net.xml"
if (Test-Path $netFile) {
    Remove-Item $netFile
    Write-Host "Deleted old network.net.xml" -ForegroundColor Yellow
}

# Create output folder
New-Item -ItemType Directory -Force -Path "$sumoDir\output" | Out-Null

& netconvert `
    "--node-files=$sumoDir\nodes.nod.xml" `
    "--edge-files=$sumoDir\edges.edg.xml" `
    "--connection-files=$sumoDir\connections.con.xml" `
    "--tllogic-files=$sumoDir\tllogic.tll.xml" `
    "--output-file=$netFile" `
    "--no-turnarounds=true" `
    "--junctions.corner-detail=5" `
    "--tls.default-type=static" `
    "--sidewalks.guess=false" `
    "--crossings.guess=false" `
    "--no-warnings"

if ($LASTEXITCODE -eq 0 -and (Test-Path $netFile)) {
    Write-Host "SUCCESS: network.net.xml created" -ForegroundColor Green
    Write-Host "Now run: sumo -c sumo\config.sumocfg" -ForegroundColor Green
} else {
    Write-Host "ERROR: netconvert failed (exit $LASTEXITCODE)" -ForegroundColor Red
    exit 1
}
