# Run Seal OS in Docker with X11 forwarding (Windows)
# Requires: VcXsrv or Xming running, Docker Desktop
#
# 1. Start VcXsrv with "Disable access control" checked
# 2. Run this script

$env:DISPLAY = "host.docker.internal:0"

Write-Host "Building Seal OS kernel + ISO in Docker..."
docker compose -f "$PSScriptRoot\docker-compose.yml" up --build
