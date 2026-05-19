#!/usr/bin/env bash
# Run Seal OS in Docker with X11 forwarding
# Usage: ./run.sh
set -e

cd "$(dirname "$0")"

# Allow X11 connections from Docker
xhost +local:docker 2>/dev/null || true

echo "Building Seal OS kernel + image in Docker..."
docker compose up --build

# Revoke X11 access
xhost -local:docker 2>/dev/null || true
