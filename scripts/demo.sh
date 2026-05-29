#!/usr/bin/env bash
# Epsilon-Hollow v0.5 Demo Script
# Shows: boot → store → teleport → race → theorems

set -e

IMAGE="ghcr.io/teerthsharma/epsilon-hollow:0.5"

echo "=== Epsilon-Hollow v0.5 Demo ==="
echo ""
echo "Building image..."
docker build -t epsilon-hollow:0.5 . 2>/dev/null

echo ""
echo "=== 1. Boot with seal splash ==="
echo '/status' | docker run --rm -i epsilon-hollow:0.5
echo ""

echo "=== 2. Store data and teleport ==="
printf '/store Hello this is semantic data stored as geometry on S²\n/store Binary payload simulating 2GB of real data\n/ls\n/mv /Hello /archive/Hello\n/ls /archive\n' | docker run --rm -i epsilon-hollow:0.5
echo ""

echo "=== 3. Race: Teleport vs Traditional Copy ==="
echo '/race 2000000000' | docker run --rm -i epsilon-hollow:0.5
echo ""

echo "=== 4. All Theorems Active ==="
echo '/theorems' | docker run --rm -i epsilon-hollow:0.5
echo ""

echo "=== Demo Complete ==="
echo "OS state = topology on S^2. File moves = O(1) topological surgery."
