# Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
# SPDX-License-Identifier: Epsilon-Hollow

#!/bin/bash
set -e

MODEL_DIR="/app/models"
MODEL_FILE="$MODEL_DIR/Phi-3-mini-4k-instruct.Q4_K_M.gguf"

if [ ! -f "$MODEL_FILE" ]; then
    echo "[BOOT] Model not found at $MODEL_FILE. Initiating download..."
    # Ensure dependencies are installed if we use python script
    # Or use curl if direct link is stable. 
    # Using the python script we established.
    python3 /app/download_model.py
else
    echo "[BOOT] Model found."
fi

echo "[BOOT] Starting APEIRON Kernel..."
exec "$@"
