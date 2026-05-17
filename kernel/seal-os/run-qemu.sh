#!/usr/bin/env bash
# Run Seal OS in QEMU
set -e

ISO="seal-os.iso"

if [ ! -f "$ISO" ]; then
    echo "No ISO found. Building..."
    ./build-iso.sh
fi

qemu-system-x86_64 \
    -cdrom "$ISO" \
    -serial stdio \
    -m 512M \
    -vga std \
    -no-reboot \
    -no-shutdown
