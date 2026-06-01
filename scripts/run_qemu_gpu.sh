#!/usr/bin/env bash
# Seal OS GPU-enabled QEMU launcher
# Copyright (c) 2024 Teerth Sharma
# SPDX-License-Identifier: MIT

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
IMG="${REPO_ROOT}/kernel/seal-os/target/x86_64-unknown-uefi/release/seal-os.img"
OVMF_CODE="/usr/share/OVMF/OVMF_CODE_4M.fd"
OVMF_VARS="/usr/share/OVMF/OVMF_VARS_4M.fd"

PASSTHROUGH_GPU=""
HEADLESS=0
PROOF=0

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Launch Seal OS in QEMU with optional GPU passthrough.

Options:
  --passthrough <BDF>   Pass through a PCI GPU (e.g., 0000:0a:00.0)
  --headless            No SDL window; serial only
  --proof               Capture headless proof bundle
  --help                Show this message

Examples:
  # Software rendering (no GPU)
  ./run_qemu_gpu.sh

  # GPU passthrough (requires IOMMU + vfio-pci)
  ./run_qemu_gpu.sh --passthrough 0000:0a:00.0

  # CI headless mode with proof capture
  ./run_qemu_gpu.sh --headless --proof
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --passthrough)
            PASSTHROUGH_GPU="$2"
            shift 2
            ;;
        --headless)
            HEADLESS=1
            shift
            ;;
        --proof)
            PROOF=1
            shift
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Verify image exists
if [[ ! -f "$IMG" ]]; then
    echo "ERROR: Disk image not found: $IMG"
    echo "Build it first: cd kernel/seal-os && cargo +nightly build --release"
    exit 1
fi

# Build QEMU argument list
QEMU_ARGS=(
    -machine q35,accel=kvm:tcg
    -cpu host
    -m 4G
    -smp 2
    -drive if=pflash,format=raw,readonly=on,file="$OVMF_CODE"
    -drive if=pflash,format=raw,file="$OVMF_VARS"
    -device ahci,id=seal_sata
    -drive if=none,id=seal_disk,file="$IMG",format=raw,media=disk
    -device ide-hd,drive=seal_disk,bus=seal_sata.0,unit=0
    -serial stdio
    -no-reboot
    -no-shutdown
)

# Display settings
if [[ "$HEADLESS" -eq 1 ]]; then
    QEMU_ARGS+=(-nographic)
else
    QEMU_ARGS+=(-display sdl,gl=on -vga none)
fi

# GPU passthrough
if [[ -n "$PASSTHROUGH_GPU" ]]; then
    echo "[QEMU-GPU] Enabling PCI passthrough for $PASSTHROUGH_GPU"
    # Verify VFIO binding
    if [[ ! -L "/sys/bus/pci/devices/$PASSTHROUGH_GPU/driver" ]] || \
       [[ "$(readlink /sys/bus/pci/devices/$PASSTHROUGH_GPU/driver)" != *"vfio-pci"* ]]; then
        echo "WARNING: $PASSTHROUGH_GPU is not bound to vfio-pci."
        echo "Run: sudo ./scripts/vfio_bind_gpu.sh $PASSTHROUGH_GPU"
    fi

    QEMU_ARGS+=(-device vfio-pci,host="$PASSTHROUGH_GPU",multifunction=on)
fi

# virtio-GPU for software testing (always added as secondary)
QEMU_ARGS+=(-device virtio-vga,max_hostmem=1G)

# Proof capture
if [[ "$PROOF" -eq 1 ]]; then
    mkdir -p "${REPO_ROOT}/qemu-proof"
    QEMU_ARGS+=(
        -chardev file,id=serial_log,path="${REPO_ROOT}/qemu-proof/serial.log"
        -serial chardev:serial_log
    )
fi

echo "[QEMU-GPU] Launching Seal OS with GPU support..."
echo "[QEMU-GPU] Image: $IMG"
echo "[QEMU-GPU] Args: ${QEMU_ARGS[*]}"

qemu-system-x86_64 "${QEMU_ARGS[@]}"
