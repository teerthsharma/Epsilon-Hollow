#!/usr/bin/env bash
# Build Seal OS kernel and create bootable UEFI disk image
set -e

echo "=== Building Seal OS Kernel (UEFI) ==="
cargo +nightly build --release

echo "=== Creating UEFI disk image ==="
cd ../seal-mkimage
cargo +stable run --release

echo "=== Done ==="
echo "Image: target/x86_64-unknown-uefi/release/seal-os.img"
echo ""
echo "Run with: ./run-qemu.sh"
