#!/usr/bin/env bash
# Build Seal OS kernel and create bootable ISO
set -e

echo "=== Building Seal OS Kernel ==="
cargo build --release

echo "=== Creating ISO structure ==="
mkdir -p iso/boot/grub
cp target/x86_64-unknown-none/release/seal-os iso/boot/kernel.bin
cp boot/grub/grub.cfg iso/boot/grub/grub.cfg

echo "=== Creating bootable ISO ==="
grub-mkrescue -o seal-os.iso iso/ 2>/dev/null

echo "=== Done ==="
echo "ISO: seal-os.iso ($(du -h seal-os.iso | cut -f1))"
echo ""
echo "Run with: qemu-system-x86_64 -cdrom seal-os.iso -serial stdio -m 512M"
