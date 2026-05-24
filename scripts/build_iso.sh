#!/usr/bin/env bash
# Epsilon-Hollow — Bootable ISO generator for Seal OS v0.4.5
# Creates a UEFI-bootable ISO that works in VirtualBox, QEMU, and real hardware.
#
# Uses El Torito UEFI boot with embedded FAT12 EFI image.
# Requires: xorriso (recommended), grub-mkrescue, or mkisofs

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EFI_BINARY="$PROJECT_ROOT/kernel/seal-os/target/x86_64-unknown-uefi/release/seal-os.efi"
OUTPUT_ISO="$PROJECT_ROOT/seal-os.iso"
ISO_STAGING="$PROJECT_ROOT/.iso_staging"

# ── Locate EFI binary ────────────────────────────────────────────────────────
if [ ! -f "$EFI_BINARY" ]; then
    echo "[build_iso] EFI binary not found at $EFI_BINARY"
    echo "[build_iso] Building kernel first..."
    cd "$PROJECT_ROOT/kernel/seal-os"
    cargo +nightly build --release
fi

if [ ! -f "$EFI_BINARY" ]; then
    echo "[build_iso] ERROR: Still can't find EFI binary after build. Aborting."
    exit 1
fi

echo "[build_iso] Input EFI: $EFI_BINARY ($(du -h "$EFI_BINARY" | cut -f1))"

# ── Build the mkimage tool to generate EFI boot image ────────────────────────
echo "[build_iso] Building mkimage tool..."
cd "$PROJECT_ROOT/kernel/seal-mkimage"
cargo +stable build --release
cargo +stable run --release

EFI_IMG="$PROJECT_ROOT/kernel/seal-os/target/x86_64-unknown-uefi/release/seal-os-efi.img"
if [ ! -f "$EFI_IMG" ]; then
    echo "[build_iso] ERROR: EFI boot image not generated."
    exit 1
fi

# ── Prepare ISO staging ──────────────────────────────────────────────────────
rm -rf "$ISO_STAGING"
mkdir -p "$ISO_STAGING"

# Copy the FAT12 EFI boot image into staging
cp "$EFI_IMG" "$ISO_STAGING/efi.img"

# Also copy the EFI binary for direct access
cp "$EFI_BINARY" "$ISO_STAGING/seal-os.efi"

# ── Method 1: xorriso with proper El Torito UEFI boot ────────────────────────
if command -v xorriso &>/dev/null; then
    echo "[build_iso] Using xorriso with El Torito UEFI boot..."
    xorriso -as mkisofs \
        -iso-level 3 \
        -V "SEALOS" \
        -J -R \
        -eltorito-platform efi \
        -eltorito-boot efi.img \
        -no-emul-boot \
        -o "$OUTPUT_ISO" \
        "$ISO_STAGING"

    if [ -f "$OUTPUT_ISO" ]; then
        echo "[build_iso] OK: $OUTPUT_ISO ($(du -h "$OUTPUT_ISO" | cut -f1))"
        rm -rf "$ISO_STAGING"
        exit 0
    fi
    echo "[build_iso] xorriso failed, trying next method..."
fi

# ── Method 2: grub-mkrescue (fallback) ───────────────────────────────────────
if command -v grub-mkrescue &>/dev/null; then
    echo "[build_iso] Using grub-mkrescue..."
    mkdir -p "$ISO_STAGING/boot/grub"
    cat > "$ISO_STAGING/boot/grub/grub.cfg" <<'GRUB'
set timeout=3
set default=0
menuentry "Seal OS" {
    chainloader /seal-os.efi
}
GRUB
    set +e
    grub-mkrescue -o "$OUTPUT_ISO" "$ISO_STAGING" \
        --modules="part_gpt fat iso9660 chainloader efi_uga efi_gop" 2>/dev/null
    set -e

    if [ -f "$OUTPUT_ISO" ]; then
        echo "[build_iso] OK: $OUTPUT_ISO ($(du -h "$OUTPUT_ISO" | cut -f1))"
        rm -rf "$ISO_STAGING"
        exit 0
    fi
    echo "[build_iso] grub-mkrescue failed, trying next method..."
fi

# ── Method 3: mkisofs / genisoimage ──────────────────────────────────────────
MKISOFS=""
if command -v mkisofs &>/dev/null; then
    MKISOFS="mkisofs"
elif command -v genisoimage &>/dev/null; then
    MKISOFS="genisoimage"
fi

if [ -n "$MKISOFS" ]; then
    echo "[build_iso] Using $MKISOFS..."
    "$MKISOFS" \
        -iso-level 3 \
        -V "SEALOS" \
        -J -R \
        -eltorito-platform efi \
        -eltorito-boot efi.img \
        -no-emul-boot \
        -o "$OUTPUT_ISO" \
        "$ISO_STAGING"

    if [ -f "$OUTPUT_ISO" ]; then
        echo "[build_iso] OK: $OUTPUT_ISO ($(du -h "$OUTPUT_ISO" | cut -f1))"
        rm -rf "$ISO_STAGING"
        exit 0
    fi
    echo "[build_iso] $MKISOFS failed."
fi

# ── Nothing worked ───────────────────────────────────────────────────────────
rm -rf "$ISO_STAGING"
echo "[build_iso] ERROR: No working ISO tool found."
echo "[build_iso] Please install one of the following:"
echo "  - xorriso        <-- RECOMMENDED for UEFI"
echo "  - grub-mkrescue  (grub-common / grub2-tools)"
echo "  - mkisofs / genisoimage"
exit 1
