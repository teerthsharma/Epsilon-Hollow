#!/usr/bin/env bash
# Epsilon-Hollow — Bootable ISO generator for Seal OS
# Creates a UEFI-bootable ISO from the seal-os.efi binary.
#
# Requires: grub-mkrescue (recommended), xorriso, or mkisofs
# On Debian/Ubuntu: sudo apt-get install grub-common grub-efi-amd64-bin xorriso

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

# ── Prepare staging directory ────────────────────────────────────────────────
rm -rf "$ISO_STAGING"
mkdir -p "$ISO_STAGING/EFI/BOOT"
cp "$EFI_BINARY" "$ISO_STAGING/EFI/BOOT/BOOTX64.EFI"

# ── Method 1: grub-mkrescue (most compatible) ────────────────────────────────
if command -v grub-mkrescue &>/dev/null; then
    echo "[build_iso] Using grub-mkrescue (recommended)..."
    mkdir -p "$ISO_STAGING/boot/grub"
    cat > "$ISO_STAGING/boot/grub/grub.cfg" <<'GRUB'
set timeout=3
set default=0

insmod efi_gop
insmod efi_uga
insmod part_gpt

menuentry "Seal OS" {
    chainloader /EFI/BOOT/BOOTX64.EFI
}
GRUB
    set +e
    grub-mkrescue -o "$OUTPUT_ISO" "$ISO_STAGING" \
        --modules="part_gpt fat iso9660 chainloader efi_uga efi_gop" 2>/dev/null
    if [ ! -f "$OUTPUT_ISO" ]; then
        grub-mkrescue -o "$OUTPUT_ISO" "$ISO_STAGING" 2>/dev/null
    fi
    set -e

    if [ -f "$OUTPUT_ISO" ]; then
        echo "[build_iso] OK: $OUTPUT_ISO ($(du -h "$OUTPUT_ISO" | cut -f1))"
        rm -rf "$ISO_STAGING"
        exit 0
    fi
    echo "[build_iso] grub-mkrescue failed, trying next method..."
fi

# ── Method 2: xorriso with UEFI direct boot ──────────────────────────────────
if command -v xorriso &>/dev/null; then
    echo "[build_iso] Using xorriso (UEFI direct boot)..."
    # Some UEFI firmware can boot an ISO 9660 filesystem directly if
    # /EFI/BOOT/BOOTX64.EFI exists. We add an El Torito UEFI catalog entry.
    xorriso -as mkisofs \
        -o "$OUTPUT_ISO" \
        -iso-level 3 \
        -V "SEALOS" \
        -J -R \
        -eltorito-platform efi \
        -e EFI/BOOT/BOOTX64.EFI \
        -no-emul-boot \
        "$ISO_STAGING" 2>/dev/null || true

    if [ -f "$OUTPUT_ISO" ]; then
        echo "[build_iso] OK: $OUTPUT_ISO ($(du -h "$OUTPUT_ISO" | cut -f1))"
        echo "[build_iso] NOTE: UEFI direct-boot ISO. May not work on all firmware."
        rm -rf "$ISO_STAGING"
        exit 0
    fi
    echo "[build_iso] xorriso failed, trying next method..."
fi

# ── Method 3: mkisofs / genisoimage ──────────────────────────────────────────
MKISOFS=""
if command -v mkisofs &>/dev/null; then
    MKISOFS="mkisofs"
elif command -v genisoimage &>/dev/null; then
    MKISOFS="genisoimage"
fi

if [ -n "$MKISOFS" ]; then
    echo "[build_iso] Using $MKISOFS (UEFI direct boot)..."
    "$MKISOFS" \
        -o "$OUTPUT_ISO" \
        -iso-level 3 \
        -V "SEALOS" \
        -J -R \
        -eltorito-platform efi \
        -e EFI/BOOT/BOOTX64.EFI \
        -no-emul-boot \
        "$ISO_STAGING" 2>/dev/null || true

    if [ -f "$OUTPUT_ISO" ]; then
        echo "[build_iso] OK: $OUTPUT_ISO ($(du -h "$OUTPUT_ISO" | cut -f1))"
        echo "[build_iso] NOTE: UEFI direct-boot ISO. May not work on all firmware."
        rm -rf "$ISO_STAGING"
        exit 0
    fi
    echo "[build_iso] $MKISOFS failed."
fi

# ── Nothing worked ───────────────────────────────────────────────────────────
rm -rf "$ISO_STAGING"
echo "[build_iso] ERROR: No working ISO tool found."
echo "[build_iso] Please install one of the following:"
echo "  - grub-mkrescue  (grub-common / grub2-tools)  <-- RECOMMENDED"
echo "  - xorriso"
echo "  - mkisofs / genisoimage"
exit 1
