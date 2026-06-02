#!/usr/bin/env bash
# Build bootable Ubuntu ISO with benchmark runner.
# Creates: ubuntu-bench.iso (UEFI-bootable, runs benchmarks on boot)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[build] Building Ubuntu benchmark ISO..."

# ── Verify dependencies ──────────────────────────────────────────────────────
need_cmd() {
    if ! command -v "$1" &>/dev/null; then
        echo "[build] ERROR: required command not found: $1"
        exit 1
    fi
}

need_cmd gcc
need_cmd cpio
need_cmd gzip
need_cmd xorriso

# grub-mkrescue is optional; we have a xorriso fallback.

# ── Build static benchmark binary ────────────────────────────────────────────
echo "[build] Compiling workloads..."
gcc -O2 -static -o workloads workloads.c

# ── Assemble initramfs ───────────────────────────────────────────────────────
echo "[build] Creating initramfs..."
INITRAMFS_DIR="$(mktemp -d)"
trap "rm -rf '$INITRAMFS_DIR'" EXIT

mkdir -p "$INITRAMFS_DIR"/{bin,dev,proc,sys,tmp,lib/modules}

# Copy busybox and create essential symlinks
BUSYBOX_SRC=""
for path in /bin/busybox /usr/bin/busybox /usr/bin/busybox.static; do
    if [ -f "$path" ]; then
        BUSYBOX_SRC="$path"
        break
    fi
done

if [ -z "$BUSYBOX_SRC" ]; then
    echo "[build] Installing busybox-static..."
    apt-get update -qq && apt-get install -y -qq busybox-static
    BUSYBOX_SRC="/bin/busybox"
fi

cp "$BUSYBOX_SRC" "$INITRAMFS_DIR/bin/busybox"
for cmd in sh echo cat mount mkdir sleep reboot poweroff sync chmod ln rm rmdir mknod; do
    ln -sf busybox "$INITRAMFS_DIR/bin/$cmd"
done

cp workloads "$INITRAMFS_DIR/bin/"
cp bench_runner.sh "$INITRAMFS_DIR/init"
chmod +x "$INITRAMFS_DIR/init"

# Essential device nodes (fallback if devtmpfs fails)
mknod -m 622 "$INITRAMFS_DIR/dev/console" c 5 1 2>/dev/null || true
mknod -m 666 "$INITRAMFS_DIR/dev/null"    c 1 3 2>/dev/null || true
mknod -m 666 "$INITRAMFS_DIR/dev/zero"    c 1 5 2>/dev/null || true
mknod -m 666 "$INITRAMFS_DIR/dev/random"  c 1 8 2>/dev/null || true
mknod -m 666 "$INITRAMFS_DIR/dev/urandom" c 1 9 2>/dev/null || true

INITRD_IMG="$SCRIPT_DIR/initrd.img"
(
    cd "$INITRAMFS_DIR"
    find . | cpio -H newc -o 2>/dev/null | gzip -9 > "$INITRD_IMG"
)
echo "[build] initrd.img: $(du -h "$INITRD_IMG" | cut -f1)"

# ── Obtain kernel ────────────────────────────────────────────────────────────
VMLINUZ="$SCRIPT_DIR/vmlinuz"
if [ ! -f "$VMLINUZ" ]; then
    SYS_KERNEL=$(ls /boot/vmlinuz-* 2>/dev/null | head -n1 || true)
    if [ -n "$SYS_KERNEL" ]; then
        cp "$SYS_KERNEL" "$VMLINUZ"
        echo "[build] Using system kernel: $SYS_KERNEL"
    else
        echo "[build] ERROR: No kernel found at /boot/vmlinuz-*"
        echo "[build] Install linux-image-generic and retry."
        exit 1
    fi
fi
echo "[build] vmlinuz: $(du -h "$VMLINUZ" | cut -f1)"

# ── Create ISO staging ───────────────────────────────────────────────────────
ISO_STAGING="$SCRIPT_DIR/iso_staging"
rm -rf "$ISO_STAGING"
mkdir -p "$ISO_STAGING/boot/grub"

cp "$VMLINUZ" "$ISO_STAGING/boot/vmlinuz"
cp "$INITRD_IMG" "$ISO_STAGING/boot/initrd.img"

cat > "$ISO_STAGING/boot/grub/grub.cfg" <<'GRUBEOF'
set timeout=0
set default=0

menuentry "Ubuntu Benchmark" {
    linux /boot/vmlinuz root=/dev/ram0 rw quiet console=ttyS0,115200n8 panic=1
    initrd /boot/initrd.img
}
GRUBEOF

# ── Build ISO ────────────────────────────────────────────────────────────────
ISO_OUT="$SCRIPT_DIR/ubuntu-bench.iso"

if command -v grub-mkrescue &>/dev/null; then
    echo "[build] Building ISO with grub-mkrescue..."
    grub-mkrescue -o "$ISO_OUT" "$ISO_STAGING" 2>/dev/null || {
        echo "[build] grub-mkrescue failed, using xorriso fallback..."
        GRUB_MKFRESCUE_FALLBACK=1
    }
fi

if [ ! -f "$ISO_OUT" ]; then
    GRUB_MKFRESCUE_FALLBACK=1
fi

if [ "${GRUB_MKFRESCUE_FALLBACK:-}" = "1" ]; then
    echo "[build] Building ISO with xorriso fallback..."

    # Build a small FAT12 EFI image containing GRUB
    EFI_IMG="$SCRIPT_DIR/efiboot.img"
    dd if=/dev/zero of="$EFI_IMG" bs=1M count=4 2>/dev/null
    mkfs.vfat -F 12 "$EFI_IMG" >/dev/null 2>&1

    mmd -i "$EFI_IMG" ::/EFI >/dev/null 2>&1
    mmd -i "$EFI_IMG" ::/EFI/BOOT >/dev/null 2>&1

    # Locate GRUB EFI binary
    GRUB_EFI=""
    for path in /usr/lib/grub/x86_64-efi-signed/grubx64.efi \
                /usr/lib/grub/x86_64-efi/monolithic/grubx64.efi \
                /usr/lib/grub/x86_64-efi/grubx64.efi \
                /boot/efi/EFI/ubuntu/grubx64.efi; do
        if [ -f "$path" ]; then
            GRUB_EFI="$path"
            break
        fi
    done

    if [ -z "$GRUB_EFI" ]; then
        echo "[build] ERROR: Could not find grubx64.efi"
        exit 1
    fi

    mcopy -i "$EFI_IMG" "$GRUB_EFI" ::/EFI/BOOT/BOOTX64.EFI >/dev/null 2>&1
    cp "$EFI_IMG" "$ISO_STAGING/boot/grub/efiboot.img"

    xorriso -as mkisofs \
        -iso-level 3 \
        -V "UBUNTUBENCH" \
        -J -R \
        -eltorito-platform efi \
        -eltorito-boot boot/grub/efiboot.img \
        -no-emul-boot \
        -o "$ISO_OUT" \
        "$ISO_STAGING"
fi

if [ -f "$ISO_OUT" ]; then
    echo "[build] OK: $ISO_OUT ($(du -h "$ISO_OUT" | cut -f1))"
else
    echo "[build] ERROR: ISO creation failed"
    exit 1
fi

# ── Cleanup ──────────────────────────────────────────────────────────────────
rm -rf "$ISO_STAGING" "$INITRAMFS_DIR" "$INITRD_IMG" 2>/dev/null || true
# Keep vmlinuz for incremental rebuilds, but clean up transient artifacts
[ -f "$SCRIPT_DIR/efiboot.img" ] && rm -f "$SCRIPT_DIR/efiboot.img"

trap - EXIT
echo "[build] Done."
