#!/usr/bin/env bash
# Run Seal OS in QEMU with UEFI firmware (OVMF)
set -e

IMG="target/x86_64-unknown-uefi/release/seal-os.img"

if [ ! -f "$IMG" ]; then
    echo "No disk image found. Building..."
    cargo +nightly build --release
    (cd ../seal-mkimage && cargo +stable run --release)
fi

# Default OVMF paths on common Linux distros
OVMF_CODE=""
for path in \
    /usr/share/OVMF/OVMF_CODE_4M.fd \
    /usr/share/ovmf/OVMF.fd \
    /usr/share/OVMF/OVMF_CODE.fd \
    /usr/share/qemu/OVMF_CODE.fd \
    /usr/share/edk2-ovmf/x64/OVMF_CODE.fd \
    /usr/share/edk2/x64/OVMF_CODE.fd; do
    if [ -f "$path" ]; then
        OVMF_CODE="$path"
        break
    fi
done

if [ -z "$OVMF_CODE" ]; then
    echo "WARNING: OVMF firmware not found. Trying without pflash (may fail for UEFI .efi)."
    qemu-system-x86_64 \
        -machine q35 \
        -device ahci,id=seal_sata \
        -drive if=none,id=seal_disk,file="$IMG",format=raw,media=disk \
        -device ide-hd,drive=seal_disk,bus=seal_sata.0,unit=0 \
        -serial stdio \
        -m 4G \
        -vga std \
        -no-reboot \
        -no-shutdown
else
    qemu-system-x86_64 \
        -machine q35 \
        -drive if=pflash,format=raw,readonly=on,file="$OVMF_CODE" \
        -device ahci,id=seal_sata \
        -drive if=none,id=seal_disk,file="$IMG",format=raw,media=disk \
        -device ide-hd,drive=seal_disk,bus=seal_sata.0,unit=0 \
        -serial stdio \
        -m 4G \
        -vga std \
        -no-reboot \
        -no-shutdown
fi
