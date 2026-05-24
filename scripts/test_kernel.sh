#!/usr/bin/env bash
# Seal OS — Kernel integration test runner
# Builds the kernel with test-mode, creates a UEFI disk image, and runs it in QEMU.
# Skips gracefully if QEMU or OVMF firmware is not available.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ── Detect QEMU ──────────────────────────────────────────────────────────────
QEMU="qemu-system-x86_64"
if ! command -v "$QEMU" &>/dev/null; then
    echo "[SKIP] QEMU not available ($QEMU not in PATH)"
    exit 0
fi

# ── Detect OVMF firmware ─────────────────────────────────────────────────────
OVMF_PATH=""
for candidate in \
    /usr/share/OVMF/OVMF_CODE_4M.fd \
    /usr/share/OVMF/OVMF_CODE.fd \
    /usr/share/edk2-ovmf/x64/OVMF_CODE.fd \
    /usr/share/qemu/edk2-x86_64-code.fd \
    /usr/share/qemu-efi-x86_64/QEMU_EFI.fd
do
    if [ -f "$candidate" ]; then
        OVMF_PATH="$candidate"
        break
    fi
done

if [ -z "$OVMF_PATH" ]; then
    echo "[SKIP] OVMF UEFI firmware not found in any standard location"
    echo "[SKIP] Checked: /usr/share/OVMF/* /usr/share/edk2-ovmf/* /usr/share/qemu/*"
    exit 0
fi

echo "[test_kernel] QEMU: $(command -v "$QEMU")"
echo "[test_kernel] OVMF: $OVMF_PATH"

# ── Build Seal OS with test-mode ─────────────────────────────────────────────
echo "[test_kernel] Building Seal OS with test-mode (release)..."
cd "$PROJECT_ROOT/kernel/seal-os"
cargo +nightly build --release --features test-mode

# ── Build disk image ─────────────────────────────────────────────────────────
echo "[test_kernel] Building disk image..."
cd "$PROJECT_ROOT/kernel/seal-mkimage"
cargo +stable build --release
cargo +stable run --release

IMG="$PROJECT_ROOT/kernel/seal-os/target/x86_64-unknown-uefi/release/seal-os.img"
if [ ! -f "$IMG" ]; then
    echo "[test_kernel] FAIL: Disk image not found at $IMG"
    exit 1
fi

# ── Run kernel tests in QEMU ─────────────────────────────────────────────────
echo "[test_kernel] Running kernel tests in QEMU..."
LOGFILE=$(mktemp)
QEMU_EXIT=0
timeout 10 "$QEMU" \
    -machine q35 \
    -drive if=pflash,format=raw,readonly=on,file="$OVMF_PATH" \
    -drive file="$IMG",format=raw \
    -nographic \
    -m 4G \
    -no-reboot \
    -no-shutdown \
    -device isa-debug-exit,iobase=0x501,iosize=0x04 \
    > "$LOGFILE" 2>&1 || QEMU_EXIT=$?

echo "[test_kernel] QEMU exited with code: $QEMU_EXIT"
echo "[test_kernel] Serial output:"
cat "$LOGFILE"

# isa-debug-exit: success = exit code 1, failure = exit code 3
if [ "$QEMU_EXIT" -eq 1 ]; then
    if grep -q "ALL TESTS PASSED" "$LOGFILE"; then
        echo "[test_kernel] PASS: All tests passed"
        rm -f "$LOGFILE"
        exit 0
    fi
fi

if [ "$QEMU_EXIT" -eq 3 ]; then
    echo "[test_kernel] FAIL: Tests reported failure via isa-debug-exit"
    rm -f "$LOGFILE"
    exit 1
fi

# Fallback: grep the log for the pass/fail message
if grep -q "ALL TESTS PASSED" "$LOGFILE"; then
    echo "[test_kernel] PASS: All tests passed (detected from log)"
    rm -f "$LOGFILE"
    exit 0
fi

if grep -q "TEST FAILED" "$LOGFILE"; then
    echo "[test_kernel] FAIL: Tests failed (detected from log)"
    rm -f "$LOGFILE"
    exit 1
fi

echo "[test_kernel] FAIL: Unexpected QEMU exit code $QEMU_EXIT"
rm -f "$LOGFILE"
exit 1
