#!/usr/bin/env bash
# Seal OS — Kernel integration test runner
# Builds the kernel with test-mode, creates a UEFI disk image, and runs it in QEMU.

set -euo pipefail

PROJECT_ROOT="$(dirname "$0")/.."

echo "[test_kernel] Building Seal OS with test-mode (release)..."
cd "$PROJECT_ROOT/kernel/seal-os"
cargo +nightly build --release --features test-mode

echo "[test_kernel] Building disk image..."
cd "$PROJECT_ROOT/kernel/seal-mkimage"
cargo build --release
cargo run --release

IMG="$PROJECT_ROOT/kernel/seal-os/target/x86_64-unknown-uefi/release/seal-os.img"
if [ ! -f "$IMG" ]; then
    echo "[test_kernel] FAIL: Disk image not found at $IMG"
    exit 1
fi

echo "[test_kernel] Running kernel tests in QEMU..."
LOGFILE=$(mktemp)
QEMU_EXIT=0
qemu-system-x86_64 \
    -bios /usr/share/OVMF/OVMF_CODE.fd \
    -drive file="$IMG",format=raw \
    -nographic \
    -m 4G \
    -no-reboot \
    -no-shutdown \
    -device isa-debug-exit,iobase=0x501,iosize=0x04 \
    -serial stdio > "$LOGFILE" 2>&1 || QEMU_EXIT=$?

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
