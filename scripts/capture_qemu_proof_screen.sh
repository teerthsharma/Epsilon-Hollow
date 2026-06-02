#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "usage: $0 <seal-os.img> <serial.log> <screen.ppm>" >&2
  exit 2
fi

IMG_PATH="$1"
LOG="$2"
PPM="$3"
OVMF_CODE="${OVMF_CODE:-/usr/share/OVMF/OVMF_CODE_4M.fd}"
OVMF_VARS_TEMPLATE="${OVMF_VARS:-/usr/share/OVMF/OVMF_VARS_4M.fd}"
SECONDS_LIMIT="${SEAL_QEMU_SCREEN_SECONDS:-240}"
MON="${SEAL_QEMU_MONITOR:-/tmp/seal-qemu-monitor-$$.sock}"
OVMF_VARS_COPY="/tmp/seal_OVMF_VARS_$$.fd"

command -v qemu-system-x86_64 >/dev/null || {
  echo "FAIL: qemu-system-x86_64 not found" >&2
  exit 1
}
command -v socat >/dev/null || {
  echo "FAIL: socat not found; install socat for QEMU monitor screenshot capture" >&2
  exit 1
}

if [ ! -f "$IMG_PATH" ]; then
  echo "FAIL: disk image not found: $IMG_PATH" >&2
  exit 1
fi
if [ ! -f "$OVMF_CODE" ]; then
  echo "FAIL: OVMF code file not found: $OVMF_CODE" >&2
  exit 1
fi
if [ ! -f "$OVMF_VARS_TEMPLATE" ]; then
  echo "FAIL: OVMF vars template not found: $OVMF_VARS_TEMPLATE" >&2
  exit 1
fi

mkdir -p "$(dirname "$LOG")" "$(dirname "$PPM")"
rm -f "$MON" "$LOG" "$PPM" "$OVMF_VARS_COPY"
cp "$OVMF_VARS_TEMPLATE" "$OVMF_VARS_COPY"

qemu-system-x86_64 \
  -machine q35 \
  -m 4096 \
  -cpu qemu64 \
  -smp 2 \
  -drive if=pflash,format=raw,readonly=on,file="$OVMF_CODE" \
  -drive if=pflash,format=raw,file="$OVMF_VARS_COPY" \
  -device ahci,id=seal_sata \
  -drive if=none,id=seal_disk,file="$IMG_PATH",format=raw,media=disk \
  -device ide-hd,drive=seal_disk,bus=seal_sata.0,unit=0 \
  -serial file:"$LOG" \
  -monitor unix:"$MON",server,nowait \
  -display none \
  -device VGA \
  -no-reboot \
  -no-shutdown &
QEMU_PID=$!

cleanup() {
  rm -f "$MON" "$OVMF_VARS_COPY"
}
trap cleanup EXIT

deadline=$((SECONDS_LIMIT + $(date +%s)))
verdict="TIMEOUT"
while kill -0 "$QEMU_PID" 2>/dev/null; do
  if [ -f "$LOG" ] &&
    grep -F "[BOOT] Desktop proof frame blit done" "$LOG" >/dev/null &&
    grep -F "[BOOT] Seal OS desktop ready." "$LOG" >/dev/null &&
    grep -F "[EVENT] Entering real event loop" "$LOG" >/dev/null; then
    verdict="PASS"
    break
  fi
  if [ -f "$LOG" ] && grep -E "!!! SEAL OS KERNEL PANIC !!!|\[FAULT\]|\[WATCHDOG\]|gurumeditation" "$LOG" >/dev/null; then
    verdict="FAIL"
    break
  fi
  if [ "$(date +%s)" -ge "$deadline" ]; then
    break
  fi
  sleep 1
done

if [ "$verdict" = "PASS" ]; then
  sleep 2
  for _ in 1 2 3 4 5; do
    [ -S "$MON" ] && break
    sleep 1
  done
  if [ ! -S "$MON" ]; then
    echo "FAIL: QEMU monitor socket missing: $MON" >&2
    verdict="FAIL"
  else
    printf 'screendump %s\nquit\n' "$PPM" | socat - "UNIX-CONNECT:$MON" >/dev/null
  fi
fi

if kill -0 "$QEMU_PID" 2>/dev/null; then
  kill "$QEMU_PID" 2>/dev/null || true
fi
wait "$QEMU_PID" 2>/dev/null || true

if [ "$verdict" != "PASS" ]; then
  echo "QEMU proof-screen capture verdict: $verdict" >&2
  tail -n 120 "$LOG" 2>/dev/null || true
  exit 1
fi

if [ ! -s "$PPM" ]; then
  echo "FAIL: QEMU proof-screen PPM missing or empty: $PPM" >&2
  tail -n 120 "$LOG" 2>/dev/null || true
  exit 1
fi

echo "QEMU proof-screen capture OK: $PPM"
