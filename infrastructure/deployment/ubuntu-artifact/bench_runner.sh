#!/bin/sh
# Ubuntu 24.04 Benchmark Runner — initramfs PID 1
# Runs 5 standard workloads and outputs [UBUNTU-BENCH] lines for JSON parsing.

set -e

mount -t proc proc /proc
mount -t sysfs sys /sys
mount -t devtmpfs dev /dev 2>/dev/null || {
    mknod /dev/null    c 1 3
    mknod /dev/zero    c 1 5
    mknod /dev/random  c 1 8
    mknod /dev/urandom c 1 9
    mknod /dev/console c 5 1
    mknod /dev/tty     c 5 0
    chmod 666 /dev/null /dev/zero /dev/random /dev/urandom
    chmod 622 /dev/console /dev/tty
}
mount -t tmpfs tmpfs /tmp

mkdir -p /dev/pts
mount -t devpts devpts /dev/pts 2>/dev/null || true

echo ""
echo "=================================="
echo "  Ubuntu 24.04 Benchmark Suite"
echo "=================================="
echo ""

ITERATIONS=64
if [ -f /bench_iterations ]; then
    ITERATIONS=$(cat /bench_iterations)
fi

echo "[ubuntu-bench] suite=ubuntu-benchmark version=1.0.0 iterations=${ITERATIONS}"
echo ""

for wl in alloc-frame mem-bandwidth fs-teleport sched-yield tcp-demux; do
    echo "[ubuntu-bench] workload=${wl} status=begin"
    /bin/workloads "$wl" "$ITERATIONS"
    echo "[ubuntu-bench] workload=${wl} status=end"
    echo ""
done

echo "[ubuntu-bench] suite_status=complete"

sync
sleep 1

# Power off using whatever method is available
poweroff -f 2>/dev/null || \
    reboot -f 2>/dev/null || \
    { echo 1 > /proc/sys/kernel/sysrq 2>/dev/null; echo o > /proc/sysrq-trigger 2>/dev/null; } || \
    while :; do sleep 1; done
