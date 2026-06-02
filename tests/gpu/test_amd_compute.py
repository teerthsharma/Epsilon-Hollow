#!/usr/bin/env python3
"""Seal OS AMD GPU Hardware Dispatch Proof — host-side test.

Prerequisites:
- Linux host with AMD GPU bound to vfio-pci (or amdgpu if testing native)
- QEMU 6.0+ with VFIO support
- Seal OS build artifact (target/release/seal-os.iso or similar)

Usage:
    pytest tests/gpu/test_amd_compute.py -v
    # or
    python3 tests/gpu/test_amd_compute.py --serial-log build/serial.log

The test looks for the canonical sentinel in Seal OS serial output:
    [GPU-BENCH] voronoi result=OK cycles=<n>
"""

import re
import subprocess
import sys
import argparse
from pathlib import Path

SENTINEL_RE = re.compile(r"\[GPU-BENCH\] voronoi result=OK cycles=(\d+)")
FAIL_RE = re.compile(r"\[GPU-BENCH\] voronoi result=FAIL reason=(\S+)")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ISO = PROJECT_ROOT / "target" / "release" / "seal-os.iso"
QEMU_SCRIPT = PROJECT_ROOT / "scripts" / "run_qemu_gpu.sh"


def run_qemu_passthrough(gpu_bdf: str, iso: Path, timeout: int = 120) -> str:
    """Boot Seal OS with AMD GPU passthrough and capture serial output."""
    cmd = [
        "bash",
        str(QEMU_SCRIPT),
        "--passthrough", gpu_bdf,
        "--headless",
        "--iso", str(iso),
    ]
    print(f"[TEST] Running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    # Serial output is usually on stdout; merge stderr just in case.
    return result.stdout + "\n" + result.stderr


def parse_serial_output(text: str) -> dict:
    """Scan serial text for GPU-BENCH sentinels."""
    ok_match = SENTINEL_RE.search(text)
    fail_match = FAIL_RE.search(text)
    return {
        "ok": ok_match is not None,
        "cycles": int(ok_match.group(1)) if ok_match else None,
        "fail_reason": fail_match.group(1) if fail_match else None,
    }


def test_amd_compute_passthrough():
    """End-to-end test: QEMU + AMD GPU passthrough -> voronoi dispatch proof."""
    if not DEFAULT_ISO.exists():
        pytest.skip(f"ISO not found: {DEFAULT_ISO}")
    if not QEMU_SCRIPT.exists():
        pytest.skip(f"QEMU script not found: {QEMU_SCRIPT}")

    # Detect AMD GPU BDF on host.
    lspci = subprocess.run(["lspci", "-nn"], capture_output=True, text=True)
    amd_gpus = [line.split()[0] for line in lspci.stdout.splitlines() if "1002:" in line]
    if not amd_gpus:
        pytest.skip("No AMD GPU found on host (lspci -nn)")

    gpu_bdf = amd_gpus[0]
    serial_text = run_qemu_passthrough(gpu_bdf, DEFAULT_ISO)
    result = parse_serial_output(serial_text)

    assert result["ok"], (
        f"AMD hardware dispatch did not print OK sentinel. "
        f"fail_reason={result['fail_reason']}"
    )
    assert result["cycles"] is not None and result["cycles"] > 0, (
        "Cycle count missing or zero"
    )
    print(f"[TEST] PASS: voronoi dispatch OK in {result['cycles']} cycles")


def test_amd_compute_from_log():
    """Unit-style test: parse a pre-captured serial log for the sentinel."""
    # This test always runs and validates regex parsing against a synthetic log.
    sample_log = (
        "[BOOT] driver init: gpu done\n"
        "[AMD-ACCEL] Ready — BAR0=00000000F0000000\n"
        "[GPU-BENCH] voronoi result=OK cycles=12345\n"
        "[GPU-BENCH] suite complete\n"
    )
    result = parse_serial_output(sample_log)
    assert result["ok"] is True
    assert result["cycles"] == 12345


def main():
    parser = argparse.ArgumentParser(description="AMD GPU compute proof test")
    parser.add_argument("--serial-log", type=Path, help="Pre-captured serial log to parse")
    parser.add_argument("--gpu-bdf", help="AMD GPU PCI BDF (e.g. 0000:0a:00.0)")
    parser.add_argument("--iso", type=Path, default=DEFAULT_ISO, help="Seal OS ISO path")
    args = parser.parse_args()

    if args.serial_log:
        text = args.serial_log.read_text()
        result = parse_serial_output(text)
        if result["ok"]:
            print(f"[PASS] Sentinel found: OK cycles={result['cycles']}")
            sys.exit(0)
        else:
            print(f"[FAIL] Sentinel missing. fail_reason={result['fail_reason']}")
            sys.exit(1)

    if args.gpu_bdf:
        serial_text = run_qemu_passthrough(args.gpu_bdf, args.iso)
        result = parse_serial_output(serial_text)
        if result["ok"]:
            print(f"[PASS] Sentinel found: OK cycles={result['cycles']}")
            sys.exit(0)
        else:
            print(f"[FAIL] Sentinel missing. fail_reason={result['fail_reason']}")
            sys.exit(1)

    parser.print_help()
    sys.exit(2)


if __name__ == "__main__":
    main()
