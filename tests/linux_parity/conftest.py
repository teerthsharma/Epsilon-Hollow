"""Shared locators for the Linux-parity diagnosis tests.

These tests inspect the Seal OS kernel source and the built UEFI image. Paths
are anchored at the repository root (three levels above this file) so the suite
runs from any worktree.
"""
import os
import shutil
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SEAL_OS_SRC = REPO_ROOT / "kernel" / "seal-os" / "src"
RELEASE_DIR = (
    REPO_ROOT / "kernel" / "seal-os" / "target" / "x86_64-unknown-uefi" / "release"
)
DISK_IMAGE = RELEASE_DIR / "seal-os.img"
EFI_BINARY = RELEASE_DIR / "seal-os.efi"


def read_src(*parts: str) -> str:
    return (SEAL_OS_SRC.joinpath(*parts)).read_text(encoding="utf-8", errors="replace")


def find_qemu() -> str | None:
    cand = shutil.which("qemu-system-x86_64")
    if cand:
        return cand
    pf = os.environ.get("ProgramFiles", r"C:\Program Files")
    p = Path(pf) / "qemu" / "qemu-system-x86_64.exe"
    return str(p) if p.exists() else None


def find_ovmf() -> str | None:
    pf = os.environ.get("ProgramFiles", r"C:\Program Files")
    for name in ("edk2-x86_64-code.fd", "OVMF_CODE.fd", "OVMF_CODE_4M.fd"):
        p = Path(pf) / "qemu" / "share" / name
        if p.exists():
            return str(p)
    return None


def boot_serial_log(seconds: int = 150) -> str:
    """Boot the built seal-os.img headless under QEMU and return the serial log.

    Uses the exact device topology of the repo's own run-qemu proof runner
    (q35 + AHCI ide-hd + OVMF pflash). Returns the captured serial text.
    """
    qemu = find_qemu()
    ovmf = find_ovmf()
    if qemu is None or ovmf is None or not DISK_IMAGE.exists():
        raise RuntimeError(
            f"missing prerequisite: qemu={qemu} ovmf={ovmf} image_exists={DISK_IMAGE.exists()}"
        )
    tmp = REPO_ROOT / "tests" / "linux_parity" / "_serial.log"
    if tmp.exists():
        tmp.unlink()
    args = [
        qemu, "-machine", "q35", "-m", "4096", "-cpu", "qemu64", "-smp", "2",
        "-drive", f"if=pflash,format=raw,readonly=on,file={ovmf}",
        "-device", "ahci,id=seal_sata",
        "-drive", f"if=none,id=seal_disk,file={DISK_IMAGE},format=raw,media=disk",
        "-device", "ide-hd,drive=seal_disk,bus=seal_sata.0,unit=0",
        "-serial", f"file:{tmp}",
        "-no-reboot", "-display", "none", "-device", "VGA",
    ]
    proc = subprocess.Popen(args)
    try:
        time.sleep(seconds)
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
    if not tmp.exists():
        raise RuntimeError("QEMU produced no serial log")
    return tmp.read_text(encoding="utf-8", errors="replace")
