"""The two inputs a distro hands its kernel besides the root disk.

A distro's boot entry passes a command line (`root=UUID=... ro quiet
console=ttyS0`) and an initramfs (`initrd /boot/initrd.img-...`). The kernel
parses the command line to find its root device and console, then unpacks the
initramfs cpio into a rootfs and runs `/init` from it, which loads modules and
pivots to the real root. Without both, an unmodified distro install does not
reach userspace.

Seal's UEFI entry opens the LoadedImage protocol but consumes neither: grep of
kernel/seal-os/src finds no command-line parsing (`root=`, `console=`,
LoadOptions decode) and no initramfs/cpio unpacking. These tests pin those two
milestones.
"""

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "kernel/seal-os/src"


def src_text():
    parts = []
    for path in SRC.rglob("*.rs"):
        parts.append(re.sub(r"//.*|/\*.*?\*/", "", path.read_text(encoding="utf-8", errors="replace"), flags=re.S))
    return "\n".join(parts)


def test_kernel_parses_a_boot_command_line():
    text = src_text()
    # A real consumer decodes UEFI LoadOptions (or a multiboot cmdline) and looks
    # for at least root= / console=. Mentioning LoadedImage is not consuming it.
    signals = [r"load_options", r"\broot=", r"console=", r"parse_cmdline", r"boot_?args", r"command_line"]
    hits = [s for s in signals if re.search(s, text)]
    assert hits, (
        "kernel/seal-os/src parses no boot command line "
        f"(none of {signals}); a distro's root=UUID/console= line is ignored"
    )


def test_kernel_unpacks_an_initramfs():
    text = src_text()
    signals = [r"initramfs", r"initrd", r"cpio", r"newc", r"InitrdMedia", r"LoadFile2"]
    hits = [s for s in signals if re.search(s, text, re.I)]
    assert hits, (
        "kernel/seal-os/src has no initramfs path "
        f"(none of {signals}); a distro's initrd.img is never unpacked, so no early userspace or module load"
    )
