"""Can a distro's own bootloader entry load this kernel?

"Distro userland unmodified" includes the distro's boot configuration: GRUB
and systemd-boot load a kernel through a `linux /boot/vmlinuz-... root=... ` +
`initrd /boot/initrd.img` entry. Both read the Linux x86 setup header to do it
(Documentation/arch/x86/boot.rst): the four bytes at offset 0x202 of the image
are the magic `HdrS`, and 0x1fe..0x200 hold the boot flag 0xAA55. A modern
vmlinuz is a PE/COFF EFI-stub that ALSO carries that setup header, so the same
file boots via GRUB, systemd-boot and direct UEFI.

Seal ships EFI/BOOT/BOOTX64.EFI and is launched as a bare UEFI application. To
replace `/boot/vmlinuz-*` under an existing GRUB/sd-boot entry, its image has
to carry the setup header and accept an initrd and a command line the way that
protocol specifies. These tests pin that milestone.

Measured on the built image
kernel/seal-os/target/x86_64-unknown-uefi/release/seal-os.efi (91f857e):
bytes 0x200.. are `05 01 00 00 00 30 0d 01`, 0x1fe.. are `61 6d`, and `HdrS`
appears nowhere in the first 64 KiB.
"""

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
EFI = REPO / "kernel/seal-os/target/x86_64-unknown-uefi/release/seal-os.efi"
MKIMAGE = REPO / "kernel/seal-mkimage/src/main.rs"


@pytest.mark.skipif(not EFI.exists(), reason="build seal-os.efi first (TESTING.md)")
def test_kernel_image_carries_the_linux_setup_header():
    head = EFI.read_bytes()[:0x10000]
    assert head[:2] == b"MZ", "not even a PE image"
    boot_flag = head[0x1FE:0x200]
    hdrs = head[0x202:0x206]
    assert boot_flag == b"\x55\xAA" and hdrs == b"HdrS", (
        f"no Linux setup header: boot flag 0x1fe = {boot_flag.hex()} (want 55aa), "
        f"0x202 = {hdrs!r} (want b'HdrS'); GRUB/systemd-boot `linux` cannot load this image"
    )


def test_mkimage_emits_a_distro_installable_vmlinuz():
    # The image builder must produce a vmlinuz-shaped artifact and stage an initrd
    # + kernel command line, not only an EFI/BOOT/BOOTX64.EFI standalone app.
    text = MKIMAGE.read_text(encoding="utf-8").lower()
    markers = ["hdrs", "setup_header", "bzimage", "vmlinuz", "initramfs", "initrd", "cmdline"]
    present = [m for m in markers if m in text]
    assert present, (
        "seal-mkimage references none of "
        f"{markers}; it builds a standalone UEFI app, not a /boot/vmlinuz + initrd a distro installs"
    )
