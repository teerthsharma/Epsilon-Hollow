"""F2 — The image builder cannot place a userland on the medium.

"Run the distro's userland unmodified" requires the userland to be on the disk
the kernel mounts at "/". seal-mkimage's build path is
`create_disk_image` = MBR + GPT + FAT32-ESP(kernel .efi) + a 512-byte MNFD
superblock. It never writes a file, inode, or directory entry into the root
(MNFD) partition, and it exposes no flag, env var, or host directory to ingest
a rootfs. This test proves that behaviorally.

Method: run the built seal-mkimage with rootfs-ingestion-style flags and a host
file carrying a unique marker, then scan the produced image for that marker.
Control: the image is produced and contains the kernel PE payload (proves
mkimage ran and built a real image, so a missing marker is a real capability
gap, not a broken invocation).
Property: the marker (a "userland" file) reaches the image. It never does -> RED.
"""
import os
import subprocess
import uuid
from pathlib import Path

from conftest import REPO_ROOT, DISK_IMAGE

MKIMAGE_DIR = REPO_ROOT / "kernel" / "seal-mkimage"
MKIMAGE_EXE = MKIMAGE_DIR / "target" / "release" / "seal-mkimage.exe"


def _run_mkimage(extra_args: list[str]) -> bytes:
    assert MKIMAGE_EXE.exists(), f"setup: build seal-mkimage first ({MKIMAGE_EXE})"
    subprocess.run(
        [str(MKIMAGE_EXE), *extra_args],
        cwd=str(MKIMAGE_DIR),
        check=False,
        capture_output=True,
        timeout=120,
    )
    assert DISK_IMAGE.exists(), "setup: mkimage did not produce seal-os.img"
    return DISK_IMAGE.read_bytes()


def test_builder_produces_a_real_image_control():
    data = _run_mkimage([])
    assert len(data) == 128 * 1024 * 1024, "control: image is the expected size"
    assert data.count(b"PE\x00\x00") >= 1, "control: kernel PE payload is in the image"


def test_builder_can_place_a_host_userland_file():
    marker = f"FOREMAN_ROOTFS_PROBE_{uuid.uuid4().hex}".encode()
    tmpdir = Path(os.environ.get("TEMP", "/tmp")) / f"foreman_rootfs_{uuid.uuid4().hex}"
    (tmpdir / "bin").mkdir(parents=True, exist_ok=True)
    hostfile = tmpdir / "bin" / "init"
    hostfile.write_bytes(b"\x7fELF" + marker + b"\x00" * 64)

    # Try the flag surface a rootfs-capable image builder would plausibly expose.
    placed = False
    for args in (
        ["--rootfs", str(tmpdir)],
        ["--add-file", f"/bin/init={hostfile}"],
        ["--overlay", str(tmpdir)],
    ):
        data = _run_mkimage(args)
        if marker in data:
            placed = True
            break

    assert placed, (
        "seal-mkimage has no way to place a host userland file into the image: "
        "no --rootfs / --add-file / --overlay ingestion path exists, and the "
        "build writes only a 512-byte MNFD superblock into the root partition. "
        "The distro userland can never reach the disk the kernel mounts at '/'."
    )
