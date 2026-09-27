"""F4 — "Use Linux drivers" has no mechanism. This is a different layer from the
userland-ABI half of the proposal, and a far larger one.

Reusing a Linux driver means running Linux kernel-internal driver code, which
binds to the Linux in-kernel API (kmalloc, the DMA/PCI subsystems, the driver
model, workqueues, sysfs kobjects, request_firmware, module loading, ...). That
requires either a shim implementing that unstable ~30k-symbol API, or a module
loader that ingests `.ko` objects. Seal OS has neither: its drivers are
hand-written native Rust bound to a handful of PCI IDs.

Control: the native driver tree is real (virtio_blk/virtio_net init fns exist).
Property: some Linux-driver reuse mechanism exists (module loader, linuxkpi/LKL
shim, .ko ingestion). None does -> RED.
"""
from conftest import SEAL_OS_SRC, read_src


def _grep_tree(needles: list[str]) -> list[str]:
    hits: list[str] = []
    for path in SEAL_OS_SRC.rglob("*.rs"):
        text = path.read_text(encoding="utf-8", errors="replace")
        for n in needles:
            if n in text:
                hits.append(f"{path.name}:{n}")
    return hits


def test_native_drivers_exist_control():
    # Control: the hand-written native drivers are present and initialized.
    assert "pub fn init()" in read_src("drivers", "net", "virtio_net.rs")
    assert "pub fn init()" in read_src("drivers", "block", "virtio_blk.rs")


def test_linux_driver_reuse_mechanism_exists():
    needles = [
        "finit_module",
        "init_module",
        "insmod",
        "linuxkpi",
        "LinuxKPI",
        "struct module",
        "load_module",
        "module_init",
        "request_firmware",
        "pci_driver",  # the Linux driver-model registration struct
        ".ko",
    ]
    hits = _grep_tree(needles)
    assert hits, (
        "no Linux-driver reuse mechanism anywhere in kernel/seal-os/src: no "
        "module loader, no linuxkpi/LKL shim, no .ko ingestion, no Linux "
        "driver-model binding. 'Use Linux drivers' is unimplemented and shares "
        f"no layer with the userland-ABI half of the proposal. hits={hits}"
    )
