"""What "use Linux drivers" costs Seal OS, priced on a real distro's modules.

Linux side: chase_linux_measured.json, written by chase_measure_linux.sh from
every .ko under /lib/modules/$(uname -r)/kernel (modinfo license/vermagic,
nm -u imports, __versions section). Repo side: deny.toml and
kernel/seal-os/src as they stand.
"""

import json
import re
import statistics
import tomllib
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEAL = ROOT / "kernel" / "seal-os" / "src"
MEASURED = json.loads((HERE / "chase_linux_measured.json").read_text())
MODULES = MEASURED["modules"]
DRIVERS = [m for m in MODULES if m["path"].startswith("drivers/")]

# MODULE_LICENSE tag -> licenses the module may be taken under, per the
# kernel's Documentation/process/license-rules.rst ("MODULE_LICENSE" table).
# "BSD" stands for whichever BSD variant the source file names.
TAG_OFFERS = {
    "GPL": {"GPL-2.0"},
    "GPL v2": {"GPL-2.0"},
    "GPL and additional rights": {"GPL-2.0", "MIT"},
    "Dual BSD/GPL": {"GPL-2.0", "BSD"},
    "Dual MIT/GPL": {"GPL-2.0", "MIT"},
    "Dual MPL/GPL": {"GPL-2.0", "MPL-1.1"},
}


def admitted(tag, allow):
    offers = TAG_OFFERS.get(tag, set())
    return any(o in allow or (o == "BSD" and any(a.startswith("BSD-") for a in allow)) for o in offers)


def test_repo_license_gate_admits_the_linux_driver_modules():
    allow = set(tomllib.loads((ROOT / "deny.toml").read_text())["licenses"]["allow"])
    refused = Counter(m["license"] for m in MODULES if not admitted(m["license"], allow))
    refused_drivers = sum(1 for m in DRIVERS if not admitted(m["license"], allow))
    assert not refused, (
        f"deny.toml [licenses].allow admits {len(MODULES) - sum(refused.values())}/{len(MODULES)} "
        f"modules of {MEASURED['provenance']['distro']} kernel {MEASURED['provenance']['kernel']}; "
        f"refused {sum(refused.values())} ({dict(refused)}), "
        f"{refused_drivers}/{len(DRIVERS)} of them under drivers/. The gate bans GPL-*; these tags offer no other licence."
    )


def seal_c_exports():
    names = set()
    for f in SEAL.rglob("*.rs"):
        names |= set(re.findall(
            r'#\[no_mangle\]\s*(?:pub(?:\([a-z]+\))?\s+)?(?:unsafe\s+)?extern\s+"C"\s+fn\s+(\w+)',
            f.read_text(errors="ignore")))
    # compiler_builtins links these into every no_std Rust binary.
    return names | {"memcpy", "memmove", "memset", "memcmp", "bcmp"}


def test_seal_provides_the_kernel_symbols_linux_driver_modules_import():
    union = set(MEASURED["driver_import_union"])
    have = union & seal_c_exports()
    per = [m["n_imports"] for m in DRIVERS]
    (vermagic, pinned), = Counter(MEASURED["module_vermagic"]).most_common(1)
    crcs = sum(m["has_versions"] for m in MODULES)
    core = ["_printk", "kfree", "__kmalloc", "mutex_lock", "pci_enable_device",
            "request_threaded_irq", "dma_alloc_attrs", "register_netdev"]
    classes = {m["path"].rsplit("/", 1)[1]: (m["n_imports"], m["license"]) for m in DRIVERS
               if m["path"].rsplit("/", 1)[1] in ("virtio_blk.ko", "e1000e.ko", "xhci-hcd.ko", "nvmet.ko")}
    assert have == union, (
        f"Seal OS exports {len(have)} of the {len(union)} kernel symbols imported by "
        f"{len(DRIVERS)} Linux {MEASURED['provenance']['kernel']} driver modules "
        f"(per-module imports min {min(per)} / median {statistics.median(per)} / max {max(per)}). "
        f"Binary reuse is also pinned: {pinned}/{len(MODULES)} modules carry vermagic {vermagic.strip()!r} "
        f"and {crcs}/{len(MODULES)} carry __versions symbol CRCs. "
        f"Spec-class drivers alone (imports, licence): {classes}. "
        f"Missing core API sample: {[n for n in core if n in union - have]}"
    )
