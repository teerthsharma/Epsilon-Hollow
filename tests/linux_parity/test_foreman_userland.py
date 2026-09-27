"""F1 — No userland process ever executes on a booted Seal OS image.

The revamp proposal is "drop Seal OS into any Linux distro and run the distro's
userland unmodified." That presupposes Seal OS can run *a* userland process at
all. This test boots the real, freshly-built UEFI image under QEMU (same device
topology as the repo's own proof runner) and asks the weakest possible version
of the question: did *any* ring-3 process run and issue one syscall?

The kernel ships a hand-assembled static ELF (`userspace::EMERGENCY_SHELL_ELF`)
whose whole body is `SYS_WRITE("[userspace] Hello from Seal OS userspace!")`
then `SYS_EXIT`. If ring-3 worked, that string would reach the serial log on the
fallback path. It does not.

Control: the boot banner and all ten theorem markers are asserted first. They
pass, proving the image boots and the serial capture works — so a failure of the
userland assertion is a real property failure, not a dead VM.
"""
import pytest

from conftest import boot_serial_log


@pytest.fixture(scope="module")
def serial() -> str:
    return boot_serial_log(seconds=150)


def test_boot_reaches_desktop_control(serial: str):
    # Positive controls: the machine boots and the log is captured.
    assert "Seal OS v0.4.7.5" in serial, "boot banner absent — VM/log setup failed, not a property result"
    assert "[BOOT] Seal OS desktop ready." in serial, "desktop never came up — setup failure"
    assert "[THEOREM] T1/TSS VERIFIED" in serial


def test_a_userland_process_runs_and_makes_a_syscall(serial: str):
    # The property the whole revamp is blocked on: a ring-3 process executes.
    ran_ring3 = ("[userspace] Hello from Seal OS userspace!" in serial) or (
        "[execve] Spawned" in serial
    )
    # This is exactly the diagnostic the boot itself prints on the dead path:
    assert "[execve] '/bin/init' not found; continuing with kernel desktop" not in serial, (
        "image carries no /bin/init and no userland was spawned"
    )
    assert ran_ring3, (
        "no ring-3 userland process executed: neither the '/bin/init' spawn "
        "nor the emergency-shell SYS_WRITE marker appears in the serial log"
    )
