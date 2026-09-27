"""The Linux ABI surface the smallest real static binary exercises, against Seal OS.

The fixture is measured, not recalled: `gcc -static -O2` of a one-line
`puts` program (glibc 2.43, Ubuntu 26.04, gcc 15.2.0) run on Linux
6.6.87.2-microsoft-standard-WSL2 under a ptrace syscall tracer that prints
`orig_rax` at every syscall-entry stop. It printed, in order:

    12 12 158 218 273 334 302 318 267 12 12 12 10 5 1 231

Names are from that system's /usr/include/x86_64-linux-gnu/asm/unistd_64.h,
which defines 385 syscall numbers in total.

These tests pin what an unmodified Linux binary needs from the kernel before
its first byte of output, which is the first measurable milestone of both the
"replace the distro kernel" proposal and a Linux-personality layer beside the
Seal ABI.
"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TABLE = REPO / "kernel/seal-os/src/syscall/table.rs"
USERSPACE = REPO / "kernel/seal-os/src/process/userspace.rs"
ELF = REPO / "kernel/seal-os/src/process/elf.rs"

GLIBC_STATIC_HELLO_TRACE = [12, 12, 158, 218, 273, 334, 302, 318, 267, 12, 12, 12, 10, 5, 1, 231]
LINUX_NAMES = {
    1: "write", 5: "fstat", 10: "mprotect", 12: "brk", 158: "arch_prctl",
    218: "set_tid_address", 231: "exit_group", 267: "readlinkat",
    273: "set_robust_list", 302: "prlimit64", 318: "getrandom", 334: "rseq",
}


def seal_numbers():
    """{number: lowercase name} for every `pub const SYS_<NAME>: u64 = <n>;` in the table."""
    text = TABLE.read_text(encoding="utf-8")
    return {int(n): name.lower() for name, n in re.findall(r"pub const SYS_([A-Z0-9_]+): u64 = (\d+);", text)}


def test_static_glibc_hello_syscalls_reach_the_linux_handler():
    seal = seal_numbers()
    wrong = []
    for nr in sorted(set(GLIBC_STATIC_HELLO_TRACE)):
        linux = LINUX_NAMES[nr]
        got = seal.get(nr)
        if got != linux:
            wrong.append(f"{nr} {linux} -> " + (f"Seal SYS_{got.upper()}" if got else "ENOSYS"))
    distinct = len(set(GLIBC_STATIC_HELLO_TRACE))
    assert not wrong, f"{len(wrong)}/{distinct} syscalls of a static glibc hello misroute:\n  " + "\n  ".join(wrong)


def test_syscall_dispatch_carries_six_arguments():
    # Linux x86_64 passes rdi, rsi, rdx, r10, r8, r9; mmap(2) uses all six.
    sig = re.search(r"pub fn dispatch\(([^)]*)\)", TABLE.read_text(encoding="utf-8")).group(1)
    args = re.findall(r"\barg\d\b", sig)
    body = re.search(r"fn do_syscall\(.*?\n}\n", USERSPACE.read_text(encoding="utf-8"), re.S).group(0)
    regs_read = sorted(set(re.findall(r"f\.(rdi|rsi|rdx|r10|r8|r9)\b", body)))
    assert len(args) == 6, f"dispatch({sig}) takes {len(args)} args; do_syscall reads {regs_read}"


def test_elf_loader_builds_the_initial_stack_libc_reads():
    # glibc/musl static start-up reads argc, argv, envp, then the aux vector;
    # glibc dereferences AT_RANDOM for its stack-protector and pointer-guard canaries.
    text = ELF.read_text(encoding="utf-8")
    tags = ["AT_NULL", "AT_PHDR", "AT_PHNUM", "AT_PAGESZ", "AT_ENTRY", "AT_RANDOM"]
    missing = [t for t in tags if t not in text]
    pushes = re.findall(r"push_stack_u64\(sp, USER_STACK_TOP, top_frame, (\w+)\)", text)
    assert not missing, f"elf.rs names none of {missing}; initial stack is {len(pushes)} pushes of {pushes}"
