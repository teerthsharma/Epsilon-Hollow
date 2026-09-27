"""What the Seal OS syscall path does with the calls unmodified Linux binaries make.

Linux side: chase_linux_measured.json, written by chase_measure_linux.sh (a
ptrace count of every syscall entry made by five Ubuntu 26.04 programs,
threads and children included). Kernel side: kernel/seal-os/src as it stands.
"""

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEAL = HERE.parents[1] / "kernel" / "seal-os" / "src"
MEASURED = json.loads((HERE / "chase_linux_measured.json").read_text())

# Linux call -> Seal ABI calls accepted as the same operation. Deliberately
# generous: any Seal handler that could serve the Linux call counts.
TWINS = {
    "read": {"READ"}, "write": {"WRITE"}, "open": {"OPEN"}, "openat": {"OPEN"},
    "close": {"CLOSE"}, "fstat": {"STAT"}, "newfstatat": {"STAT"}, "statx": {"STAT"},
    "lseek": {"LSEEK"}, "mmap": {"MMAP"}, "brk": {"BRK"}, "ioctl": {"IOCTL"},
    "pipe": {"PIPE"}, "pipe2": {"PIPE"}, "dup2": {"DUP2"}, "getpid": {"GETPID"},
    "getppid": {"GETPPID"}, "clone": {"CLONE", "FORK"}, "execve": {"EXEC"},
    "exit_group": {"EXIT"}, "wait4": {"WAITPID"}, "rt_sigaction": {"SIGACTION"},
    "rt_sigreturn": {"SIGRETURN"}, "sigaltstack": {"SIGALTSTACK"},
    "getrandom": {"GETRANDOM"}, "prlimit64": {"GETRLIMIT", "SETRLIMIT"},
}


def seal_syscalls():
    """Seal ABI number -> name, from every `pub const SYS_*` in src/syscall."""
    table = {}
    for f in (SEAL / "syscall").glob("*.rs"):
        for name, nr in re.findall(r"pub const SYS_([A-Z0-9_]+): u64 = (\d+);", f.read_text()):
            table[int(nr)] = name
    return table


def issued_by_linux_programs():
    """Linux syscall number -> (name, programs that issued it)."""
    issued = {}
    for prog, calls in MEASURED["syscalls_by_program"].items():
        for name, v in calls.items():
            issued.setdefault(v["nr"], (name, []))[1].append(prog)
    return issued


def test_linux_syscall_numbers_reach_the_same_call_in_seal_dispatch():
    seal = seal_syscalls()
    issued = issued_by_linux_programs()
    same, misrouted, enosys = [], [], []
    for nr, (name, _progs) in sorted(issued.items()):
        target = seal.get(nr)
        if target is None:
            enosys.append(f"{nr} {name}")
        elif target in TWINS.get(name, set()):
            same.append(name)
        else:
            misrouted.append(f"{nr} {name}->SYS_{target}")
    assert not misrouted and not enosys, (
        f"{len(same)}/{len(issued)} Linux syscall numbers issued by "
        f"{sorted(MEASURED['syscalls_by_program'])} ({MEASURED['provenance']['distro']}) "
        f"reach the same call in Seal dispatch ({', '.join(same)}). "
        f"{len(misrouted)} reach a DIFFERENT Seal call: {'; '.join(misrouted)}. "
        f"{len(enosys)} reach no handler (ENOSYS): {', '.join(enosys)}"
    )


def test_elf_loader_builds_the_linux_process_entry_stack():
    """x86-64 psABI process entry: argc, argv[], NULL, envp[], NULL, then an
    auxiliary vector; glibc's static start-up reads AT_PHDR (TLS), AT_PAGESZ
    and AT_RANDOM (stack-protector canary) from it."""
    elf = (SEAL / "process" / "elf.rs").read_text()
    body = elf[elf.index("pub fn load("):]
    body = body[: body.index("\n}\n")]
    pushes = re.findall(r"push_stack_u64\(sp, USER_STACK_TOP, top_frame, ([^)]*)\)", body)
    auxv = {"AT_PHDR": 3, "AT_PAGESZ": 6, "AT_ENTRY": 9, "AT_RANDOM": 25}
    present = [k for k in auxv if k in body]
    assert present == list(auxv), (
        f"elf::load builds the entry stack from {len(pushes)} push(es) of {pushes} "
        f"and names auxv entries {present or 'none'} of {list(auxv)}"
    )


def test_syscall_dispatch_takes_all_six_linux_argument_registers():
    table = (SEAL / "syscall" / "table.rs").read_text()
    params = re.search(r"pub fn dispatch\(([^)]*)\)", table).group(1)
    args = [p for p in params.split(",") if p.strip() and not p.strip().startswith("num")]
    entry = (SEAL / "process" / "userspace.rs").read_text()
    body = entry[entry.index("fn do_syscall("):]
    body = body[: body.index("\n}\n")]
    abi = ["rdi", "rsi", "rdx", "r10", "r8", "r9"]
    forwarded = [r for r in abi if re.search(rf"\bf\.{r}\b", body)]
    six_arg_users = sorted(p for p, c in MEASURED["syscalls_by_program"].items() if "mmap" in c)
    assert len(args) == 6 and forwarded == abi, (
        f"dispatch() takes {len(args)} argument(s) and do_syscall forwards "
        f"{len(forwarded)} of 6 Linux argument registers ({', '.join(forwarded)}); "
        f"dropped: {', '.join(r for r in abi if r not in forwarded)}. "
        f"mmap(addr, len, prot, flags, fd, off) needs all six and is issued by "
        f"{len(six_arg_users)}/{len(MEASURED['syscalls_by_program'])} traced programs: {six_arg_users}"
    )
