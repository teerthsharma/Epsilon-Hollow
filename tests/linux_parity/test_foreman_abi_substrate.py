"""F3 — The ring-3 ABI substrate is unbuilt: no auxv on the initial stack, and
the syscall entry stub never switches to a kernel stack.

These are the two layer-specific facts underneath "run the distro's userland."
Whatever ABI Seal OS targets (its own or Linux), a process image cannot start
and cannot trap correctly without them.

  (a) ELF startup contract. Every x86-64 SysV `_start` (Seal's own static ELF
      included) reads argc / argv / envp / the auxiliary vector off the initial
      stack. glibc/musl static binaries abort without AT_RANDOM/AT_PHDR/AT_PAGESZ.
      The loader in process/elf.rs pushes three zero words and nothing else.

  (b) SYSCALL entry contract. On `syscall` from ring 3 the CPU keeps the *user*
      RSP; the kernel entry stub must `swapgs` and load a kernel stack (from the
      TSS or a per-CPU area) before it can safely push. syscall_entry's first
      instruction is `push r15` against the user RSP; `swapgs` appears nowhere.

Each test asserts a positive control (the loader / the stub exist and are wired)
so the failing assertion is a genuine missing-property, not a moved file.
"""
from conftest import read_src


def test_initial_user_stack_has_auxv():
    elf = read_src("process", "elf.rs")
    # Control: the loader exists and builds a user stack + entry.
    assert "pub stack_pointer" in elf
    assert "USER_STACK_TOP" in elf
    assert "push_stack_u64" in elf, "control: the stack-building primitive exists"
    # Property: a SysV auxiliary vector is constructed. Any one of these tokens
    # would indicate the loader lays down argc/argv/envp/auxv.
    auxv_tokens = ["AT_RANDOM", "AT_PHDR", "AT_PAGESZ", "AT_ENTRY", "AT_NULL", "auxv"]
    present = [t for t in auxv_tokens if t in elf]
    assert present, (
        "process/elf.rs builds no auxiliary vector: the initial user stack is "
        "three zero words (see push_stack_u64 x3), so a SysV/Linux _start reads "
        "garbage for argc/argv/auxv. tokens_found=" + repr(present)
    )


def test_syscall_entry_switches_to_kernel_stack():
    us = read_src("process", "userspace.rs")
    # Control: the fast-syscall entry stub and its return exist and are programmed.
    assert "syscall_entry:" in us
    assert "sysretq" in us
    assert "MSR_LSTAR" in us, "control: LSTAR is programmed to the entry stub"
    # Property: the stub swaps GS and loads a kernel stack before pushing state.
    assert "swapgs" in us, (
        "syscall_entry never issues swapgs and never loads a kernel RSP: it "
        "pushes register state onto whatever stack ring 3 held. A real ring-3 "
        "`syscall` cannot be serviced safely."
    )
