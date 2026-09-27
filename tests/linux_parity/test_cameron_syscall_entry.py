"""The SYSCALL entry path a systems or security reviewer reads first.

`syscall` does not change RSP, GS or RFLAGS beyond what IA32_FMASK clears
(Intel SDM Vol. 2B, SYSCALL). A kernel entry stub therefore has to leave the
user stack before its first memory write, and FMASK has to clear the flags
ring 0 code must not inherit from ring 3: AC (a set AC disables SMAP for the
whole handler), DF (Rust and memcpy assume DF=0) and TF. Linux sets
FMASK = TF|DF|IF|IOPL|AC|NT for the same reasons.

Both properties sit on the path every syscall of every userland takes, Seal
ABI or Linux ABI, so they are a precondition of either route.
"""

import re
from pathlib import Path

USERSPACE = Path(__file__).resolve().parents[2] / "kernel/seal-os/src/process/userspace.rs"


def entry_instructions():
    text = USERSPACE.read_text(encoding="utf-8")
    block = re.search(r'"syscall_entry:",(.*?)\);', text, re.S).group(1)
    return re.findall(r'"([^"]+)"', block)


def test_syscall_entry_leaves_the_user_stack_before_its_first_store():
    insns = entry_instructions()
    first_store = next(i for i, ins in enumerate(insns) if ins.startswith("push") or "[rsp" in ins)
    before = insns[:first_store]
    switched = any(re.match(r"(mov|xchg|lea)\s+rsp\s*,", ins) for ins in before)
    assert "swapgs" in before and switched, (
        f"first store is insn #{first_store} '{insns[first_store]}' on the caller-chosen RSP; "
        f"preceded by {before or 'nothing'}"
    )


def test_fmask_clears_ac_df_tf_and_if():
    value = int(re.search(r"Msr::new\(MSR_SFMASK\)\.write\((0x[0-9A-Fa-f_]+)\)", USERSPACE.read_text(encoding="utf-8"))
                .group(1).replace("_", ""), 16)
    need = {"TF": 0x100, "IF": 0x200, "DF": 0x400, "AC": 0x40000}
    kept = [flag for flag, bit in need.items() if not value & bit]
    assert not kept, f"IA32_FMASK = {value:#x}; ring 0 inherits user {kept}"
