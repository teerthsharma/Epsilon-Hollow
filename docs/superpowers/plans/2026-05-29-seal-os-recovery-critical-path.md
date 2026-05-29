# Seal OS Recovery Critical Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop scope drift and move Seal OS from "boots in QEMU" toward "VM-proven topological OS with persistent storage truth" using only evidence-backed fixes.

**Architecture:** Main session owns integration and final proof. Agents may audit independent lanes, but no agent deletes or rewrites source without a Rust/Aether replacement and a passing gate. The next hard blocker is storage: QEMU now exposes an AHCI controller, but the OS still falls back to ramfs because no SATA disk is detected.

**Tech Stack:** Bare-metal Rust `no_std`, x86_64 UEFI, QEMU/WSL2, Oracle VirtualBox, PowerShell runners, `seal-mkimage` Rust audit gates, Aether-Lang, assembly edges. No host Python commands.

---

## Scope Lock

This pass is not "make every OS feature better than Ubuntu." That claim needs benchmark data. This pass is the recovery path:

- Keep QEMU boot proof green.
- Make AHCI/SATA disk detection real or document the exact red gap.
- Keep T1-T10, proof screen, Seal ABI, language hygiene, and O(1) allocator gates green.
- Keep Aether-Lang parser/runtime app contract green.
- Keep README as the bible: checked boxes only for proven claims.
- Keep Oracle VirtualBox as a target, but call out host service failure separately from OS boot failure.

## File Map

- Modify: `kernel/seal-os/run-qemu.ps1`
  - Owns QEMU proof disk wiring. Must use a VM-visible disk controller.
- Modify if root cause proves it: `kernel/seal-os/src/drivers/pci.rs`
  - Owns PCI command bits for MMIO and bus mastering.
- Modify if root cause proves it: `kernel/seal-os/src/drivers/block/ahci.rs`
  - Owns block-layer AHCI initialization and port probing.
- Modify if root cause proves it: `kernel/seal-os/src/drivers/disk/ahci.rs`
  - Owns disk-layer AHCI probing and persistent-root discovery path.
- Modify: `README.md`
  - Owns public status table and topological OS claim discipline.
- Modify: `docs/VM_RUNBOOK.md`
  - Owns QEMU/Oracle operator truth.
- Modify: `docs/SEAL_OS_GUIDE.md`
  - Owns long-form proof baseline, gaps, and benchmark status.

## Task 1: Freeze Current Proof

- [ ] **Step 1: Build kernel**

Run from `C:\Users\seal\Documents\GitHub\Epsilon-Hollow\kernel\seal-os`:

```powershell
cargo +nightly build --release
```

Expected:

```text
Finished `release` profile
```

- [ ] **Step 2: Rebuild image**

Run from `C:\Users\seal\Documents\GitHub\Epsilon-Hollow\kernel\seal-mkimage`:

```powershell
cargo +stable run --release
```

Expected:

```text
[mkimage] IMG:
[mkimage] EFI boot image:
```

- [ ] **Step 3: QEMU proof**

Run from `C:\Users\seal\Documents\GitHub\Epsilon-Hollow\kernel\seal-os`:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run-qemu.ps1 -HeadlessProof -ProofSeconds 240
```

Expected:

```text
QEMU proof completed
```

- [ ] **Step 4: Read storage evidence**

Run from `C:\Users\seal\Documents\GitHub\Epsilon-Hollow\kernel\seal-os`:

```powershell
rg -n "\[AHCI\]|\[disk::ahci\]|\[VFS\]|ramfs|persistent|No SATA|SATA device" target\x86_64-unknown-uefi\release\qemu-proof\serial.log
```

Expected current red evidence if still broken:

```text
[AHCI] Found controller
[AHCI] No SATA device found
[disk::ahci] No AHCI device present
[VFS] No persistent disk found. Falling back to ramfs.
```

## Task 2: AHCI Root Cause Before More Patches

- [ ] **Step 1: Finish agent diagnosis**

Wait once for the AHCI/QEMU audit agent. Use its findings only as evidence; main session still reviews code.

- [ ] **Step 2: Check whether QEMU exposes the disk on another controller**

Run from `C:\Users\seal\Documents\GitHub\Epsilon-Hollow\kernel\seal-os`:

```powershell
rg -n "\[PCI\]|class=0x01|subclass=0x06|subclass=0x08|AHCI|NVMe" target\x86_64-unknown-uefi\release\qemu-proof\serial.log
```

Expected:

```text
At least one storage controller line.
```

- [ ] **Step 3: Inspect AHCI probe behavior**

Read these files:

```text
kernel/seal-os/src/drivers/block/ahci.rs
kernel/seal-os/src/drivers/disk/ahci.rs
kernel/seal-os/src/drivers/pci.rs
```

Root-cause questions:

- Does the code scan all AHCI controllers or only the first matching PCI device?
- Does it enable PCI Memory Space before reading ABAR?
- Does it skip ports with invalid `ssts` or only rely on signature?
- Does QEMU need a different `-device` wiring for the disk to appear on the implemented port?

- [ ] **Step 4: Patch exactly one confirmed root cause**

Allowed fixes:

- QEMU runner fix in `run-qemu.ps1` if the disk is wired to the wrong bus.
- PCI command fix in `pci.rs` if MMIO/bus mastering is not enabled.
- AHCI scan fix in `block/ahci.rs` and `disk/ahci.rs` if only the first empty controller is inspected.
- AHCI port detection fix if `ssts` reports presence but signature read is invalid.

Not allowed:

- Fake success logs.
- Marking persistent root green while ramfs fallback is still used.
- Broad driver rewrite without a smaller proof.

## Task 3: Re-Prove After AHCI Patch

- [ ] **Step 1: Format**

Run from repo root:

```powershell
cargo +stable fmt --check
```

Expected: exit code 0.

- [ ] **Step 2: Kernel build**

Run from `kernel\seal-os`:

```powershell
cargo +nightly build --release
```

Expected: exit code 0.

- [ ] **Step 3: Rebuild image**

Run from `kernel\seal-mkimage`:

```powershell
cargo +stable run --release
```

Expected: exit code 0.

- [ ] **Step 4: QEMU proof**

Run from `kernel\seal-os`:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run-qemu.ps1 -HeadlessProof -ProofSeconds 240
```

Expected: exit code 0.

- [ ] **Step 5: Audit gates**

Run from `kernel\seal-mkimage`:

```powershell
cargo +stable run --release -- --verify ..\seal-os\target\x86_64-unknown-uefi\release\seal-os.img ..\seal-os\target\x86_64-unknown-uefi\release\seal-os.efi
cargo +stable run --release -- --check-theorem-log ..\seal-os\target\x86_64-unknown-uefi\release\qemu-proof\serial.log
cargo +stable run --release -- --check-proof-screen ..\seal-os\target\x86_64-unknown-uefi\release\qemu-proof\screen.ppm
cargo +stable run --release -- --check-seal-abi ..\..
cargo +stable run --release -- --check-language-hygiene ..\..
cargo +stable run --release -- --check-o1-allocator ..\..
```

Expected:

```text
[mkimage] VERIFY OK
[seal-audit] THEOREM LOG OK
[seal-audit] PROOF SCREEN OK
[seal-audit] SEAL ABI OK
[seal-audit] LANGUAGE HYGIENE OK
[seal-audit] O(1) ALLOCATOR OK
```

## Task 4: Aether-Lang App Contract

- [ ] **Step 1: Run targeted Aether parser/runtime tests**

Run from repo root:

```powershell
cargo +stable test -p aether-lang parser::tests::parses_tilde_statement_terminators
cargo +stable test -p aether-lang parser::tests::parses_windowed_app_host_script_shape
cargo +stable test -p aether-lang interpreter::tests::test_windowed_module_methods_execute_without_callbacks
```

Expected: all pass.

- [ ] **Step 2: Check QEMU serial for Aether app failure**

Run from `kernel\seal-os`:

```powershell
rg -n "AetherAppHost|script init error|parse error|Method .* not found" target\x86_64-unknown-uefi\release\qemu-proof\serial.log
```

Expected:

```text
No parser/runtime error lines.
```

## Task 5: Oracle VirtualBox Truth

- [ ] **Step 1: Build VDI**

Run from `kernel\seal-os`:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\build-vbox.ps1
```

Expected:

```text
seal-os.vdi
```

- [ ] **Step 2: Oracle smoke only if host VirtualBox responds**

Run from `kernel\seal-os`:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\smoke-vbox.ps1 -Seconds 240 -VBoxCommandTimeoutSeconds 30
```

Pass condition:

```text
[BOOT] Seal OS desktop ready.
```

Known host-blocked condition:

```text
VBoxManage createvm/showvminfo times out before boot.
```

If host-blocked, docs must say Oracle proof is blocked by local VirtualBox service/COM state, not by a Seal OS boot panic.

## Task 6: README And Runbook Truth

- [ ] **Step 1: Update `README.md` proof table**

Required statuses:

```markdown
| Claim | Status | Proof |
| --- | --- | --- |
| QEMU boots to desktop event loop | Verified | `run-qemu.ps1 -HeadlessProof` |
| T1-T10 theorem gate | Verified | `seal-mkimage --check-theorem-log` |
| Seal ABI/no-POSIX gate | Verified | `seal-mkimage --check-seal-abi` |
| O(1) allocator source gate | Verified | `seal-mkimage --check-o1-allocator` |
| Aether-Lang boot app syntax | Verified | targeted `aether-lang` tests |
| Persistent ManifoldFS root | Verified in QEMU | `seal-mkimage --check-vm-proof` requires AHCI disk, block device `0x800`, readable sector 0, and `[VFS] ManifoldFS mounted from disk` |
| Oracle VirtualBox smoke | Pending or blocked | `smoke-vbox.ps1` fresh result |
| Beats Ubuntu for HFT/ML | Not yet proven | benchmark suite pending |
```

- [ ] **Step 2: Fix broken Mermaid only where touched**

Run from repo root:

```powershell
rg -n "```mermaid|<br>|\\(|\\)|-->|subgraph|end" README.md docs
```

Fix only malformed Mermaid blocks in README/docs touched by this pass. Do not rewrite diagrams as decoration.

- [ ] **Step 3: Update `docs/VM_RUNBOOK.md`**

Must include:

- QEMU command.
- Oracle command.
- Current storage-controller expectation.
- Current known host-blocked VirtualBox condition if still true.
- Exact serial log and screenshot artifact paths.

## Task 7: Final Audit

- [ ] **Step 1: Gather exact proof paths**

Use:

```text
kernel/seal-os/target/x86_64-unknown-uefi/release/qemu-proof/serial.log
kernel/seal-os/target/x86_64-unknown-uefi/release/qemu-proof/screen.ppm
kernel/seal-os/target/x86_64-unknown-uefi/release/qemu-proof/screen.png
kernel/seal-os/target/x86_64-unknown-uefi/release/seal-os.img
kernel/seal-os/target/x86_64-unknown-uefi/release/seal-os.efi
```

- [ ] **Step 2: Final report table**

Report these rows:

- QEMU boot proof.
- Desktop proof screen.
- T1-T10 theorem gate.
- O(1) allocator gate.
- Seal ABI/no-POSIX gate.
- Aether-Lang app contract.
- Persistent storage/AHCI.
- Oracle VirtualBox.
- Ubuntu benchmark status.

Each row must say `proved`, `red`, or `blocked`, with evidence.

## Execution Order

1. Task 1: Freeze current proof.
2. Task 2: AHCI root cause before more patches.
3. Task 3: Re-prove after AHCI patch.
4. Task 4: Aether-Lang app contract.
5. Task 5: Oracle VirtualBox truth.
6. Task 6: README and runbook truth.
7. Task 7: Final audit.

## Stop Conditions

Stop and report if:

- QEMU no longer reaches desktop event loop.
- Any theorem gate fails.
- O(1) allocator gate fails.
- Aether parser/runtime app tests fail.
- AHCI fix attempts exceed three without root cause. At that point, question driver architecture before more patches.

## Self-Review

- Spec coverage: boot proof, T1-T10, Aether-Lang, no-POSIX, O(1), Oracle, README truth, Ubuntu benchmark honesty.
- Placeholder scan: no empty work markers. Every task has commands and pass/fail evidence.
- Type consistency: file paths match current repo layout and previous proof artifacts.
- Scope check: this is a recovery lane, not a fake "whole OS finished" claim.
