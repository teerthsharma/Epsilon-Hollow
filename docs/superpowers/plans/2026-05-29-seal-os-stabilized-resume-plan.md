# Seal OS Stabilized Resume Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resume Seal OS work from the current proved QEMU baseline and move only through audited, measurable gates.

**Architecture:** Treat the README as the product contract, but treat proof artifacts as the source of truth. The kernel remains bare-metal Rust/no_std with Aether-Lang as the OS language layer, ManifoldFS as persistent topology storage, and QEMU/Oracle as VM proof targets.

**Tech Stack:** Bare-metal Rust, x86_64 UEFI, QEMU, Oracle VirtualBox, PowerShell runners, Rust `seal-mkimage` audit gates, Aether-Lang. No host Python commands.

---

## Current Proved Baseline

These facts are the starting checkpoint. Do not re-litigate them unless a later command fails.

| Area | Current state | Required artifact |
| --- | --- | --- |
| QEMU boot | Proved | `kernel/seal-os/target/x86_64-unknown-uefi/release/qemu-proof/serial.log` |
| Desktop | Proved | `kernel/seal-os/target/x86_64-unknown-uefi/release/qemu-proof/screen.png` |
| T1-T10 gate | Proved | `seal-mkimage --check-theorem-log ...\serial.log` |
| Persistent ManifoldFS root | Proved | `seal-mkimage --check-vm-proof ...\serial.log` |
| AHCI disk | Proved in QEMU | serial lines for `QEMU HARDDISK`, block device `0x800`, and sector 0 read |
| Aether boot app tests | Proved for targeted parser/runtime contract | targeted `cargo +stable test -p aether-lang ...` tests |
| Seal ABI/no-POSIX | Proved by Rust scanner | `seal-mkimage --check-seal-abi ..\..` |
| O(1) allocator source gate | Proved by Rust scanner | `seal-mkimage --check-o1-allocator ..\..` |
| Oracle VM | Not yet proved on this host | host VirtualBox service/registry is the suspect until fresh run says otherwise |
| Ubuntu superiority | Not proved | benchmark artifacts against same-machine Ubuntu are still required |

## Agent Dispatch Matrix

Use one fresh agent per lane. Agents may inspect and report; only the coordinator merges edits after review.

| Agent | Scope | May edit | Must not edit | Output |
| --- | --- | --- | --- | --- |
| Newton | VM operator: QEMU, Oracle, disk image, proof artifacts | `kernel/seal-os/run-qemu.ps1`, `kernel/seal-os/smoke-vbox.ps1`, `docs/VM_RUNBOOK.md` | kernel theorem code, allocator code | VM status table with exact command output summary |
| Noether | Theorem and O(1) audit: T1-T10, allocator, scheduler, topology claims | docs and Rust audit checker only after finding mismatch | VM scripts | list of claims that are mechanically checked vs only documented |
| Hilbert | Aether-Lang and Python-to-Rust migration audit | Rust Aether crates and docs | VM scripts | list of remaining Python surface and corresponding Rust replacement path |
| Curie | Graphics/desktop proof audit | `kernel/seal-os/src/wm/*`, `kernel/seal-os/src/graphics/*` | storage, allocator | screenshot assessment and minimal desktop fix list |
| Gauss | README/mermaid/benchmark truth | `README.md`, `docs/*.md`, benchmark docs | kernel code | README delta with only true tick marks |

## Task 1: Freeze Baseline Before New Edits

**Files:**
- Read: `kernel/seal-os/target/x86_64-unknown-uefi/release/qemu-proof/serial.log`
- Read: `kernel/seal-os/target/x86_64-unknown-uefi/release/qemu-proof/screen.png`
- Read: `kernel/seal-mkimage/src/main.rs`

- [ ] **Step 1: Re-run the Rust proof gates from repo root**

Run:

```powershell
cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml --release -- --check-vm-proof kernel\seal-os\target\x86_64-unknown-uefi\release\qemu-proof\serial.log
cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml --release -- --check-theorem-log kernel\seal-os\target\x86_64-unknown-uefi\release\qemu-proof\serial.log
cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml --release -- --check-proof-screen kernel\seal-os\target\x86_64-unknown-uefi\release\qemu-proof\screen.ppm
cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml --release -- --check-seal-abi .
cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml --release -- --check-o1-allocator .
```

Expected: all commands exit 0. If any fail, stop execution and fix that gate before touching unrelated code.

- [ ] **Step 2: Record the exact proof lines**

Required serial lines:

```text
[BOOT] GOP mode set to 1024x768
[BOOT] All T1-T10 theorems VERIFIED
[AHCI] Device model: QEMU HARDDISK
[AHCI] Registered as block device 0x800
[disk::ahci] First disk readable (sector 0 OK)
[VFS] ManifoldFS mounted from disk
[BOOT] Desktop proof frame blit done
[BOOT] Seal OS desktop ready.
[EVENT] Entering real event loop
```

Expected: every line is present in `serial.log`, with no `Falling back to ramfs` and no `No AHCI device present`.

## Task 2: Put VM Proof Into CI/Local Gates

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `scripts/local_ci.sh`
- Read: `kernel/seal-os/run-qemu.ps1`
- Read: `docs/CI.md`
- Read: `docs/LOCAL_CI.md`

- [ ] **Step 1: Inspect existing CI VM command**

Run:

```powershell
rg -n "qemu|seal-os.img|check-theorem-log|check-proof-screen|check-vm-proof|timeout" .github\workflows\ci.yml scripts\local_ci.sh docs\CI.md docs\LOCAL_CI.md
```

Expected: identify whether CI already attaches the raw disk through AHCI and whether it runs `--check-vm-proof`.

- [ ] **Step 2: Add the strong VM proof gate**

CI must run the same checker used locally:

```powershell
cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml --release -- --check-vm-proof /tmp/seal-os.log
```

Expected: the CI job fails if storage falls back to ramfs, AHCI is absent, desktop does not initialize, or theorem proof is missing.

- [ ] **Step 3: Keep CI honest about graphical proof**

If CI can run graphical QEMU with screenshot capture, keep `--check-proof-screen`. If CI is `-nographic` only, document that screenshot proof remains a local/headless artifact from `run-qemu.ps1 -HeadlessProof`.

Expected: docs never imply CI proved a screen when CI did not capture one.

## Task 3: Oracle VM Truth

**Files:**
- Read/modify if needed: `kernel/seal-os/smoke-vbox.ps1`
- Read/modify if needed: `kernel/seal-os/build-vbox.ps1`
- Modify: `docs/VM_RUNBOOK.md`
- Modify: `README.md`

- [ ] **Step 1: Build or verify VDI conversion**

Run from `kernel\seal-os`:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\build-vbox.ps1
```

Expected: `target\x86_64-unknown-uefi\release\seal-os.vdi` exists, or the command gives a clear VirtualBox/VBoxManage failure.

- [ ] **Step 2: Attempt Oracle smoke only after VBoxManage responds**

Run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\smoke-vbox.ps1
```

Expected: either a fresh Oracle boot proof, or a precise host-blocked reason such as VirtualBox COM/service timeout. Do not call this an OS failure unless serial/VM evidence shows kernel failure.

- [ ] **Step 3: Update status**

README row must say one of:

```markdown
| Oracle VirtualBox automated smoke | ✓ | ... fresh artifact path ... |
| Oracle VirtualBox automated smoke | △ | VDI conversion works; host VirtualBox service/registry blocked fresh smoke |
| Oracle VirtualBox automated smoke | ✗ | ... exact OS boot failure ... |
```

Expected: one row, one truth, no hype.

## Task 4: Aether-Lang And Python Migration Audit

**Files:**
- Read/modify if needed: `kernel/aether/Aether-Lang/crates/aether-lang/src/parser.rs`
- Read/modify if needed: `kernel/aether/Aether-Lang/crates/aether-lang/src/interpreter.rs`
- Read/modify if needed: `kernel/epsilon/epsilon/crates/aether-core/src/*.rs`
- Audit deleted legacy files: `kernel/epsilon/epsilon_core/*.py`

- [ ] **Step 1: Prove no production Python dependency remains**

Run:

```powershell
rg -n "\.py|python|Python|meta_controller|hyperbolic_geometry|spectral_entropy|governor_convergence" kernel README.md docs
```

Expected: Python references are either historical docs, explicit migration notes, or deleted legacy paths with Rust equivalents. No boot/build/runtime path depends on Python.

- [ ] **Step 2: Run targeted Aether tests**

Run:

```powershell
cargo +stable test -p aether-lang parser::tests::parses_tilde_statement_terminators
cargo +stable test -p aether-lang parser::tests::parses_windowed_app_host_script_shape
cargo +stable test -p aether-lang interpreter::tests::test_windowed_module_methods_execute_without_callbacks
```

Expected: all pass.

- [ ] **Step 3: Document remaining migration gaps**

If any deleted Python file lacks a Rust equivalent, document the gap in `docs/SEAL_OS_GUIDE.md` and add a Rust task before removing the historical claim from README.

Expected: no silent deletion of Python logic.

## Task 5: Topological OS Claim Audit

**Files:**
- Read: `README.md`
- Read: `docs/TOPOLOGICAL_OS_CONTRACT.md`
- Read: `docs/MANIFOLDFS-O1-DESIGN.md`
- Read: `kernel/seal-os/src/memory/phys.rs`
- Read: `kernel/seal-os/src/memory/topo_ram.rs`
- Read: `kernel/seal-os/src/process/scheduler.rs`
- Read: `kernel/seal-os/src/fs/block_store.rs`

- [ ] **Step 1: Map each big claim to a proof**

Create this table in the final audit response:

```markdown
| Claim | Current proof | Remaining proof gap |
| --- | --- | --- |
| T1-T10 theorem gate | boot serial + checker | theorem-to-runtime coverage still needs deeper formal proof |
| Single-frame O(1) allocation | source scanner + boot marker | latency benchmark pending |
| Persistent ManifoldFS root | AHCI serial + VM checker | Oracle hardware path pending |
| File teleportation O(1) | metadata design/code path | benchmark artifact pending |
| VRAM fast path | design contract | GPU driver/VRAM allocator implementation pending unless code proves otherwise |
| HFT/ML lead over Ubuntu | benchmark plan | same-machine numbers pending |
```

Expected: no README row gets a check mark unless code or artifact proves it.

- [ ] **Step 2: Fix any false ticks**

If README marks a benchmark or hardware feature as done without artifact, change it to `△` and point to the next gate.

Expected: README becomes "bible" because it is strict, not because it shouts.

## Task 6: Desktop/Graphics Polish Only After Proof Is Frozen

**Files:**
- Read/modify if needed: `kernel/seal-os/src/wm/desktop.rs`
- Read/modify if needed: `kernel/seal-os/src/wm/app_state.rs`
- Read/modify if needed: `kernel/seal-os/src/wm/compositor.rs`
- Read/modify if needed: `kernel/seal-os/src/graphics/framebuffer.rs`

- [ ] **Step 1: Inspect current screenshot**

Open:

```text
kernel/seal-os/target/x86_64-unknown-uefi/release/qemu-proof/screen.png
```

Expected: desktop is legible at 1024x768, no broken draw buffer, no blank screen.

- [ ] **Step 2: Make only minimal visual fixes**

Allowed fixes:

```text
text clipping
window overlap
unreadable colors
missing terminal/app proof widget
```

Not allowed in this pass:

```text
large compositor rewrite
new renderer
new UI framework
```

Expected: after any visual edit, re-run QEMU proof and screen checker.

## Task 7: Final Verification And Audit Report

**Files:**
- Read: `git status --short`
- Read: all proof artifacts

- [ ] **Step 1: Run complete local proof pack**

Run:

```powershell
cargo +nightly build --manifest-path kernel\seal-os\Cargo.toml --release
cd kernel\seal-mkimage
cargo +stable run --release
cd ..\..
cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml --release -- --verify kernel\seal-os\target\x86_64-unknown-uefi\release\seal-os.img kernel\seal-os\target\x86_64-unknown-uefi\release\seal-os.efi
cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml --release -- --check-vm-proof kernel\seal-os\target\x86_64-unknown-uefi\release\qemu-proof\serial.log
cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml --release -- --check-proof-screen kernel\seal-os\target\x86_64-unknown-uefi\release\qemu-proof\screen.ppm
cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml --release -- --check-seal-abi .
cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml --release -- --check-language-hygiene .
cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml --release -- --check-o1-allocator .
```

Expected: all pass.

- [ ] **Step 2: Give final audit in proved/blocked/red format**

Use:

```markdown
| Area | Status | Evidence |
| --- | --- | --- |
| QEMU OS boot | proved | ... |
| Persistent root | proved | ... |
| Oracle VM | blocked/proved/red | ... |
| Ubuntu benchmark lead | red until measured | ... |
```

Expected: user can see exactly what works, what is blocked by host tooling, and what still needs engineering.

## Execution Order

1. Freeze baseline.
2. Dispatch Newton/Noether/Hilbert/Curie/Gauss as independent agents.
3. Merge only reviewed edits.
4. Run the complete proof pack.
5. Update README last, after artifacts exist.
6. Give final audit.

## Stop Conditions

Stop immediately and report if:

- `--check-vm-proof` fails.
- QEMU no longer reaches desktop event loop.
- ManifoldFS falls back to ramfs.
- Theorem proof fails.
- O(1) allocator gate fails.
- Any Python runtime dependency is found.
- Oracle failure is ambiguous between host tooling and OS boot.

## Self-Review

- Spec coverage: covers OS boot, topological theorem contract, AHCI/ManifoldFS persistence, Aether-Lang, no Python, Oracle, README truth, benchmark honesty.
- Placeholder scan: no empty work markers or fill-later tasks.
- Type consistency: all commands use existing Rust/PowerShell artifacts and known paths.
