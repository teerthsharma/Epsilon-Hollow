# Seal OS Audit-First Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Drive Seal OS toward a measurable bare-metal topological OS benchmark with bootable VM proof, T1-T10 theorem-gated integrity, Seal ABI/no-POSIX discipline, Aether-Lang as the native OS language, and benchmark evidence for the HFT/ML target workloads.

**Architecture:** Repair the repo from audit evidence before running expensive VM or CI tests. First make the source tree truthful and hygienic, then add mechanical gates that prevent drift, then verify with lightweight local checks, then run image and VM smoke, then compare against Ubuntu only through a written benchmark harness and published results.

**Tech Stack:** Bare-metal Rust `no_std`, UEFI, x86_64, Aether-Lang, Lean proof artifacts, GitHub Actions, PowerShell/Bash local scripts, QEMU, Oracle VM VirtualBox, Markdown docs.

---

## Non-Negotiable Truth Contract

Seal OS is not Linux, not Unix, not a POSIX ABI, and not a libc target. It is a bare-metal Rust kernel with a native Seal ABI and Aether-Lang as the OS language layer. Familiar syscall names may exist only as Seal-defined semantics. If a doc says POSIX compatibility, Unix contract, libc inheritance, or Linux inheritance, it must be marked historical or removed.

Seal OS can aim to surpass Ubuntu for selected HFT/ML workloads, but the repo must not claim broad superiority until the benchmark plan has reproducible runs against Ubuntu 24.04 or a newer selected Ubuntu baseline on the same host, same VM/hardware class, same CPU/RAM/disk constraints, and named workloads.

The README is the bible, but the code is the law. If README and code differ, either the code changes to match the README or the README changes to state the current truth and the plan to close the gap.

## Audit Source Map

Use these findings as the repair queue:

| Lane | Evidence | Required closure |
| --- | --- | --- |
| Aether hardware safety | `kernel/seal-os/src/lang/mod.rs`, `kernel/aether/Aether-Lang/crates/aether-lang/src/interpreter.rs` expose `mmio_read32` and `mmio_write32` | Capability-gated MMIO ranges; no arbitrary address path from untrusted Aether scripts |
| User-copy safety | `kernel/seal-os/src/security/smap_smep.rs`, syscall callers in `pipe.rs` and `time.rs` | Validate mapped user spans, writable spans for output, cross-page checks, and page fault recovery plan |
| Theorem gate proof strength | `kernel/seal-os/src/lib.rs`, `kernel/aether/aether-verified/lean` | Distinguish Rust numeric gates, Lean proof status, and runtime use; remove "complete proof" claims where placeholders remain |
| CI theorem parsing | `.github/workflows/ci.yml` | Require every `[THEOREM] Tn/... VERIFIED` line and fail on any `FAILED`, not just the summary slogan |
| Seal ABI/no-POSIX | `README.md`, `docs/SYSCALLS.md`, `kernel/seal-os/src/syscall` | Add scanner and docs contract: Seal ABI names are native, not POSIX compatibility |
| Unsafe policy | `kernel/seal-os/src/**/*.rs`, `.github/workflows/ci.yml` | Inventory unsafe sites, require safety comments for new or touched unsafe, avoid broad lint suppression drift |
| VM proof | `kernel/seal-os/smoke-vbox.ps1`, `docs/BOOT.md`, `kernel/seal-os/README.md` | Reliable QEMU and VirtualBox runbook, serial sentinel proof, image verifier, screenshot/log artifacts |
| Docs drift | `README.md`, `docs/CI.md`, `docs/THEOREMS.md`, `docs/research/*.tex` | Honest wording for storage, theorems, boot milestones, benchmarks, LAAMBA app boundary |
| Hygiene | `.gitignore`, `test_extern.pdb`, image/log artifacts, generated schemas | Generated artifact policy and ignore rules that match the actual build outputs |

## File Map

### Hygiene and generated artifact policy

- Modify: `.gitignore`
- Create: `docs/GENERATED_FILES.md`
- Inspect: `apps/laamba-governor/.gitignore` if present
- Remove from working tree if untracked junk: `test_extern.pdb`
- Keep if source-like and intentional: `kernel/seal-os/build-vbox.ps1`, `kernel/seal-os/smoke-vbox.ps1`, `kernel/seal-os/run-qemu.ps1`, `scripts/local_ci.sh`, `scripts/push_with_ci.sh`

### README and OS guide

- Modify: `README.md`
- Modify: `kernel/seal-os/README.md`
- Modify: `docs/BOOT.md`
- Modify: `docs/CI.md`
- Modify: `docs/THEOREMS.md`
- Modify: `docs/SYSCALLS.md`
- Create: `docs/VM_RUNBOOK.md`
- Create: `docs/BENCHMARK_PLAN.md`

### Safety and gates

- Modify: `.github/workflows/ci.yml`
- Create: `scripts/check_seal_abi.py`
- Create: `scripts/check_theorem_smoke.py`
- Create: `scripts/check_unsafe_inventory.py`
- Modify later with tests: `kernel/seal-os/src/lang/mod.rs`
- Modify later with tests: `kernel/aether/Aether-Lang/crates/aether-lang/src/interpreter.rs`
- Modify later with tests: `kernel/seal-os/src/security/smap_smep.rs`

### VM boot proof

- Modify: `kernel/seal-os/smoke-vbox.ps1`
- Modify: `kernel/seal-os/build-vbox.ps1`
- Inspect: `kernel/seal-mkimage/src/main.rs`
- Inspect: `kernel/seal-os/src/boot/uefi_entry.rs`
- Inspect: `kernel/seal-os/src/lib.rs`
- Inspect: `kernel/seal-os/src/wm/desktop.rs`
- Inspect: `kernel/seal-os/src/wm/compositor.rs`

## Task 1: Capture Baseline and Protect User Work

**Files:**
- Read: all files listed in File Map
- Modify: none

- [ ] **Step 1: Record dirty tree**

Run:

```powershell
git status --short
```

Expected: output may be large. Do not revert unrelated changes. Treat all existing dirty files as user or prior-agent work unless this plan explicitly touches them.

- [ ] **Step 2: Record audit search surface**

Run:

```powershell
rg -n "complete proofs|No POSIX|POSIX|Aether-Lang|LAAMBA|Lambda|VirtualBox|QEMU|Ubuntu|benchmark|All T1-T10|every byte on disk" README.md docs kernel\seal-os\README.md .github\workflows\ci.yml
```

Expected: lines showing the docs drift. Use this output to patch the smallest true sections.

- [ ] **Step 3: Confirm VM script state**

Run:

```powershell
Get-Content -Path kernel\seal-os\smoke-vbox.ps1 -TotalCount 260
```

Expected: script has success sentinel `[BOOT] Seal OS desktop ready.` and failure sentinel patterns. If the script still treats VBoxManage warnings as fatal, patch it before running VirtualBox.

## Task 2: Hygiene and Generated Artifact Policy

**Files:**
- Modify: `.gitignore`
- Create: `docs/GENERATED_FILES.md`
- Remove if present and untracked: `test_extern.pdb`

- [ ] **Step 1: Expand `.gitignore` for OS build outputs**

Add rules for:

```gitignore
# --- OS / VM / firmware build artifacts ---
*.img
*.iso
*.vdi
*.vmdk
*.qcow2
*.efi
*.fd
*.ovmf
*.pdb
*.map
*.sym
*.dSYM/
kernel/seal-os/target/
kernel/seal-os/target/**/vbox-smoke/
kernel/seal-os/target/**/serial.log
kernel/seal-os/target/**/screenshot.png
```

Do not ignore Rust source, docs, scripts, lockfiles, or generated schemas until the policy states whether they are source artifacts or reproducible generated artifacts.

- [ ] **Step 2: Create `docs/GENERATED_FILES.md`**

Content must define:

```markdown
# Generated Files Policy

Seal OS keeps source, specs, lockfiles, and small reproducible metadata in git.
Seal OS ignores VM images, disk images, firmware blobs, debug symbols, logs,
screenshots, benchmark raw output, and build directories.

## Checked In

- Rust source and Cargo manifests.
- `Cargo.lock` files for reproducible application and kernel builds.
- Scripts used by local CI, QEMU, VirtualBox, and image verification.
- Small generated API schemas only when they are required to build without
  host-side regeneration.

## Ignored

- `target/`, `node_modules/`, `.next/`, `.tauri/`, caches.
- `*.img`, `*.iso`, `*.vdi`, `*.vmdk`, `*.qcow2`, `*.efi`, `*.pdb`.
- Serial logs, screenshots, local benchmark raw outputs, temporary reports.

## Regeneration Commands

- Seal OS EFI/image: `powershell -NoProfile -ExecutionPolicy Bypass -File kernel\seal-os\build-vbox.ps1`
- Image verifier: `cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml --release -- --verify <img> <efi>`
- QEMU smoke: use `docs/VM_RUNBOOK.md`.
- VirtualBox smoke: use `kernel\seal-os\smoke-vbox.ps1` from `kernel\seal-os`.
```

- [ ] **Step 3: Remove stale local debug artifact**

Run:

```powershell
if (Test-Path -LiteralPath test_extern.pdb) { Remove-Item -LiteralPath test_extern.pdb -Force }
```

Expected: file is gone from `git status --short`. This command is safe because the file is an untracked local debug artifact named in the audit.

- [ ] **Step 4: Verify ignore behavior**

Run:

```powershell
git status --short
git check-ignore -v test_extern.pdb kernel\seal-os\target\x86_64-unknown-uefi\release\seal-os.img kernel\seal-os\target\x86_64-unknown-uefi\release\seal-os.vdi
```

Expected: ignored paths match `.gitignore`; source files and docs still appear normally.

## Task 3: README Bible Truth Repair

**Files:**
- Modify: `README.md`
- Modify: `kernel/seal-os/README.md`
- Modify: `docs/SYSCALLS.md`

- [ ] **Step 1: Replace the opening contract**

The README opening must state:

```markdown
Seal OS is a bare-metal x86_64 operating system for ML and HFT research.
It is written in Rust, boots through UEFI, uses a native Seal ABI, and treats
OS decisions as topology problems. Aether-Lang is the native OS language layer:
the Seal equivalent of HolyC in spirit, but implemented as a real lexer, parser,
AST, interpreter, VM, and kernel bridge.

Seal OS is not Linux, not Unix, not POSIX, and not a libc target. Familiar
syscall names are Seal ABI entry points with Seal semantics.
```

Do not claim every persistent byte on disk is already a point cloud unless current code proves it on-disk. Current truthful wording:

```markdown
ManifoldFS stores raw bytes for faithful reads and writes, plus topological
ManifoldPayload embeddings for search, Voronoi indexing, teleport metadata,
and future payload-first disk layout work.
```

- [ ] **Step 2: Add explicit Aether-Lang and LAAMBA boundary**

Add a section:

```markdown
## OS Language and Bundled Apps

Aether-Lang is the native Seal OS language layer. Kernel Rust owns hardware,
memory safety, scheduling, drivers, and theorem gates. Aether-Lang owns native
scripts, app logic, shell automation, topology commands, and future self-hosting
flows.

LAAMBA/Lambda Governor is a bundled native app workload and control surface.
It is important, but it is not the kernel identity and not the OS architecture.
```

- [ ] **Step 3: Replace no-POSIX wording everywhere**

Use this wording in README and syscall docs:

```markdown
No POSIX ABI or Unix compatibility contract. Seal ABI may use familiar names
such as `read`, `write`, or `execve`, but their contract is Seal-defined and
can expose topology and theorem state directly.
```

- [ ] **Step 4: Remove broad Ubuntu victory claims**

Replace broad comparison claims with:

```markdown
Seal OS is being built to beat Ubuntu on selected HFT/ML workloads: boot-to-
strategy latency, theorem-gated scheduling, topology-aware file moves,
predictive prefetch, and tensor-locality experiments. Broad superiority is
not claimed until `docs/BENCHMARK_PLAN.md` has reproducible results.
```

## Task 4: VM Runbook and VM Script Truth

**Files:**
- Create: `docs/VM_RUNBOOK.md`
- Modify: `docs/BOOT.md`
- Modify: `kernel/seal-os/README.md`
- Modify: `kernel/seal-os/smoke-vbox.ps1`

- [ ] **Step 1: Create `docs/VM_RUNBOOK.md`**

The runbook must include:

```markdown
# Seal OS VM Runbook

## Expected Boot Sentinels

- `Seal OS - UEFI entry reached, serial online`
- `[BOOT] GOP mode set`
- `[BOOT] Heap initialized`
- `[BOOT] IDT + PIC initialized`
- `[BOOT] SYSCALL/SYSRET MSRs programmed`
- `[T4/AGCR] Governor online`
- `[T1/TSS]  Voronoi index: 8 cells`
- `[THEOREM] T1/TSS VERIFIED`
- `[THEOREM] T2/SCM VERIFIED`
- `[THEOREM] T3/GMC VERIFIED`
- `[THEOREM] T4/AGCR VERIFIED`
- `[THEOREM] T5/HCS VERIFIED`
- `[THEOREM] T6/RGCS VERIFIED`
- `[THEOREM] T7/PHKP VERIFIED`
- `[THEOREM] T8/TEB VERIFIED`
- `[THEOREM] T9/CMA VERIFIED`
- `[THEOREM] T10/WPHB VERIFIED`
- `[BOOT] All T1-T10 theorems VERIFIED; T1-T5 ACTIVE in runtime paths`
- `[BOOT] Seal OS desktop ready.`

## Windows Build

From repo root:

```powershell
cargo +nightly build --manifest-path kernel\seal-os\Cargo.toml --release
cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml --release
```

## VirtualBox Smoke

From `kernel\seal-os`:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\smoke-vbox.ps1 -Seconds 240
```

PASS requires the desktop-ready sentinel. FAIL includes panic, fault, watchdog,
guru meditation, VM stop before success, or timeout.
```

- [ ] **Step 2: Fix wrong cwd in `kernel/seal-os/README.md`**

Commands that run `.\\build-vbox.ps1` and `.\\smoke-vbox.ps1` must explicitly say:

```powershell
cd kernel\seal-os
powershell -NoProfile -ExecutionPolicy Bypass -File .\build-vbox.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\smoke-vbox.ps1 -Seconds 240
```

- [ ] **Step 3: Harden `smoke-vbox.ps1`**

Required properties:

- It accepts `-SuccessPattern`, default `\[BOOT\] Seal OS desktop ready\.`
- It accepts `-FailurePattern`, including panic, `[FAULT]`, `[WATCHDOG]`, and `gurumeditation`
- It never mutates the source `seal-os.vdi`; it copies to a smoke directory
- It uses `--audio-driver none` for VirtualBox 7.x
- It does not treat non-fatal VBoxManage warning text as command failure when exit code is zero
- It removes the temporary VM on exit

- [ ] **Step 4: Do not run VM yet**

VM smoke is heavy and host-sensitive. Run only after Tasks 2-7 are patched and lightweight checks pass.

## Task 5: CI Theorem and Seal ABI Gates

**Files:**
- Modify: `.github/workflows/ci.yml`
- Create: `scripts/check_theorem_smoke.py`
- Create: `scripts/check_seal_abi.py`
- Modify: `docs/CI.md`

- [ ] **Step 1: Create `scripts/check_theorem_smoke.py`**

The script must:

- Read a serial log path from `argv[1]`
- Fail if any line contains `[THEOREM]` and `FAILED`
- Require each theorem tag:
  - `[THEOREM] T1/TSS VERIFIED`
  - `[THEOREM] T2/SCM VERIFIED`
  - `[THEOREM] T3/GMC VERIFIED`
  - `[THEOREM] T4/AGCR VERIFIED`
  - `[THEOREM] T5/HCS VERIFIED`
  - `[THEOREM] T6/RGCS VERIFIED`
  - `[THEOREM] T7/PHKP VERIFIED`
  - `[THEOREM] T8/TEB VERIFIED`
  - `[THEOREM] T9/CMA VERIFIED`
  - `[THEOREM] T10/WPHB VERIFIED`
- Require the summary line
- Print missing patterns before exiting nonzero

- [ ] **Step 2: Create `scripts/check_seal_abi.py`**

The script must scan `kernel/seal-os/src` and fail on:

- `extern crate libc`
- `use libc`
- `std::os::unix`
- `std::os::linux`
- `posix_spawn`
- `pthread_`
- `forkpty`
- `termios`

The script must not fail on docs comments that explicitly say "not POSIX" or "historical".

- [ ] **Step 3: Wire scripts into CI**

Add CI steps after QEMU serial capture:

```yaml
- name: Verify theorem gate lines
  run: python3 scripts/check_theorem_smoke.py qemu.log

- name: Verify Seal ABI/no-POSIX discipline
  run: python3 scripts/check_seal_abi.py
```

Keep the existing hard boot milestone checks, but make theorem verification use the script.

- [ ] **Step 4: Update `docs/CI.md`**

The docs must split:

- Hard gates: UEFI entry, heap, IDT/PIC, syscall MSRs, all theorem lines, image verifier
- Soft milestones: Shell, Desktop, app load, optional hardware drivers

## Task 6: Theorem Status Truth and Proof Roadmap

**Files:**
- Modify: `docs/THEOREMS.md`
- Modify: `docs/research/epsilon_hollow.tex`
- Modify: `docs/research/epsilon_hollow_hft_ml.tex`
- Inspect: `kernel/aether/aether-verified/lean`
- Inspect: `kernel/seal-os/src/lib.rs`

- [ ] **Step 1: Normalize theorem status**

Use three status columns everywhere:

| Status type | Meaning |
| --- | --- |
| Runtime active | Kernel path uses theorem-backed logic during boot/event/runtime |
| Boot-gated | Kernel validates numeric/checkable invariants at boot |
| Formal proof | Lean proof exists without placeholder `True` lemmas |

- [ ] **Step 2: Replace "complete proofs" claims**

Use:

```markdown
T1-T10 have Rust boot gates and theorem-specific runtime checks. Lean proof
coverage is in progress; placeholder lemmas remain documented until replaced
by full formal proofs.
```

- [ ] **Step 3: Add proof closure criteria**

Formal proof is closed only when:

- `rg -n "sorry|admit|True" kernel/aether/aether-verified/lean` has no theorem-placeholder hits
- Rust boot gate inputs are tied to proof/certificate hashes or named proof artifacts
- Property tests cover theorem domains, not just fixed constants

## Task 7: Aether Hardware Capability and User-Copy Safety Plan

**Files:**
- Modify later: `kernel/seal-os/src/lang/mod.rs`
- Modify later: `kernel/aether/Aether-Lang/crates/aether-lang/src/interpreter.rs`
- Modify later: `kernel/seal-os/src/security/smap_smep.rs`
- Create: `docs/AETHER_HARDWARE_CAPS.md`
- Create: `docs/USER_COPY_SAFETY.md`

- [ ] **Step 1: Document Aether hardware capability model**

`docs/AETHER_HARDWARE_CAPS.md` must say:

```markdown
# Aether Hardware Capability Model

Aether-Lang may be the native OS language, but untrusted Aether scripts do not
get arbitrary physical memory or MMIO access. Hardware access requires a Seal
capability object minted by kernel Rust after PCI/ACPI discovery.

Allowed operations are range-checked, width-checked, and tied to a driver-owned
BAR or kernel-approved device window. Raw integer addresses from script space
are rejected.
```

- [ ] **Step 2: Document user-copy safety**

`docs/USER_COPY_SAFETY.md` must state:

```markdown
# User Copy Safety

Seal ABI copies between userspace and kernel only after validating that the
entire byte span is below `USER_SPACE_LIMIT`, page-mapped, user-accessible, and
writable when the kernel writes to it. Cross-page spans are validated page by
page. Page-fault recovery is required before untrusted user pointers become a
stable public ABI surface.
```

- [ ] **Step 3: Patch code only with tests**

Do not patch `smap_smep.rs` or Aether MMIO by guess. First add tests or host-testable validators:

- Aether MMIO rejects raw `0xffff_ffff` without capability
- Aether MMIO accepts a whitelisted fake BAR in test mode
- User-copy validator rejects null, kernel pointer, unmapped pointer, read-only span, and cross-page overflow

## Task 8: Unsafe Inventory Policy

**Files:**
- Create: `scripts/check_unsafe_inventory.py`
- Create: `docs/UNSAFE_INVENTORY.md`
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1: Create inventory doc**

`docs/UNSAFE_INVENTORY.md` must group unsafe by subsystem:

- boot and UEFI handoff
- framebuffers and MMIO
- page tables and CR3
- interrupts and APIC
- syscall entry/exit
- user-copy SMAP/SMEP
- process context switch
- drivers

Each touched unsafe site needs a `SAFETY:` comment describing caller requirements or invariant.

- [ ] **Step 2: Create checker**

`scripts/check_unsafe_inventory.py` must count unsafe blocks/functions in `kernel/seal-os/src` and fail if a new touched unsafe lacks a nearby `SAFETY:` comment. The first version may be baseline-oriented: print counts and fail only on files changed in the current diff.

Run:

```powershell
python scripts\check_unsafe_inventory.py
```

Expected: output includes counts by file and a nonzero exit only when changed unsafe sites lack comments.

## Task 9: Benchmark Plan Before Ubuntu Claims

**Files:**
- Create: `docs/BENCHMARK_PLAN.md`
- Modify: `README.md`
- Modify: `docs/design/PERFORMANCE.md`

- [ ] **Step 1: Create benchmark plan**

Benchmark plan must define:

- Host machine identity: CPU, RAM, disk, firmware, hypervisor, date
- Guest configs: Seal OS image, Ubuntu baseline version, CPU count, RAM, disk format, storage controller
- Workloads:
  - boot to theorem gate
  - boot to desktop-ready sentinel
  - Seal ABI syscall latency
  - scheduler wake latency
  - ManifoldFS teleport vs byte copy
  - block read/write latency
  - network stack ping/TCP when NIC exists
  - Aether-Lang script startup and topological command latency
  - tensor-locality microbench for ML path
  - HFT event loop synthetic tick-to-action latency
- Required output:
  - raw logs
  - summary table
  - exact commands
  - failure budget
  - comparison caveats

- [ ] **Step 2: Define surpass gate**

README may say:

```markdown
Seal OS aims to surpass Ubuntu on the benchmark set in `docs/BENCHMARK_PLAN.md`.
The claim becomes true only for workloads where the measured Seal OS result is
better than the Ubuntu result under the same constraints.
```

## Task 10: Lightweight Verification Before Heavy Tests

**Files:**
- All changed files

- [ ] **Step 1: Formatting and static scans**

Run:

```powershell
cargo +nightly fmt --manifest-path kernel\seal-os\Cargo.toml
cargo +nightly check --manifest-path kernel\seal-os\Cargo.toml
cargo +stable check --manifest-path kernel\seal-mkimage\Cargo.toml
python scripts\check_theorem_smoke.py kernel\seal-os\target\x86_64-unknown-uefi\release\vbox-smoke\serial.log
python scripts\check_seal_abi.py
git diff --check
```

Expected:

- Rust formatting exits zero
- Rust checks exit zero
- theorem smoke script may fail if the last serial log is stale or incomplete; that is useful evidence
- Seal ABI scanner exits zero
- `git diff --check` exits zero

- [ ] **Step 2: Image verification**

Run:

```powershell
cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml --release -- --verify kernel\seal-os\target\x86_64-unknown-uefi\release\seal-os.img kernel\seal-os\target\x86_64-unknown-uefi\release\seal-os.efi
```

Expected: `[mkimage] VERIFY OK`.

## Task 11: Heavy VM and CI Proof Last

**Files:**
- `kernel/seal-os/smoke-vbox.ps1`
- `.github/workflows/ci.yml`
- CI logs

- [ ] **Step 1: QEMU smoke**

Run CI or local equivalent after lightweight checks pass. Required evidence:

- QEMU exit is timeout 124 or long-running success state
- no panic/fault/watchdog
- all ten theorem lines are present
- desktop-ready sentinel is present if GUI path is being tested

- [ ] **Step 2: VirtualBox smoke**

From `kernel\seal-os`:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\smoke-vbox.ps1 -Seconds 240
```

Expected:

- PASS verdict
- serial log path printed
- screenshot path printed
- last serial lines include `[BOOT] Seal OS desktop ready.`

- [ ] **Step 3: CI monitoring**

Only after local proof is sane, monitor GitHub Actions. Do not treat local stale logs as CI proof. CI proof must come from fresh workflow logs for the current commit.

## Task 12: Final Audit Gate

**Files:**
- All changed docs/scripts/code

- [ ] **Step 1: Requirement-by-requirement audit**

Create a final audit table with these rows:

- bootable VM proof
- T1-T10 theorem-gated runtime integrity
- Seal ABI/no-POSIX discipline
- Aether-Lang native OS language role
- LAAMBA/Lambda Governor as app boundary
- generated-file hygiene
- unsafe/user-copy/MMIO safety status
- Ubuntu benchmark plan and measured results status
- README/docs truth alignment

Each row must have:

- evidence file or command
- current status: proved, incomplete, contradicted, or missing
- next required action

- [ ] **Step 2: No false completion**

Do not mark the goal complete unless every row is proved. If Ubuntu surpass evidence is not measured yet, the goal remains active with benchmark plan and implementation progress recorded.

## Self-Review

- Spec coverage: This plan covers VM proof, theorem gates, Seal ABI/no-POSIX, Aether as native OS language, LAAMBA app boundary, hygiene, README/docs, safety, and Ubuntu benchmark evidence.
- Placeholder scan: No `TBD`, `TODO`, or "fill later" steps are used. Unknowns are stated as audit gaps with closure evidence.
- Type/path consistency: Script names and file paths match current repo layout from the audit search.
- Scope check: This is a mountain. The plan splits it into independent lanes so agents can work on docs, hygiene, CI scripts, and safety patches without stepping on the same files.

