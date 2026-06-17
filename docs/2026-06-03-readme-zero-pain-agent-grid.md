# README Zero Pain Agent Grid Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove every README pain point by assigning one exact proof-building agent per cliff, with no fake claims and no poison stubs.

**Architecture:** The README is the claim ledger. Every agent must move one row from pain to proof by adding kernel code, `seal-mkimage` gates, docs, tests, and VM evidence, or by narrowing the README claim so it matches current proof. Agents work in parallel only when write scopes do not overlap.

**Tech Stack:** Bare-metal Rust `no_std`, `seal-mkimage` Rust proof gates, QEMU/VirtualBox proof logs, Aether-Lang runtime, Lean theorem docs, PowerShell VM runners, Git.

---

## Global Agent Parameters

Use these exact parameters for every agent in this document.

| Parameter | Value |
|---|---|
| Workspace | `C:\Users\seal\Documents\GitHub\Epsilon-Hollow` |
| Branch prefix | `codex/` for isolated worktrees |
| Persona | `caveman-genius`: short words, exact code names, hard truth |
| Process skills | `superpowers:subagent-driven-development`, `superpowers:systematic-debugging`, `superpowers:verification-before-completion` |
| Rust skills | `rust-router`, `m10-performance`, `unsafe-checker` when touching `unsafe`, `domain-embedded` for kernel/no_std |
| Forbidden | Host-script shortcuts, fake pass markers, mock-only hardware claims, broad rewrites, deleting existing work to hide pain |
| Required proof style | Serial marker + `seal-mkimage` parser + rejection tests + README/doc row update |
| Required commands | `cargo +stable fmt --all`; `cargo test --manifest-path kernel\seal-mkimage\Cargo.toml`; `cargo +nightly check --manifest-path .\kernel\seal-os\Cargo.toml -Z build-std=core,alloc --target x86_64-unknown-uefi`; `cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml -- --check-doc-claim-contract .`; `cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml -- --check-language-hygiene .`; `git diff --check` |
| VM proof command | `.\kernel\seal-os\run-qemu.ps1 -HeadlessProof -ProofSeconds 240` when boot path changed |
| Commit rule | One commit per agent if its proof passes; commit message `feat(<area>): gate <claim>` or `docs(<area>): narrow <claim>` |
| Maximum parallelism | 50 agents may run at once for read-only audit; implementation runs in waves with disjoint files |

## Parallel Wave Map

| Wave | Parallel agents | Rule |
|---|---:|---|
| Wave A: audit only | 50 | All agents read only and produce a proof plan for one README row |
| Wave B: disjoint code | 12 | Only one agent per subsystem writes kernel code |
| Wave C: gates | 8 | Only one agent edits `kernel/seal-mkimage/src/main.rs` at a time; others prepare patches in notes |
| Wave D: docs | 20 | Docs agents may edit separate docs; one README editor integrates final language |
| Wave E: VM proof | 4 | One QEMU runner, one VirtualBox runner, one CI watcher, one log parser |

## Master 50-Agent Matrix

| Agent | Lane | README Pain Point | Write Scope | Marker/Gate | Required Evidence |
|---:|---|---|---|---|---|
| 01 | Memory | Physical frame allocation still has bitmap/contiguous scan paths | `kernel/seal-os/src/memory/phys.rs`, `kernel/seal-mkimage/src/main.rs`, `README.md` | `[BENCH] alloc-frame` extended with `bitmap_scan_delta=0` | QEMU serial proves 64 alloc/free cycles with no bitmap fallback |
| 02 | Memory | Multi-page contiguous O(1) claim has capped run, not all sizes | `kernel/seal-os/src/memory/topo_ram.rs`, `README.md`, `docs/BENCHMARK_PLAN.md` | `[BENCH] toporam-contiguous` | Fixture proves 2, 4, 8, 16, 32, 64 page bounded candidate probes |
| 03 | Memory | DMA allocation proof lacks device-class fixture | `kernel/seal-os/src/memory/phys.rs`, `kernel/seal-os/src/drivers/block/mod.rs` | `[BENCH] dma-frame` | AHCI/NVMe-capable DMA frame request returns aligned frame and bounded probes |
| 04 | Memory | COW proof needs syscall-path stress | `kernel/seal-os/src/memory`, `kernel/seal-os/src/process` | `[MM] cow-stress` | Fork/clone failure injection shows rollback and no parent fallback |
| 05 | Memory | KPTI proof lacks syscall hot-loop stress | `kernel/seal-os/src/security/kpti.rs`, `kernel/seal-os/src/syscall` | `[SECURITY] kpti-syscall-stress` | 1024 syscall entries preserve user/kernel CR3 invariants |
| 06 | Scheduler | Context switch row says benchmark pending | `kernel/seal-os/src/process/context_switch.rs`, `kernel/seal-os/src/process/scheduler.rs` | `[BENCH] context-switch` | Two kernel tasks ping-pong cookies; no hang; switch counter advances |
| 07 | Scheduler | Wake latency not measured | `kernel/seal-os/src/process/scheduler.rs`, `kernel/seal-os/src/drivers/interrupts.rs` | `[BENCH] wake-latency` | Timer interrupt wakes sleeping task; p50/p95/max monotonic |
| 08 | Scheduler | SMP scheduling proof thin | `kernel/seal-os/src/cpu/smp.rs`, `kernel/seal-os/src/process/scheduler.rs` | `[BENCH] smp-scheduler` | Per-CPU run queues preserve task count and no duplicate running task |
| 09 | Scheduler | Signal delivery needs user-stack proof | `kernel/seal-os/src/process/signal.rs`, `kernel/seal-os/src/process/userspace.rs` | `[PROCESS] signal-proof` | Pending/mask/handler frame behavior exercised for listed signals |
| 10 | Scheduler | `sigaltstack` claim needs proof | `kernel/seal-os/src/process/signal.rs`, `kernel/seal-os/src/syscall` | `[PROCESS] sigaltstack-proof` | Alternate stack frame address range verified |
| 11 | Storage | NVMe read row pending | `kernel/seal-os/src/drivers/nvme.rs`, `kernel/seal-os/run-qemu.ps1` | `[BENCH] nvme-read` | QEMU NVMe device reads known sector and completion status is checked |
| 12 | Storage | NVMe write row pending | `kernel/seal-os/src/drivers/nvme.rs`, `kernel/seal-os/run-qemu.ps1` | `[BENCH] nvme-write` | Scratch LBA write/read/restore proof with timeout and status checks |
| 13 | Storage | NVMe timeout path unsafe | `kernel/seal-os/src/drivers/nvme.rs` | Parser test plus unit-style host test if possible | `read_sector` and `write_sector` fail if completion never arrives |
| 14 | Storage | AHCI write persistence proof thin | `kernel/seal-os/src/drivers/disk/ahci.rs`, `kernel/seal-os/src/fs` | `[BENCH] ahci-write-read` | Dedicated scratch sector roundtrip then restore |
| 15 | Storage | VFS root depends on AHCI only | `kernel/seal-os/src/fs/vfs.rs`, `kernel/seal-os/src/drivers/block` | `[VFS] block-device-proof` | Device registry shows AHCI and NVMe registration paths without fake mount |
| 16 | Filesystem | ManifoldFS full parity fixture missing | `kernel/seal-os/src/fs/manifold.rs`, `kernel/seal-mkimage/src/main.rs` | `[BENCH] manifold-parity` | create/write/read/rename/delete/readdir/stat match expected data |
| 17 | Filesystem | FAT/ext2 mounted fixture parity missing | `kernel/seal-os/src/fs/fat.rs`, `kernel/seal-os/src/fs/ext2.rs` | `[FS] fixture-parity` | Mounted image fixture proves read/write/create/mkdir/unlink/rmdir/rename/stat/readdir |
| 18 | Filesystem | Content-addressable lookup still bucket-size | `kernel/seal-os/src/fs/voronoi_cap.rs` | `[BENCH] cap-lookup` | Hard bucket cap and collision fixture prove bounded probes |
| 19 | Filesystem | VRAM file teleport design not implemented | `kernel/seal-os/src/fs`, `kernel/seal-os/src/drivers/gpu` | `[BENCH] vram-teleport` | Only if VRAM/GART proof exists; else README narrowed |
| 20 | Filesystem | Lypnos AEAD/KDF gate pending | `kernel/seal-os/src/security`, `kernel/seal-os/src/fs` | `[SECURITY] lypnos-crypto` | Lock/unlock uses KDF + AEAD and rejects tampered ciphertext |
| 21 | Network | External TCP session matrix missing | `kernel/seal-os/src/net/tcp.rs`, `kernel/seal-os/src/drivers/net` | `[BENCH] tcp-session-matrix` | SYN, data, FIN, RST, retransmit fixtures |
| 22 | Network | UDP/DNS proof thin | `kernel/seal-os/src/net/udp.rs`, `kernel/seal-os/src/net/dns.rs` | `[BENCH] udp-dns` | Query/response parse fixture with checksum proof |
| 23 | Network | DHCP external lease not guaranteed | `kernel/seal-os/src/net/dhcp.rs` | `[NET] dhcp-proof` | QEMU/slirp lease or honest no-lease marker with scope |
| 24 | Network | TLS X.509/ECDHE missing | `kernel/seal-os/src/drivers/net/tls.rs` | `[BENCH] tls-handshake` | Real ECDHE transcript fixture or README keeps PSK-only scope |
| 25 | Network | e1000 RX/TX ring proof missing | `kernel/seal-os/src/drivers/net/e1000.rs` | `[BENCH] e1000-ring` | Descriptor ownership changes and packet loopback with checksum |
| 26 | Graphics | 3D render row pending | `kernel/seal-os/src/graphics/topo_render.rs`, `kernel/seal-os/src/lib.rs` | `[BENCH] topo-render-3d` | 1024-triangle software raster produces nonblack output and hash |
| 27 | Graphics | Tensor render row pending | `kernel/seal-os/src/ml_engine/tensor_viz.rs`, `kernel/seal-os/src/lib.rs` | `[BENCH] tensor-render` | 100x100 CSV to mesh to nonblank offscreen render |
| 28 | Graphics | Desktop proof scan too slow | `kernel/seal-os/src/lib.rs`, `kernel/seal-os/src/graphics/framebuffer.rs` | `[GFX] desktop-proof` faster same fields | 240 s QEMU reaches desktop ready with same proof fields |
| 29 | Graphics | Frame pacing not calibrated | `kernel/seal-os/src/lib.rs`, `kernel/seal-os/src/wm/compositor.rs` | `[GFX] frame-pacing` | 60-frame compose/blit cycles against 16.7 ms target |
| 30 | Graphics | Proof screen uses PPM only | `kernel/seal-mkimage/src/main.rs`, `kernel/seal-os/run-qemu.ps1` | `--check-proof-screen` | PPM contains desktop signal after faster proof path |
| 31 | GPU | Hardware dispatch missing | `kernel/seal-os/src/drivers/gpu` | `[GPU-BENCH] hardware` | `hardware_dispatch=1`, shader checksum, CPU recompute match |
| 32 | GPU | Shader binaries absent | `kernel/seal-os/src/drivers/gpu/shaders` | build script input check | Real `.bin` blobs checked in with hash manifest |
| 33 | GPU | VRAM residency proof missing | `kernel/seal-os/src/drivers/gpu`, `kernel/seal-os/src/memory` | `[GPU] vram-proof` | BAR/GART map, write/read pattern, no CPU fallback claim |
| 34 | GPU | CPU fallback claim needs limit text | `README.md`, `docs/GPU_ACCELERATION.md` | doc contract | README says CPU fallback correctness only until hardware proof exists |
| 35 | Aether | `.aeo` payload builder uses scaffold payload | `kernel/seal-mkimage/src/aether_build.rs`, `kernel/aether` | `[Aether-Build] object-proof` | Real bytecode serialization from Aether compiler |
| 36 | Aether | Aether in kernel lacks app suite proof | `kernel/seal-os/src/lang`, `kernel/seal-os/src/apps/aether_app_host.rs` | `[Aether-Lang] app-suite` | Multiple scripts parse, interpret, render, and return exact output |
| 37 | Aether | Aether standard library OS binding proof thin | `kernel/seal-os/src/lang/stdlib.rs` | `[Aether-Lang] stdlib-proof` | File/window/tensor calls hit real OS functions |
| 38 | Aether | LAAMBA Governor data path scoped | `kernel/seal-os/src/apps/laamba_governor.rs` | `[LAAMBA] dataset-proof` | Dataset selection changes rendered state and model config |
| 39 | Security | Unsafe Rust inventory not closed | `kernel/seal-os/src`, `docs/THREAT_MODEL.md` | `--check-unsafe-inventory` | Every `unsafe` has reason, invariant, and owner tag |
| 40 | Security | KASLR missing | `kernel/seal-os/src/boot`, `kernel/seal-os/src/memory` | `[SECURITY] kaslr-proof` | Randomized kernel virtual base or README states no KASLR |
| 41 | Security | Crypto random proof incomplete | `kernel/seal-os/src/security`, `kernel/seal-os/src/drivers/entropy.rs` | `[SECURITY] entropy-proof` | RDSEED/RDRAND failure path returns error and success path sampled |
| 42 | Package | Public registry fixture missing | `kernel/seal-os/src/pkg`, `docs/MANIFOLDPKG.md` | `[ManifoldPkg] registry-proof` | Signed remote-like registry fixture install/list/remove |
| 43 | Package | Package signature label says fixture | `kernel/seal-os/src/pkg`, `README.md` | `[ManifoldPkg] signature-proof` | Ed25519 key fixture verification with bad signature rejection |
| 44 | Userland | `/bin/init` missing | `kernel/seal-os/src/process`, `kernel/seal-os/src/fs` | `[USERLAND] init-proof` | `/bin/init` exists or README states kernel desktop fallback |
| 45 | Userland | Shell commands lack proof matrix | `kernel/seal-os/src/apps/shell.rs` | `[SHELL] command-matrix` | help, fs, tensor, topcrypt, ml commands run deterministic outputs |
| 46 | Userland | IDE completion contract source-only | `kernel/seal-os/src/apps/seal_ide.rs` | `[IDE] completion-proof` | Completion request returns real candidates for Aether/Rust snippets |
| 47 | ML | ML training benchmark absent | `kernel/seal-os/src/ml_engine.rs` | `[BENCH] ml-train` | Tiny model trains one epoch with decreasing loss fixture |
| 48 | ML | HFT benchmark comparison incomplete | `tools`, `docs/BENCHMARK_PLAN.md` | `[UBUNTU-BENCH] hft` plus compare gate | Same-machine Ubuntu artifact and Seal artifact stored in proof manifest |
| 49 | Formal | Lean theorem strength has weak lemmas | `kernel/aether/aether-verified/lean`, `docs/THEOREMS.md` | `lake build` plus theorem scan | No theorem reduced to `True` for T1/T3/T7/T10 named gaps |
| 50 | README | Final pain-zero bible update | `README.md`, `docs/CI.md`, `docs/SEAL_OS_GUIDE.md` | doc claim contract | No `benchmark pending` row remains unless paired with exact next proof gate and honest △ |

## Execution Checklist

- [ ] **Step 1: Freeze current evidence**

Run:

```powershell
git status --short --branch
rg -n "benchmark pending|raw Ubuntu artifact pending|out of scope|stub|placeholder|△|✗" README.md docs kernel -g "*.md" -g "*.rs"
```

Expected: a current pain list with file paths and no hidden dirty work ignored.

- [ ] **Step 2: Start Wave A with 50 read-only scouts**

For each agent in the matrix, send this exact prompt:

```text
You own Agent <NN> from docs/superpowers/plans/2026-06-03-readme-zero-pain-agent-grid.md.
Read only. No host scripts. Do not edit files.
Return: current evidence, exact files to modify, proof marker fields, commands to verify, and whether the README claim must be narrowed instead of implemented.
```

Expected: 50 scout reports, one per pain point.

- [ ] **Step 3: Choose implementation wave**

Select agents whose write scopes do not overlap. Maximum first implementation wave: Agents 01, 06, 11, 16, 21, 26, 31, 35, 39, 42, 44, 49.

Expected: no two selected agents edit the same Rust module or the same doc file.

- [ ] **Step 4: Enforce one-marker rule**

Before any agent writes, reserve marker names in a tracking note:

```text
Agent NN marker=[BENCH] name files=<paths> checker=parse_<name>_benchmark
```

Expected: no duplicate serial markers.

- [ ] **Step 5: Gate every proof**

Every implementation agent must add:

```text
kernel serial marker
seal-mkimage parser
positive parser test
negative parser tests for wrong API, weak fixture, failed result
README claim row update
CI/runbook marker list update
```

Expected: `cargo test --manifest-path kernel\seal-mkimage\Cargo.toml` rejects fake proof lines.

- [ ] **Step 6: Run hard verification**

Run:

```powershell
cargo +stable fmt --all
cargo test --manifest-path kernel\seal-mkimage\Cargo.toml
cargo +nightly check --manifest-path .\kernel\seal-os\Cargo.toml -Z build-std=core,alloc --target x86_64-unknown-uefi
cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml -- --check-doc-claim-contract .
cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml -- --check-language-hygiene .
git diff --check
```

Expected: all exit 0.

- [ ] **Step 7: VM proof for boot changes**

Run:

```powershell
.\kernel\seal-os\run-qemu.ps1 -HeadlessProof -ProofSeconds 240
```

Expected: the new marker appears and `seal-mkimage --check-benchmark-log <serial.log>` accepts the log. If desktop proof times out, record the exact last serial line and do not claim desktop-ready success.

- [ ] **Step 8: Commit only verified slices**

Run:

```powershell
git add <changed files>
git commit -m "feat(<area>): gate <claim>"
git push origin feat/seal-os-complete-2026-06-03
```

Expected: pushed commit contains one proof slice, not a chaos bundle.

## Self-Review

- Spec coverage: all README pain rows and known cliffs have an agent owner.
- No poison stubs: this plan forbids fake pass markers and mock-only hardware claims.
- Type consistency: marker names map to parser names and README rows.
- Remaining risk: full objective is not complete until every agent row has proof evidence or an honest narrowed claim.
