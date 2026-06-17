# Runtime Proof Max Parallel Agents Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the OS runtime proof chain strong enough that every README claim has a boot log, parser gate, VM artifact, and rejection test.

**Architecture:** Runtime truth flows from kernel serial markers to `seal-mkimage` parsers to VM proof manifests to README status. This plan builds that chain with 50 agents working at maximum safe parallelism: many scouts, few writers per shared file, one final integrator.

**Tech Stack:** Rust `no_std` kernel, `seal-mkimage`, QEMU headless proof, Oracle VirtualBox smoke proof, PowerShell runners, GitHub Actions, Lean/Lake, Aether-Lang.

---

## Global Runtime Parameters

| Parameter | Value |
|---|---|
| Workspace | `C:\Users\seal\Documents\GitHub\Epsilon-Hollow` |
| Runtime target | `x86_64-unknown-uefi` |
| Kernel check | `cargo +nightly check --manifest-path .\kernel\seal-os\Cargo.toml -Z build-std=core,alloc --target x86_64-unknown-uefi` |
| Gate test | `cargo test --manifest-path kernel\seal-mkimage\Cargo.toml` |
| QEMU proof | `.\kernel\seal-os\run-qemu.ps1 -HeadlessProof -ProofSeconds 240` |
| VirtualBox proof | `.\kernel\seal-os\smoke-vbox.ps1` after image exists |
| Final docs checks | `cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml -- --check-doc-claim-contract .` and `--check-language-hygiene .` |
| Agent count | 50 |
| Maximum read-only parallelism | 50 |
| Maximum code parallelism | 10, because `lib.rs`, `main.rs`, and README are shared choke points |
| No-go rule | No agent may make a broad claim from a narrow fixture |

## File Ownership Rules

| Shared File | Lock Owner | Rule |
|---|---|---|
| `kernel/seal-os/src/lib.rs` | Runtime Integrator | Agents submit snippets; integrator places boot calls |
| `kernel/seal-mkimage/src/main.rs` | Gate Integrator | One parser patch at a time |
| `README.md` | Bible Integrator | One final language pass after proof exists |
| `docs/CI.md` | CI Integrator | Marker numbering updated once per wave |
| `kernel/seal-os/run-qemu.ps1` | VM Integrator | QEMU args changed only after hardware fixture proof design |
| `.github/workflows/ci.yml` | CI Integrator | CI devices and timeouts changed only with VM proof |

## 50 Runtime Agents

| Agent | Runtime Domain | Exact Task | Primary Files | Output Marker Or Gate | Verification Command |
|---:|---|---|---|---|---|
| 01 | Boot | Add boot phase timing around memory, drivers, desktop | `kernel/seal-os/src/lib.rs` | `[BOOT-TIME] phase=` | `--check-theorem-log` rejects missing fatal markers |
| 02 | Boot | Prove boot reaches event loop under 240 s | `kernel/seal-os/run-qemu.ps1`, `kernel/seal-mkimage/src/main.rs` | proof manifest field `desktop_ready=ok` | QEMU proof command |
| 03 | Boot | Add panic/fault watchdog hard reject list | `kernel/seal-mkimage/src/main.rs` | `reject_fatal_boot_markers` | mkimage tests |
| 04 | Memory | Strengthen allocation marker duplicate detection | `kernel/seal-mkimage/src/main.rs` | unique `[BENCH] alloc-frame` | mkimage tests |
| 05 | Memory | Add free-frame leak delta checker | `kernel/seal-os/src/lib.rs`, `kernel/seal-mkimage/src/main.rs` | `free_before`, `free_after` invariant | benchmark log check |
| 06 | Memory | Add TopoRAM target-cell sample expansion | `kernel/seal-os/src/memory/topo_ram.rs` | `[BENCH] toporam-alloc samples=256` | kernel check and QEMU serial |
| 07 | Memory | Add slab class stress expansion | `kernel/seal-os/src/lib.rs` | `[BENCH] slab-alloc classes=6` with higher iterations | mkimage tests |
| 08 | Scheduler | Build kernel-task context ping-pong proof | `kernel/seal-os/src/process` | `[BENCH] context-switch mode=kernel_task_pingpong` | QEMU serial and parser |
| 09 | Scheduler | Prove scheduler selector never changes ready count | `kernel/seal-mkimage/src/main.rs` | stricter parser fields | mkimage tests |
| 10 | Scheduler | Add signal frame runtime marker | `kernel/seal-os/src/process/signal.rs` | `[PROCESS] signal-frame-proof` | kernel check |
| 11 | Storage | Add NVMe QEMU device profile | `kernel/seal-os/run-qemu.ps1`, `.github/workflows/ci.yml` | QEMU args include `-device nvme` only in NVMe proof mode | dry-run command output |
| 12 | Storage | Fix NVMe completion timeout correctness | `kernel/seal-os/src/drivers/nvme.rs` | error return on timeout/status fail | kernel check |
| 13 | Storage | Add NVMe read benchmark marker | `kernel/seal-os/src/drivers/nvme.rs`, `kernel/seal-os/src/lib.rs` | `[BENCH] nvme-read` | QEMU NVMe proof |
| 14 | Storage | Add NVMe write scratch restore marker | `kernel/seal-os/src/drivers/nvme.rs` | `[BENCH] nvme-write` | QEMU NVMe proof |
| 15 | Storage | Add AHCI write/read/restore proof | `kernel/seal-os/src/drivers/disk/ahci.rs` | `[BENCH] ahci-write-read` | QEMU AHCI proof |
| 16 | Filesystem | Add ManifoldFS parity boot fixture | `kernel/seal-os/src/fs/manifold.rs` | `[BENCH] manifold-parity` | benchmark log check |
| 17 | Filesystem | Add VFS operation matrix proof | `kernel/seal-os/src/fs/vfs.rs` | `[VFS] op-matrix` | theorem log check |
| 18 | Filesystem | Add journal flush readback proof | `kernel/seal-os/src/fs/journal.rs` | `[FS] journal-proof` | QEMU serial |
| 19 | Filesystem | Add package install file byte proof | `kernel/seal-os/src/pkg` | `[ManifoldPkg] proof metadata_only=0` strict parser | mkimage tests |
| 20 | Filesystem | Add block device registry proof | `kernel/seal-os/src/drivers/block` | `[BLOCK] registry-proof` | theorem log check |
| 21 | Network | Add e1000 descriptor proof | `kernel/seal-os/src/drivers/net/e1000.rs` | `[BENCH] e1000-ring` | QEMU with NIC |
| 22 | Network | Add TCP session matrix parser | `kernel/seal-mkimage/src/main.rs` | `parse_tcp_session_matrix` | mkimage tests |
| 23 | Network | Add UDP checksum proof | `kernel/seal-os/src/net/udp.rs` | `[BENCH] udp-checksum` | benchmark log check |
| 24 | Network | Add DNS parse proof | `kernel/seal-os/src/net/dns.rs` | `[BENCH] dns-query` | benchmark log check |
| 25 | Network | Add TLS bad-tag rejection proof | `kernel/seal-os/src/drivers/net/tls.rs` | `[BENCH] tls-auth-reject` | mkimage tests and QEMU |
| 26 | Graphics | Add 3D render marker | `kernel/seal-os/src/lib.rs` | `[BENCH] topo-render-3d` | benchmark log check |
| 27 | Graphics | Add tensor render marker | `kernel/seal-os/src/lib.rs` | `[BENCH] tensor-render` | benchmark log check |
| 28 | Graphics | Speed up framebuffer clear | `kernel/seal-os/src/graphics/framebuffer.rs` | no new marker; same desktop proof faster | QEMU reaches desktop ready |
| 29 | Graphics | Reduce pre-proof blits from three to one | `kernel/seal-os/src/lib.rs` | same `[GFX] desktop-proof` | QEMU reaches proof |
| 30 | Graphics | Convert full-frame proof to region proof with same signals | `kernel/seal-os/src/lib.rs`, `kernel/seal-mkimage/src/main.rs` | `[GFX] desktop-proof scanned_pixels=<bounded>` | proof parser rejects weak signals |
| 31 | GPU | Add shader binary hash manifest | `kernel/seal-os/src/drivers/gpu` | `[GPU] shader-manifest` | build output and parser |
| 32 | GPU | Add hardware dispatch probe mode | `kernel/seal-os/src/drivers/gpu/topology_accel.rs` | `[GPU-BENCH] hardware_probe` | QEMU says no hardware; real hardware accepts |
| 33 | GPU | Add VRAM/GART memory proof | `kernel/seal-os/src/drivers/gpu` | `[GPU] vram-proof` | hardware serial |
| 34 | GPU | Keep CPU fallback wording strict | `README.md`, `docs/GPU_ACCELERATION.md` | doc claim contract | doc checks |
| 35 | Aether | Prove Aether parser/interpreter/render outputs | `kernel/seal-os/src/lang` | `[Aether-Lang] runtime proof` extended fields | theorem log check |
| 36 | Aether | Add Aether object bytecode build proof | `kernel/seal-mkimage/src/aether_build.rs` | `[Aether-Build] object-proof` | mkimage tests |
| 37 | Aether | Add Aether app host window proof | `kernel/seal-os/src/apps/aether_app_host.rs` | `[Aether-Lang] app-host-proof` | QEMU serial |
| 38 | LAAMBA | Add LAAMBA dataset action proof | `kernel/seal-os/src/apps/laamba_governor.rs` | `[LAAMBA] dataset-proof` | QEMU serial |
| 39 | Security | Add unsafe inventory checker | `kernel/seal-mkimage/src/main.rs` | `--check-unsafe-inventory` | mkimage tests |
| 40 | Security | Add entropy failure-path proof | `kernel/seal-os/src/drivers/entropy.rs` | `[SECURITY] entropy-proof` | kernel check and parser |
| 41 | Security | Add audit append stress proof | `kernel/seal-os/src/security` | `[SECURITY] audit-stress` | QEMU serial |
| 42 | Security | Add auth bad-hash rejection cases | `kernel/seal-os/src/security/auth.rs` | `[SECURITY] auth proof` extended | mkimage tests |
| 43 | Package | Add remote registry fixture | `kernel/seal-os/src/pkg` | `[ManifoldPkg] registry-proof` | QEMU serial |
| 44 | Userland | Add `/bin/init` fixture or narrow claim | `kernel/seal-os/src/fs`, `README.md` | `[USERLAND] init-proof` | QEMU serial |
| 45 | Userland | Add shell command matrix | `kernel/seal-os/src/apps/shell.rs` | `[SHELL] command-matrix` | QEMU serial |
| 46 | Userland | Add file manager action proof | `kernel/seal-os/src/apps/file_manager.rs` | `[FILES] action-proof` | desktop live proof |
| 47 | Userland | Add IDE real completion proof | `kernel/seal-os/src/apps/seal_ide.rs` | `[IDE] completion-proof` | QEMU serial |
| 48 | ML/HFT | Add allocator Ubuntu artifact intake | `tools`, `kernel/seal-mkimage/src/main.rs` | `[UBUNTU-BENCH] alloc-frame` strict | compare gate |
| 49 | Formal | Add Lean theorem weakness scanner | `kernel/seal-mkimage/src/main.rs`, `kernel/aether/aether-verified/lean` | `--check-lean-proof-strength` | `lake build` and mkimage tests |
| 50 | Release | Bind all proof artifacts to current commit | `kernel/seal-mkimage/src/main.rs`, `docs/CI.md` | proof manifest gates | current manifest checks |

## Exact Execution Steps

- [ ] **Step 1: Launch 50 read-only runtime agents**

Prompt:

```text
You own Agent <NN> from docs/superpowers/plans/2026-06-03-runtime-proof-max-parallel-agents.md.
Read only. No host scripts. Inspect the listed files. Return the smallest real runtime proof, exact marker fields, parser rejection cases, and commands. If hardware is required, say the exact VM/device requirement.
```

Expected: 50 reports with no file edits.

- [ ] **Step 2: Select first code wave**

Use this first wave because write scopes do not overlap:

```text
Agents 05, 08, 12, 16, 21, 25, 28, 35, 39, 45
```

Expected: no shared Rust file except agents that submit notes to integrators.

- [ ] **Step 3: Add parser tests before parser implementation**

For each new marker, add one accepted fixture and at least three rejected fixtures:

```rust
assert!(parse_new_marker(GOOD_LOG).is_ok());
assert!(parse_new_marker(&GOOD_LOG.replace("result=pass", "result=fail")).is_err());
assert!(parse_new_marker(&GOOD_LOG.replace("api=real_path", "api=fake_path")).is_err());
assert!(parse_new_marker(&GOOD_LOG.replace("nonblack_px=1", "nonblack_px=0")).is_err());
```

Expected: fake marker cannot pass.

- [ ] **Step 4: Add kernel marker emitter**

Each emitter must print:

```text
api=<exact function path>
fixture=<exact fixture name>
mode=<real mode>
result=pass
p50_cycles=<n> p95_cycles=<n> max_cycles=<n>
```

Expected: marker is tied to a real function call.

- [ ] **Step 5: Run local proof commands**

Run:

```powershell
cargo +stable fmt --all
cargo test --manifest-path kernel\seal-mkimage\Cargo.toml
cargo +nightly check --manifest-path .\kernel\seal-os\Cargo.toml -Z build-std=core,alloc --target x86_64-unknown-uefi
git diff --check
```

Expected: all exit 0.

- [ ] **Step 6: Run VM proof if boot path changed**

Run:

```powershell
.\kernel\seal-os\run-qemu.ps1 -HeadlessProof -ProofSeconds 240
cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml -- --check-benchmark-log <serial.log>
```

Expected: marker appears in serial and parser accepts the serial artifact.

- [ ] **Step 7: Update docs only after evidence**

Modify:

```text
README.md
docs/CI.md
docs/VM_RUNBOOK.md
docs/SEAL_OS_GUIDE.md
docs/BENCHMARK_PLAN.md
```

Expected: docs say exactly what proof shows and name what remains outside scope.

- [ ] **Step 8: Push verified wave**

Run:

```powershell
git add <files>
git commit -m "feat(runtime): gate <marker-name>"
git push origin feat/seal-os-complete-2026-06-03
```

Expected: one proof wave lands cleanly.

## Runtime Completion Bar

The README has zero pain only when:

- every performance row has a serial marker or honest narrowed language;
- every marker has a `seal-mkimage` parser and rejection tests;
- QEMU proof reaches desktop ready or the README states the exact boot blocker;
- VirtualBox proof has a current manifest when Oracle VM support is claimed;
- GPU hardware claims remain scoped until `hardware_dispatch=1` is proven;
- Ubuntu-beating claims remain off until same-machine Ubuntu artifacts exist.

## Self-Review

- Spec coverage: boot, memory, scheduler, storage, filesystem, network, graphics, GPU, Aether, security, package, userland, ML/HFT, formal proof, and release binding all have owners.
- No fake runtime: agents may narrow claims, but cannot emit pass markers without real function calls.
- Type consistency: every marker name maps to a parser or manifest gate.
