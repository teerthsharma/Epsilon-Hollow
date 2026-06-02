# Seal OS Cliff Eradication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn every known Seal OS cliff into a named proof gate, then remove the cliffs one by one with VM evidence, Rust tests, and docs that cannot overclaim.

**Architecture:** Each cliff gets one owner gate in `kernel/seal-mkimage`, one boot/runtime marker when the proof must come from the OS, and one doc row that states the exact exit condition. A cliff is not "gone" because the README sounds confident; it is gone when a command fails on regression and passes on current proof artifacts.

**Tech Stack:** Bare-metal Rust `no_std`, `seal-mkimage` Rust audit gates, QEMU/OVMF, VirtualBox smoke scripts, Aether-Lang runtime markers, Lean 4 theorem package, GitHub Actions, no Python runtime path.

---

## Non-Negotiable Rules

- No Python in the OS/runtime proof path.
- No "better than Ubuntu" claim without same-machine Ubuntu artifact and `--check-current-benchmark-proof`.
- No GPU-native claim without hardware dispatch proof artifact.
- No "any VM" claim until every named VM/backend has its own proof bundle and current-manifest provenance.
- No README victory row without a `seal-mkimage` gate or named external artifact.
- Every new cliff doc must include the command that kills it.

## Cliff Ledger

| Cliff | Current evidence | Missing proof | Exit gate that vanishes it |
|---|---|---|---|
| Weak desktop serial proof | `[GFX] desktop-proof` proves desktop pixels, `[GFX] desktop-live-proof` proves icon-click routing/focus/blit, `[GFX] desktop-soak` proves compose timing, and local PPM proof checks pixels | Calibrated frame pacing and broad input coverage still need future artifacts | `seal-mkimage --check-desktop-soak serial.log` must require `[GFX] desktop-proof`, `[GFX] desktop-live-proof`, and `[GFX] desktop-soak` together |
| Oracle/VirtualBox parity | `smoke-vbox.ps1` captures serial/screenshot and current manifest proof passes | VirtualBox PNG is visual evidence, not QEMU PPM parity | `seal-mkimage --check-proof-manifest vbox-smoke/proof-manifest.txt` plus screenshot artifact and current dirty/commit provenance |
| Ubuntu superiority | Seal allocator and TopoRAM markers exist | Same-machine Ubuntu 26.04 artifact missing for full HFT/ML matrix | `seal-mkimage --check-current-benchmark-proof qemu-proof/proof-manifest.txt ubuntu-alloc.log .` and future HFT/ML benchmark gates in `docs/BENCHMARK_PLAN.md` |
| GPU-native topology | PM4/firmware scaffold and CPU fallback `[GPU-BENCH]` exist | `hardware_dispatch=1` proof missing; shader stubs do not compute topology | New hardware `[GPU-BENCH]` gate must prove dispatch, wait, upload, download, CPU recompute match, and real shader checksum |
| O(1) allocator global claim | Single-frame and TopoRAM target-cell markers exist | Multi-page and all VM memory maps still need stronger artifact coverage | `--check-o1-allocator`, `[ALLOC] O(1) proof`, `[BENCH] alloc-frame`, `[BENCH] toporam-alloc`, and a contiguous-DMA benchmark row with bounded candidate probes |
| ManifoldFS full parity | Same-inode mock block teleport marker exists | Full FAT/ext2 parity and persistence fixtures are pending | Fixture gate must prove write/read/delete/rename persistence and keep `persistence_bytes_per_move=0` scoped to `fs_mode=mock_block` teleport proof |
| Aether-Lang native OS language | `--check-aether-runtime` proves parser/interpreter/app host boot probe | Aether self-hosting and kernel build scripts are not Aether-native yet | Runtime gate remains; future self-host gate must run an Aether script that builds or transforms a checked kernel artifact |
| LAAMBA Governor native bridge | Rust/native bridge gate exists and `[LAAMBA] app proof:` is now a boot-log contract | Fresh QEMU and VirtualBox proof artifacts must be regenerated after the marker lands | `seal-mkimage --check-laamba-app-proof serial.log` must prove LAAMBA window/app lane launches through the Aether/native manifest path with `python_runtime=0` |
| Security | ASLR/seccomp/MAC scaffolding exists | KPTI wiring, AEAD/KDF TopCrypt, X.509/ECDHE are missing | Dedicated gates: KPTI boot selftest marker, TopCrypt AEAD/KDF fixture, TLS X.509/ECDHE fixture |
| Release honesty | Release workflow now requires explicit tag and proof gates | Release must continue rejecting stale `latest` fallbacks and missing proof assets | `seal-mkimage --check-release-workflow-contract .` in CI |
| Topological semantic closure | `[BOOT] All T1-T10 theorems VERIFIED; T1-T5 ACTIVE in runtime paths` and `--check-runtime-theorems` source-gate callsites | Runtime markers prove topology is wired into core paths, not full semantic equivalence of every subsystem to every theorem claim | Add subsystem-specific theorem consequence markers after allocator, scheduler, ManifoldFS, and compositor paths |
| Any VM / real hardware | QEMU and VirtualBox have separate proof gates | No VMware, Hyper-V, cloud hypervisor, or physical-machine proof contract | Add one backend-specific proof contract per target; never generalize QEMU success |
| COW/page-table safety | README admits COW fork double-free risk | No regression fixture for fork/COW page-table lifecycle under memory pressure | Add a Rust kernel test or mkimage source gate for COW clone/free ownership invariants before claiming security maturity |

## Task 1: Close The Weak Desktop Serial Proof Cliff

**Files:**
- Modify: `kernel/seal-mkimage/src/main.rs`
- Modify: `kernel/seal-os/src/lib.rs`
- Modify: `README.md`
- Modify: `docs/CI.md`
- Modify: `docs/VM_RUNBOOK.md`
- Modify: `docs/SEAL_OS_GUIDE.md`

- [x] **Step 1: Write failing mkimage tests**

Add tests named:

```rust
desktop_soak_requires_serial_graphics_desktop_proof
desktop_proof_rejects_blank_or_weak_framebuffer_claims
```

Expected red run:

```powershell
cargo test --manifest-path .\kernel\seal-mkimage\Cargo.toml --release desktop_proof
```

Expected failure before implementation: `parse_graphics_desktop_proof` missing, or old soak-only log accepted.

- [x] **Step 2: Add parser gate**

`check_desktop_soak_text` must call `parse_graphics_desktop_proof(text)` and `parse_graphics_desktop_live_proof(text)` before accepting `[GFX] desktop-soak`.

Required fields:

```text
[GFX] desktop-proof version=1 surface=framebuffer width=1024 height=768 bpp=32 pitch=<n> back_buffer=1 window_count=12 focused_window_id=<n> scanned_pixels=786432 nonblack_px=<n> visible_icons=10 icon_region_signal=<n> icon_color_buckets=<n> control_region_signal=<n> primary_titlebar_signal=<n> start_button_signal=<n> theorem_indicator_signal=<n> minimized_app_lane_signal=<n> power_button_signal=<n> sampled_pixels=<n> nonblack_samples=<n> sample_hash=<n> result=pass
[GFX] desktop-live-proof version=1 route=desktop_handle_input action=desktop_icon_launch app=Files app_id=3 events=2 handled=1 icon_hit=1 launched_app_id=3 pre_focused=<n> post_focused=<n> post_window_id=<n> window_count=12 pre_hash=<n> post_hash=<n> changed_samples=<n> blit=1 result=pass
```

Minimums enforced by parser:

```text
width >= 1024
height >= 768
bpp = 32
back_buffer = 1
window_count >= 12
scanned_pixels >= width * height
nonblack_px >= 20000
visible_icons >= 10
icon_region_signal >= 2000
icon_color_buckets >= 6
control_region_signal >= 8000
primary_titlebar_signal >= 6000
start_button_signal >= 1000
theorem_indicator_signal >= 500
minimized_app_lane_signal >= 300
power_button_signal >= 220
sampled_pixels >= 1024
nonblack_samples >= 512
sample_hash != 0
```

Live input proof minimums enforced by parser:

```text
route = desktop_handle_input
action = desktop_icon_launch
app = Files
app_id = 3
events >= 2
handled = 1
icon_hit = 1
launched_app_id = 3
window_count >= 12
changed_samples >= 32
blit = 1
pre_focused != post_focused
post_focused = post_window_id
pre_hash != 0
post_hash != 0
pre_hash != post_hash
```

- [x] **Step 3: Emit live kernel marker**

`kernel/seal-os/src/lib.rs` must measure pixels from `Framebuffer::get_pixel` after desktop render and compositor compose, before `[BOOT] Desktop proof frame blit done`; then route a Files icon click through `wm::desktop::handle_input`, blit, and emit `[GFX] desktop-live-proof`.

- [x] **Step 4: Update docs**

Docs must show `[GFX] desktop-proof` and `[GFX] desktop-live-proof` in the boot proof path and explain that `[GFX] desktop-soak` now includes serial pixel proof plus live input proof.

Commands:

```powershell
cargo +stable run --manifest-path .\kernel\seal-mkimage\Cargo.toml --release -- --check-doc-claim-contract .
cargo +stable run --manifest-path .\kernel\seal-mkimage\Cargo.toml --release -- --check-language-hygiene .
```

- [x] **Step 5: Prove in VM**

Run:

```powershell
cd kernel\seal-os
powershell -NoProfile -ExecutionPolicy Bypass -File .\run-qemu.ps1 -HeadlessProof -ProofSeconds 240
```

Pass condition:

```text
qemu-proof\serial.log contains [GFX] desktop-proof ... result=pass
qemu-proof\serial.log contains [GFX] desktop-live-proof ... result=pass
qemu-proof\proof-manifest.txt has proof_verdict=PASS
seal-mkimage --check-current-proof-manifest qemu-proof\proof-manifest.txt . passes
```

## Task 2: Plan And Gate Oracle/VirtualBox Parity

**Files:**
- Modify: `kernel/seal-os/smoke-vbox.ps1`
- Modify: `kernel/seal-mkimage/src/main.rs`
- Modify: `docs/VM_RUNBOOK.md`
- Modify: `docs/CI.md`

- [x] **Step 1: Add or strengthen a VirtualBox manifest test**

The test must reject a VirtualBox manifest that claims QEMU PPM parity and accept a manifest with serial log, screenshot PNG, image, EFI, `virtualbox_backend=headless`, and current proof gates.

Run:

```powershell
cargo test --manifest-path .\kernel\seal-mkimage\Cargo.toml --release proof_manifest_accepts_vbox_bundle_without_ppm_claim proof_manifest_rejects_vbox_ppm_parity_claim
```

- [x] **Step 2: Run VirtualBox smoke**

Run:

```powershell
cd kernel\seal-os
powershell -NoProfile -ExecutionPolicy Bypass -File .\smoke-vbox.ps1 -Seconds 240
```

Pass condition:

```text
vbox-smoke\serial.log passes --check-vm-proof
vbox-smoke\serial.log passes --check-desktop-soak
vbox-smoke\proof-manifest.txt passes --check-proof-manifest
```

## Task 2B: Gate LAAMBA As A Kernel Boot App

**Files:**
- Modify: `kernel/seal-mkimage/src/main.rs`
- Modify: `kernel/seal-os/src/lib.rs`
- Modify: `kernel/seal-os/smoke-vbox.ps1`
- Modify: `.github/workflows/ci.yml`
- Modify: `.github/workflows/release.yml`
- Modify: `README.md`
- Modify: `docs/CI.md`
- Modify: `docs/VM_RUNBOOK.md`
- Modify: `docs/SEAL_OS_GUIDE.md`
- Modify: `docs/BENCHMARK_PLAN.md`

- [x] **Step 1: Add parser and red gate**

`seal-mkimage --check-laamba-app-proof serial.log` rejects missing windows,
Python-backed runtime claims, wrong launcher ids, and missing Aether host window
ids.

- [x] **Step 2: Emit kernel boot marker**

```text
[LAAMBA] app proof: version=1 native_app=kernel window=LAAMBA_Governor window_id=<n> launcher_id=10 desktop_icon=1 start_menu=1 aether_host_window_id=<n> runtime_bridge=rust_native_manifest python_runtime=0 result=pass
```

- [x] **Step 3: Wire CI, release, and VirtualBox waits**

QEMU CI, release preflight, and `smoke-vbox.ps1` all wait for the marker instead
of trusting a host app compile alone.

- [x] **Step 4: Regenerate fresh QEMU and VirtualBox proof artifacts**

Both canonical proof directories must contain the new marker and pass
`--check-current-proof-manifest`.

Pass condition:

```text
qemu-proof\serial.log passes --check-laamba-app-proof
qemu-proof\proof-manifest.txt passes --check-current-proof-manifest
vbox-smoke\serial.log passes --check-laamba-app-proof
vbox-smoke\proof-manifest.txt passes --check-current-proof-manifest
```

## Task 3: Gate Full Ubuntu/HFT/ML Superiority Without Lying

**Files:**
- Modify: `docs/BENCHMARK_PLAN.md`
- Modify: `kernel/seal-mkimage/src/main.rs`
- Modify: `tools/ubuntu-alloc-bench/src/main.rs`

- [ ] **Step 1: Preserve current allocator-only truth**

Run:

```powershell
cargo +stable run --manifest-path .\kernel\seal-mkimage\Cargo.toml --release -- --check-doc-claim-contract .
```

Pass condition: README keeps `raw Ubuntu artifact pending` until artifact exists.

- [ ] **Step 2: Add next HFT/ML benchmark parser test before implementation**

Add one new benchmark family at a time. First target should be a deterministic packet/data-move fixture, not a giant benchmark suite.

Run:

```powershell
cargo test --manifest-path .\kernel\seal-mkimage\Cargo.toml --release benchmark_proof_requires_current_vm_manifest_and_native_ubuntu_artifact
```

Pass condition: same-machine native Ubuntu artifact remains mandatory.

## Task 4: Gate GPU Hardware Dispatch Separately From CPU Fallback

**Files:**
- Modify: `kernel/seal-os/src/apps/gpu_benchmark.rs`
- Modify: `kernel/seal-mkimage/src/main.rs`
- Modify: `docs/GPU_ACCELERATION.md`
- Modify: `README.md`

- [ ] **Step 1: Keep CPU fallback honest**

Run:

```powershell
cargo test --manifest-path .\kernel\seal-mkimage\Cargo.toml --release benchmark_log_rejects_fake_gpu_hardware_marker
```

Pass condition: fake `hardware_dispatch=1` markers are rejected.

- [ ] **Step 2: Add hardware-only parser behind a separate marker**

The future marker must be named `[GPU-HW-BENCH]`, not overloaded onto `[GPU-BENCH]`.

Required fields:

```text
[GPU-HW-BENCH] suite version=1 backend=<vendor> hardware_dispatch=1 shader_used=1 kernels=3 result=pass
```

Pass condition: CPU fallback proof remains valid without claiming hardware GPU.

## Self-Review

- Spec coverage: every known cliff has a current marker, missing proof, and exit gate.
- Placeholder scan: there are no "TBD" or "implement later" placeholders; pending work has concrete commands and pass conditions.
- Type consistency: marker names match current `seal-mkimage` gates: `[LAAMBA] app proof:`, `[GFX] desktop-proof`, `[GFX] desktop-live-proof`, `[GFX] desktop-soak`, `[GPU-BENCH]`, `--check-current-benchmark-proof`, `--check-proof-manifest`, `--check-doc-claim-contract`.
