# Seal OS 0.4.5 Guide

Seal OS is a bare-metal Rust and Aether-Lang operating system. It is not a
Linux distribution, not a POSIX personality, and not a Unix clone. The kernel
boots through UEFI, owns its own Seal ABI, and treats OS state as topology:
frames, files, tasks, windows, and runtime control all have theorem-facing
geometry.

This guide is the practical companion to the README. The README is the project
bible; this file is the workbench.

## Current Proof Target

The current measurable target is:

1. Build `seal-os.efi` from Rust.
2. Package a bootable UEFI disk image.
3. Boot the image in a VM from Windows through WSL2 QEMU or Oracle VirtualBox.
4. Capture serial proof that all T1-T10 theorem gates pass.
5. Capture pixel proof that the desktop frame is nonblank and has the primary
   control surface visible.
6. Mount the persistent ManifoldFS partition from the AHCI disk.
7. Reach the Seal desktop and event loop.
8. Run Rust audit gates for theorem log, Seal ABI, language hygiene, O(1)
   allocator hot path, and unsafe inventory.

If any of those fail, the OS is not considered working for this version.

## Language Surface

Seal OS source language surface is:

- Rust for kernel, drivers, memory, scheduler, filesystem, networking, graphics,
  theorem gates, and VM boot code.
- Assembly for low-level page-table/context-switch/hardware paths where Rust
  cannot express the instruction contract cleanly.
- Aether-Lang for native OS scripts, app logic, theorem-facing automation, and
  the future self-hosted application layer.

Legacy host-script files are not part of the OS language surface. They are
either tests/specs, old research code awaiting Rust/Aether conversion, tooling,
or vendor-like subprojects. They must not be deleted blindly; conversion must
keep semantic parity.

## Build

From the repository root:

```powershell
cd kernel\seal-os
cargo +nightly build --release
```

Expected artifact:

```text
kernel\seal-os\target\x86_64-unknown-uefi\release\seal-os.efi
```

Package the VM disk:

```powershell
cd ..\seal-mkimage
cargo +stable run --release
```

Expected artifacts:

```text
kernel\seal-os\target\x86_64-unknown-uefi\release\seal-os.img
kernel\seal-os\target\x86_64-unknown-uefi\release\seal-os-efi.img
```

`seal-os.img` is GPT. The FAT EFI System Partition boots the kernel; the second
partition is `ManifoldFS` and starts at LBA 133120 with an `MNFD` superblock.

## WSL2 QEMU Proof

The Windows runner detects native QEMU first. If native QEMU is unavailable, it
uses WSL Ubuntu QEMU and OVMF.

```powershell
cd kernel\seal-os
powershell -NoProfile -ExecutionPolicy Bypass -File .\run-qemu.ps1 -HeadlessProof -ProofSeconds 240
```

Proof output:

```text
kernel\seal-os\target\x86_64-unknown-uefi\release\qemu-proof\serial.log
kernel\seal-os\target\x86_64-unknown-uefi\release\qemu-proof\screen.ppm
kernel\seal-os\target\x86_64-unknown-uefi\release\qemu-proof\screen.png
kernel\seal-os\target\x86_64-unknown-uefi\release\qemu-proof\proof-manifest.txt
```

The serial log must contain:

```text
[ALLOC] O(1) proof: topo_cells=8, l3_word_probes_per_cell=2, single_word_probes_per_cell=8192, contiguous_candidate_probes=128, contiguous_max_run_pages=64, toporam_max_run_pages=64, marking=bounded_by_contiguous_max_run_pages
[ALLOC] runtime counters: fast_hits=1, bounded_misses=0, max_contiguous_probes_seen=0
[BENCH] toporam-alloc iterations=64 ok=64 p50_cycles=<n> p95_cycles=<n> max_cycles=<n> target_cell_hits_delta=64 target_cell_fallbacks_delta=0 low_to_high_fallbacks_delta=0 high_to_low_fallbacks_delta=0 pcie_to_high_fallbacks_delta=0 pcie_to_low_fallbacks_delta=0 free_before=<n> free_after=<n>
[BENCH] alloc-frame iterations=64 ok=64 p50_cycles=<n> p95_cycles=<n> max_cycles=<n> fast_hits_delta=64 bounded_misses_delta=0 max_contiguous_probes_seen_delta=0 free_before=<n> free_after=<n>
[BENCH] manifold-teleport api=teleport fs_mode=ramfs persistence=write_through samples=3 ok=3 same_inode=3 src_gone=3 dst_present=3 entries_min=8 entries_max=256 payload_bytes=64 p50_cycles=<n> p95_cycles=<n> max_cycles=<n> ticks_max=<n> metadata_ops_max=7 write_through_bytes_per_move=64 payload_points=<n>
[THEOREM] T1/TSS VERIFIED
[THEOREM] T2/SCM VERIFIED
[THEOREM] T3/GMC VERIFIED
[THEOREM] T4/AGCR VERIFIED
[THEOREM] T5/HCS VERIFIED
[THEOREM] T6/RGCS VERIFIED
[THEOREM] T7/PHKP VERIFIED
[THEOREM] T8/TEB VERIFIED
[THEOREM] T9/CMA VERIFIED
[THEOREM] T10/WPHB VERIFIED
[BOOT] All T1-T10 theorems VERIFIED; T1-T5 ACTIVE in runtime paths
[AHCI] Registered as block device 0x800
[disk::ahci] First disk readable (sector 0 OK)
[VFS] ManifoldFS mounted from disk
[BOOT] Desktop proof frame blit done
[GFX] desktop-soak frames=24 p50_cycles=<n> p95_cycles=<n> max_cycles=<n> missed_16ms=unscaled input_events=0 dirty_px_max=786432
[BOOT] Seal OS desktop ready.
[EVENT] Entering real event loop
```

The theorem log audit now requires desktop readiness and event-loop entry, not
just theorem banner lines. The screenshot audit parses `screen.ppm` directly and
requires a nonblank 1024x768 desktop with the icon lane, control region, and
primary terminal titlebar visible.

The manifest audit parses `proof-manifest.txt` and verifies the proof bundle
against the current artifacts. It requires the QEMU backend, commit/dirty flag,
image and EFI fingerprints, serial/screen fingerprints, and every hard gate
status to be recorded as `ok`.

The desktop soak marker is the first anti-jank gate. It proves the boot path ran
a deterministic compositor compose+blit exercise before declaring the desktop
ready. The current marker reports raw cycle percentiles, not calibrated
microseconds, so it is a readiness gate rather than a final 60 FPS proof.

The allocator benchmark markers are the first Seal-side performance artifacts
for the Ubuntu comparison table. `[BENCH] toporam-alloc` proves 64 hint-biased
TopoRAM allocations hit the target Voronoi cell with no target-cell fallback, no
zone fallback, and no frame leak. `[BENCH] alloc-frame` proves 64 physical
single-frame alloc/free iterations completed through the topological fast path
with no bounded misses and no frame leak. Neither marker is a Ubuntu win until
the same-machine Ubuntu baseline is captured.

The ManifoldFS teleport marker is the first file-movement performance artifact.
It proves a same-inode move from source directory to destination directory over
8-256 source entries with bounded metadata operations. It also reports
`persistence=write_through` and `write_through_bytes_per_move`, because current
raw-byte persistence still touches payload bytes; full metadata-only persistence
is not claimed yet.

## Oracle VirtualBox Proof

Convert the image:

```powershell
cd kernel\seal-os
powershell -File .\build-vbox.ps1
```

VM settings:

- Type: Other/Unknown 64-bit
- EFI: enabled
- RAM: 4096 MB recommended
- Video: VMSVGA, 128 MB video memory
- Disk: generated VDI from `seal-os.img`
- Optional NIC: Intel PRO/1000 MT Desktop

Boot success means the GUI reaches the desktop, the serial log reaches the same
T1-T10 theorem proof, and the VM does not hang during driver init.

Current workspace status: Oracle VirtualBox is green for the current 0.4.5
image when the VDI is rebuilt from the current raw image. `smoke-vbox.ps1
-Seconds 240` reached the theorem gate, `VBOX HARDDISK`, block device `0x800`,
readable sector 0, persistent ManifoldFS, current allocator markers, desktop
proof-frame blit, Aether runtime proof marker, desktop readiness, and
event-loop entry. `-SkipBuild` is only valid after `seal-os.vdi` was regenerated
from the current `seal-os.img`.

## Audit Gates

Run these from the repository root unless stated otherwise:

```powershell
cargo +stable test --workspace
cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml --release -- --verify kernel\seal-os\target\x86_64-unknown-uefi\release\seal-os.img kernel\seal-os\target\x86_64-unknown-uefi\release\seal-os.efi
cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml --release -- --check-vm-proof kernel\seal-os\target\x86_64-unknown-uefi\release\qemu-proof\serial.log
cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml --release -- --check-vbox-proof kernel\seal-os\target\x86_64-unknown-uefi\release\vbox-smoke\serial.log
cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml --release -- --check-aether-runtime kernel\seal-os\target\x86_64-unknown-uefi\release\qemu-proof\serial.log
cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml --release -- --check-theorem-log kernel\seal-os\target\x86_64-unknown-uefi\release\qemu-proof\serial.log
cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml --release -- --check-proof-screen kernel\seal-os\target\x86_64-unknown-uefi\release\qemu-proof\screen.ppm
cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml --release -- --check-proof-manifest kernel\seal-os\target\x86_64-unknown-uefi\release\qemu-proof\proof-manifest.txt
cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml --release -- --check-seal-abi .
cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml --release -- --check-language-hygiene .
cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml --release -- --check-aether-migration .
cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml --release -- --check-o1-allocator .
cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml --release -- --check-runtime-theorems .
cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml --release -- --unsafe-inventory .
```

`--check-language-hygiene` is the source-surface gate. It requires
`.gitattributes` to quarantine legacy host scripts, frontend scaffolding,
Dockerfiles, requirements text, configs, assets, docs, and generated proof
output from GitHub language stats. The visible OS language surface is Rust,
assembly, and Aether-family DSL. The gate also rejects host scripts in
production OS/Rust roots and requires `docs/HOST_LANGUAGE_QUARANTINE.md` to
describe why remaining host scripts are not runtime Seal OS code.

`kernel/seal-os` is intentionally outside the root Cargo workspace because it
uses a UEFI custom target. Its proof is the release build plus VM boot proof.
The supported kernel test compilation gate is
`cargo +nightly build --release --features test-mode`. Raw
`cargo +nightly test --lib` inside `kernel/seal-os` is not a supported proof
today because the crate-level Cargo config targets UEFI with `build-std`, which
causes duplicate `core` metadata on a host test build.

## O(1) Allocation Contract

Allowed claim:

- Single-frame allocation is O(1) over installed RAM size through a fixed
  topological summary index. The boot log must print the allocator proof marker.
- Multi-page contiguous allocation uses 128 bounded topological candidate
  probes and a hard `MAX_CONTIGUOUS_RUN_PAGES = 64` cap. Search is O(1) over
  installed RAM size, and per-call marking is bounded by the cap.
- Public TopoRAM allocation and free-side topology repair reject requests above
  the same 64-page cap, so TopoRAM cannot reintroduce unbounded request-size
  loops around the physical allocator.
- TopoRAM entropy checks are interval-gated and sampled with a fixed upper
  bound. Voronoi reseeding refreshes only the fixed recent-allocation ring
  during runtime allocation.
- The bitmap is still the truth store, but mutable access is allocator-private.
  TopoRAM can inspect bounded read-only samples; it cannot mutate around the
  topological free index.
- Larger transfers must use chunked/scatter-gather I/O rather than one
  unbounded contiguous allocation.

Not allowed claim:

- "All allocation work is O(1) for unbounded request sizes." The runtime proof
  covers bounded single-frame and capped contiguous runs; unbounded bulk
  transfer belongs in chunked/scatter-gather I/O.

Next closure:

- Scatter-gather paths for devices that do not require physical contiguity.
- Benchmarks proving allocator latency does not scale with RAM size.
- Per-path counters for candidate probes, failed probes, and request-size
  marking cost.

## Benchmark Direction

Seal OS should be compared to Ubuntu only on the workloads it is designed to
win:

- HFT tick-to-action latency.
- ML tensor locality and data-movement latency.
- ManifoldFS metadata teleport; current persistence still rewrites file bytes.
- Topological allocator hot path.
- Live scheduler selection through `[BENCH] scheduler-select-next`; wake latency
  remains a separate end-to-end benchmark.
- Boot-to-theorem readiness.
- Jitter under scheduler pressure.

Claims must cite logs, screenshots, benchmark output, and exact commands.
Anything else is noise wearing a hat.

## Seal OS 0.4.5 Proof Baseline

Current passed baseline on this workspace:

- Kernel release build: `cargo +nightly build --release`
- Image build: `cargo +stable run --release` in `kernel\seal-mkimage`
- VM proof: `powershell -NoProfile -ExecutionPolicy Bypass -File .\run-qemu.ps1 -HeadlessProof -ProofSeconds 240`
- Full serial proof gate: `seal-mkimage --check-vm-proof`
- Aether runtime gate: `seal-mkimage --check-aether-runtime`
- Serial proof: T1-T10 verified, desktop proof frame blitted, desktop ready,
  event loop active
- Aether proof: boot log contains
  `[Aether-Lang] runtime proof: parser=ok interpreter=ok app_host=ok script=aether_boot_probe result=seal-topology-ok`
- Storage proof: AHCI finds `QEMU HARDDISK`, registers block device `0x800`, reads
  sector 0, and mounts ManifoldFS from disk instead of ramfs
- Oracle proof: AHCI finds `VBOX HARDDISK`, registers block device `0x800`,
  reads sector 0, mounts ManifoldFS from disk, reaches desktop, and passes
  `seal-mkimage --check-vbox-proof` after `smoke-vbox.ps1 -Seconds 240`
- Screenshot proof: `seal-mkimage --check-proof-screen` passes against
  `qemu-proof\screen.ppm`

This proves Seal OS boots, mounts persistent ManifoldFS, and paints the first
desktop frame in the current WSL2/QEMU environment. It does not prove Seal OS
beats Ubuntu on HFT/ML workloads; that needs the benchmark plan below to produce
numbers.

## Current Known Gaps

- `kernel/seal-os` raw host unit tests need a dedicated host target harness;
  use the UEFI test-mode build plus VM proof until that harness exists.
- Unsafe Rust inventory is visible but not yet reduced.
- Legacy host-script theorem/app files remain as conversion backlog, but
  `--check-aether-migration` now proves Rust replacement symbols and rejects
  live imports of deleted theorem modules.
- T1-T5 runtime topology is source-gated by `--check-runtime-theorems`; this
  is a coverage gate, not a formal theorem proof for every runtime behavior.
- Real vendor GPU acceleration is not present; the desktop uses software
  framebuffer rendering.
- Long-running VM soak tests are still needed.
- Scatter-gather DMA closure is still needed to reduce contiguous allocation
  pressure further.
- Long-running Oracle soak beyond the 240 s smoke proof is still pending.
