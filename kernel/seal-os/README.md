# Seal OS v0.4.7.5

**The Geometrical Operating System** - where OS state is geometry on S^2 and same-filesystem moves have an O(1) metadata-surgery core. The boot benchmark proves that core through a persistent mock block-store path with `persistence_bytes_per_move=0`.

Seal OS is a bare-metal x86_64 operating system built from scratch. Runtime kernel subsystems are source-gated against T1-T5, captured VM proofs pass the T1-T10 theorem gate through the `aether_verified` no_std crate, and Aether-Lang is the native OS language layer above the Rust kernel.

## Quick Start

### Run in Oracle VM VirtualBox

```powershell
cd kernel\seal-os
cargo +nightly build --release

cd ..\seal-mkimage
cargo +stable run --release
```

Return to `kernel\seal-os`, then convert the raw image to VDI:

```powershell
cd ..\seal-os
powershell -NoProfile -ExecutionPolicy Bypass -File .\build-vbox.ps1
```

Automated smoke proof:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\smoke-vbox.ps1 -Seconds 240
```

Or manually:

```powershell
VBoxManage convertfromraw --format VDI target\x86_64-unknown-uefi\release\seal-os.img seal-os.vdi
```

Attach `seal-os.vdi` as a SATA hard disk in Oracle VM VirtualBox.

Recommended VM settings:

- Type: `Other`, Version: `Other/Unknown (64-bit)`
- System: enable EFI, 4096 MB RAM, 1-2 CPUs
- Display: VMSVGA, 128 MB video memory
- Storage: SATA controller, `seal-os.vdi` attached as hard disk
- Network: Intel PRO/1000 MT Desktop if networking is needed

### Build from source

```powershell
# Prerequisites: Rust nightly + rust-src
cd kernel/seal-os
cargo +nightly build --release

# Create bootable UEFI disk image
cd ..\seal-mkimage
cargo +stable run --release
```

### Run in QEMU

```bash
cd kernel/seal-os
./run-qemu.sh
```

```powershell
cd kernel\seal-os
powershell -File .\run-qemu.ps1
```

### Optional ISO

```bash
# From repo root. Requires xorriso, grub-mkrescue, or mkisofs/genisoimage.
./scripts/build_iso.sh
```

## Architecture

Allocator note for 0.4.7.5: the current kernel uses slab/page allocation backed by
the bounded topological physical allocator. Single-frame allocation is O(1) over
installed RAM size, and contiguous DMA allocation is capped at 64 pages per
request; larger transfers must use chunked/scatter-gather I/O. The boot proof
also emits `[BENCH] toporam-alloc` to prove hint-biased TopoRAM target-cell
allocation has no target-cell or zone fallback in the measured hot path.

```
Layer 10  │ Terminal, Seal IDE, File Manager, Theorem Viewer
Layer 9   │ Desktop: wallpaper (Schwarzschild + Faraday), taskbar
Layer 8   │ Window Manager: compositor, decorations, cursor
Layer 7   │ SealShell: look, move (metadata teleport), search, seal, race
Layer 6   │ Seal ABI + Epsilon extensions (manifold_query, teleport)
Layer 5   │ ManifoldScheduler: T1 Voronoi task groups, T4 adaptive timeslice
Layer 4   │ ManifoldFS: raw bytes + S^2 ManifoldPayload embeddings
Layer 3   │ Framebuffer: 1024x768x32, 8x16 font, boot splash
Layer 2   │ Interrupts: IDT, PIC, timer (IRQ0), keyboard, mouse
Layer 1   │ Memory: 16MB heap, bump allocator
Layer 0   │ Boot: UEFI PE/COFF, GOP framebuffer, GPT/FAT ESP image
          └────── T1-T10 BOOT GATE; T1-T5 RUNTIME THEOREMS ──────
```

## Theorem Integration

| Subsystem | T1 (Voronoi) | T2 (SCM) | T3 (Entropy) | T4 (Governor) | T5 (Hyperbolic) |
|-----------|:---:|:---:|:---:|:---:|:---:|
| Memory | frame locality | — | — | heap growth | — |
| Scheduler | task groups | predict next | — | timeslice adapt | — |
| Filesystem | Voronoi bucket lookup | file prefetch | dir reorganize | cache pressure | path resolution |
| Window Mgr | hit-test | — | — | FPS adaptation | — |

### T1: Topological State Space (TSS)
`SphericalVoronoiIndex<8>` partitions all state spaces — memory frames, process groups, filesystem cells, screen regions — into Voronoi cells on S² for O(1) lookup.

### T2: Spectral Contraction Map (SCM)
`SpectralContractionOperator<D>` predicts the next access pattern via Banach contraction in D-dimensional state space. Used for file prefetch and scheduler prediction.

### T3: Geometric Measure Convergence (GMC)
Betti-0 (connected components) of point clouds monitors entropy. When filesystem cluster entropy exceeds threshold, auto-merges smallest clusters.

### T4: Adaptive Geometric Convergence Rate (AGCR)
`GeometricGovernor` uses PD control on epsilon to adapt timeslices, compositor FPS, and I/O rates. Low deviation = conservative; high deviation = aggressive.

### T5: Hyperbolic Curvature Scaling (HCS)
Directory hierarchy uses hyperbolic geometry metrics. Deep paths have logarithmic cost scaling vs linear in Euclidean filesystems.

## ManifoldFS

The filesystem where files keep raw bytes for faithful reads/writes and also carry `ManifoldPayload`s: 64-point clouds on the unit sphere S^2 for topology-aware search, placement, and moves.

```
Write: data -> trigram hash -> JL project -> L2 normalize -> 64 pts on S^2 (1,536 bytes)
Move:  metadata topology surgery -> same inode; mock block-store persistence -> no raw file-byte rewrite
Find:  encode query -> Voronoi cell -> O(bucket size) content-addressable search
```

### Shell Commands

```
seal:/$ help
look [path]   — list directory (shows Voronoi cells, payload points)
create <name> — create directory
write <name> <content> — create file (encodes to S² geometry)
move <src> <dst>       — metadata teleport; same-filesystem mock block-store path reports persistence_bytes_per_move=0
search <query> — content-addressable search via T1 Voronoi
open <path>   — change directory
seal          — show T1-T10 theorem status
race <size>   — benchmark: traditional copy vs manifold teleport
stats         — filesystem statistics
ps            — list processes
uname         — system info
```

### Teleportation Benchmark Target

```
seal:/$ race 2000000000
Race: 2GB transfer
========================
Traditional copy: <measured>
Manifold teleport: <measured> metadata core, raw file bytes stay in place
Speedup: pending raw artifact

Payload: 64 pts * 3 coords * 8 bytes = 1,536 bytes
Current boot gate: [BENCH] manifold-teleport ... fs_mode=mock_block metadata_ops_max=7 persistence_bytes_per_move=0
```

## Built-in Applications

### Terminal Emulator
Full terminal with scrolling, cursor, command history. Default shell with ManifoldFS integration.

### Seal IDE
Built-in development environment with:
- Rust syntax highlighting (keywords, strings, comments, function names)
- File explorer sidebar (ManifoldFS-native)
- Line numbers and gutter
- Status bar: file, line/col, language, theorem state
- Catppuccin Mocha color scheme

### File Manager
Visual ManifoldFS browser:
- Directory listing with Voronoi cell color indicators
- S² stereographic projection view (files as colored points)
- Latitude/longitude grid with 8 Voronoi cell regions

### Theorem Viewer
Real-time dashboard:
- T1-T10 theorem status with progress bars
- Governor epsilon, entropy, Betti numbers
- Prefetch accuracy, teleport count, scheduler ticks

## Desktop

- **Wallpaper**: Procedural black hole visualization with accretion disk rings, plus grid pattern
- **Taskbar**: Bottom bar with T1-T10 status indicators, epsilon readout, Seal OS branding
- **Windows**: Compositing WM with title bars, close buttons, dragging, z-ordering

## Build Details

| Property | Value |
|----------|-------|
| Target | `x86_64-unknown-uefi` |
| Kernel binary | `target/x86_64-unknown-uefi/release/seal-os.efi` |
| VM disk image | `target/x86_64-unknown-uefi/release/seal-os.img` |
| EFI boot image | `target/x86_64-unknown-uefi/release/seal-os-efi.img` |
| Heap | Reserved heap with slab classes, page-backed large allocations, and topological frame backing |
| Display | UEFI GOP framebuffer, 1024x768 target |
| Font | Embedded 8x16 bitmap (full printable ASCII) |
| Dependencies | `aether-core`, `aether_verified`, `aether-lang`, `x86_64`, `spin`, `libm`, `uefi` |

## Project Structure

```
kernel/seal-os/
├── .cargo/config.toml      # Build target + linker config
├── Cargo.toml               # Workspace + dependencies
├── build.rs                 # Linker script rerun trigger
├── Dockerfile               # Multi-stage: build + QEMU runner
├── docker-compose.yml       # X11 forwarding for display
├── run-qemu.sh / run-qemu.ps1 # QEMU launch scripts
├── build-vbox.ps1           # Convert raw image to VirtualBox VDI
└── src/
    ├── main.rs              # Kernel entry, boot sequence, layer init
    ├── boot/
    │   ├── uefi_entry.rs    # UEFI entry and BootInfo handoff
    │   ├── boot_info.rs     # Firmware data passed to kernel_main
    │   └── ap_trampoline.rs # SMP AP bootstrap
    ├── memory/
    │   └── mod.rs           # 16MB bump allocator
    ├── drivers/
    │   ├── serial.rs        # COM1 UART (115200 baud)
    │   └── interrupts.rs    # IDT, PIC, timer, keyboard, mouse
    ├── graphics/
    │   ├── framebuffer.rs   # Pixel writing, fill_rect, volatile I/O
    │   ├── font.rs          # 8x16 bitmap font (95 glyphs)
    │   ├── console.rs       # Scrolling text console
    │   ├── splash.rs        # Boot splash + progress bar
    │   └── wallpaper.rs     # Equation wallpaper renderer
    ├── fs/
    │   ├── encoder.rs       # Data → S² manifold encoder (no_std)
    │   └── manifold_fs.rs   # ManifoldFS with T1-T5 runtime math
    ├── process/
    │   ├── task.rs          # Task struct with manifold embedding
    │   └── scheduler.rs     # ManifoldScheduler (T1+T2+T4)
    ├── syscall/
    │   └── table.rs         # Dispatch: Seal ABI + Epsilon extensions
    ├── wm/
    │   ├── compositor.rs    # Compositing WM with T1+T4
    │   ├── window.rs        # Window struct + decorations
    │   ├── event.rs         # Unified input event model
    │   ├── cursor.rs        # Software mouse cursor
    │   ├── desktop.rs       # Desktop surface + equation art
    │   └── taskbar.rs       # Bottom bar with theorem indicators
    └── apps/
        ├── shell.rs         # Built-in SealShell
        ├── terminal.rs      # Terminal emulator
        ├── seal_ide.rs      # IDE with syntax highlighting
        ├── file_manager.rs  # Visual ManifoldFS browser
        └── theorem_viewer.rs # T1-T10 dashboard
```

## License

MIT — Copyright (c) 2024 Teerth Sharma
