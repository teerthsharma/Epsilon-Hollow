# Seal OS v1.0.0-alpha

**The Geometrical Operating System** — where all data is geometry on S² and file moves are O(1) topological surgery.

Seal OS is a bare-metal x86_64 operating system built from scratch. Every kernel subsystem — memory, scheduling, filesystem, display — is driven by five topology theorems (T1-T5) from the aether-core mathematics library.

## Quick Start

### Run in Docker (recommended)

```bash
# From repo root — builds kernel, creates ISO, launches QEMU
cd kernel/seal-os

# Linux/macOS (requires X11)
./run.sh

# Windows (requires VcXsrv or Xming with "Disable access control")
powershell -File run.ps1

# Or directly with docker compose
docker compose up --build
```

### Build from source

```bash
# Prerequisites: Rust nightly, grub-mkrescue, xorriso, qemu
cd kernel/seal-os
cargo +nightly build --release

# Create bootable ISO
mkdir -p iso/boot/grub
cp target/x86_64-unknown-none/release/seal-os iso/boot/kernel.bin
cp boot/grub/grub.cfg iso/boot/grub/grub.cfg
grub-mkrescue -o seal-os.iso iso/

# Run in QEMU
qemu-system-x86_64 -cdrom seal-os.iso -serial stdio -m 512M -vga std
```

### Boot on real hardware

```bash
dd if=seal-os.iso of=/dev/sdX bs=4M status=progress
# Boot from USB
```

## Architecture

```
Layer 10  │ Terminal, Seal IDE, File Manager, Theorem Viewer
Layer 9   │ Desktop: wallpaper (Schwarzschild + Faraday), taskbar
Layer 8   │ Window Manager: compositor, decorations, cursor
Layer 7   │ Shell: ls, mv (O(1) teleport), find, theorems, race
Layer 6   │ Syscalls: POSIX + Epsilon extensions (manifold_query, teleport)
Layer 5   │ ManifoldScheduler: T1 Voronoi task groups, T4 adaptive timeslice
Layer 4   │ ManifoldFS: THE filesystem — all data = 64 pts on S²
Layer 3   │ Framebuffer: 1024x768x32, 8x16 font, boot splash
Layer 2   │ Interrupts: IDT, PIC, timer (IRQ0), keyboard, mouse
Layer 1   │ Memory: 4MB heap, bump allocator
Layer 0   │ Boot: Multiboot2, 32→64 trampoline, GRUB ISO
          └────── ALL LAYERS DRIVEN BY T1-T5 THEOREMS ──────
```

## Theorem Integration

| Subsystem | T1 (Voronoi) | T2 (SCM) | T3 (Entropy) | T4 (Governor) | T5 (Hyperbolic) |
|-----------|:---:|:---:|:---:|:---:|:---:|
| Memory | frame locality | — | — | heap growth | — |
| Scheduler | task groups | predict next | — | timeslice adapt | — |
| Filesystem | file lookup O(1) | file prefetch | dir reorganize | cache pressure | path resolution |
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

The filesystem where files are not byte sequences — they are `ManifoldPayload`s: 64-point clouds on the unit sphere S².

```
Write: data → trigram hash → JL project → L2 normalize → 64 pts on S² (1,536 bytes)
Move:  extract payload → topological surgery → O(1) regardless of file size
Find:  encode query → Voronoi cell → O(n/K) content-addressable search
```

### Shell Commands

```
seal:/$ help
ls [path]     — list directory (shows Voronoi cells, payload points)
mkdir <name>  — create directory
touch <name> <content> — create file (encodes to S² geometry)
mv <src> <dst>         — O(1) teleport via topological surgery
find <query>  — content-addressable search via T1 Voronoi
cd <path>     — change directory
theorems      — show T1-T5 live status
race <size>   — benchmark: traditional copy vs manifold teleport
stats         — filesystem statistics
ps            — list processes
uname         — system info
```

### Teleportation Benchmark

```
seal:/$ race 2000000000
Race: 2GB transfer
========================
Traditional copy: 1000 ms
Manifold teleport: 50 us (O(1) topological surgery)
Speedup: 20000x

Payload: 64 pts * 3 coords * 8 bytes = 1,536 bytes
Regardless of original file size!
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
- T1-T5 live metrics with progress bars
- Governor epsilon, entropy, Betti numbers
- Prefetch accuracy, teleport count, scheduler ticks

## Desktop

- **Wallpaper**: Procedural black hole visualization with accretion disk rings, plus grid pattern
- **Taskbar**: Bottom bar with T1-T5 status indicators, epsilon readout, Seal OS branding
- **Windows**: Compositing WM with title bars, close buttons, dragging, z-ordering

## Build Details

| Property | Value |
|----------|-------|
| Target | `x86_64-unknown-none` |
| Kernel size | ~260KB |
| ISO size | < 10MB |
| Heap | 4MB static bump allocator |
| Display | 1024x768x32bpp (Multiboot2 framebuffer) |
| Font | Embedded 8x16 bitmap (full printable ASCII) |
| Dependencies | `aether-core` (T1-T5), `x86_64`, `spin`, `libm` |

## Project Structure

```
kernel/seal-os/
├── .cargo/config.toml      # Build target + linker config
├── Cargo.toml               # Workspace + dependencies
├── build.rs                 # Linker script rerun trigger
├── linker.ld                # Kernel at 1MB, multiboot header first
├── Dockerfile               # Multi-stage: build + QEMU runner
├── docker-compose.yml       # X11 forwarding for display
├── boot/grub/grub.cfg       # GRUB menu entry + framebuffer mode
├── run.sh / run.ps1         # Launch scripts
└── src/
    ├── main.rs              # Kernel entry, boot sequence, layer init
    ├── boot/
    │   ├── boot.S           # 32→64 trampoline with page tables
    │   ├── mod.rs           # global_asm! include
    │   └── multiboot2.rs    # Multiboot2 header + FB request tag
    ├── memory/
    │   └── mod.rs           # 4MB bump allocator
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
    │   └── manifold_fs.rs   # ManifoldFS with T1-T5 (no_std)
    ├── process/
    │   ├── task.rs          # Task struct with manifold embedding
    │   └── scheduler.rs     # ManifoldScheduler (T1+T2+T4)
    ├── syscall/
    │   └── table.rs         # Dispatch: POSIX + Epsilon extensions
    ├── wm/
    │   ├── compositor.rs    # Compositing WM with T1+T4
    │   ├── window.rs        # Window struct + decorations
    │   ├── event.rs         # Unified input event model
    │   ├── cursor.rs        # Software mouse cursor
    │   ├── desktop.rs       # Desktop surface + equation art
    │   └── taskbar.rs       # Bottom bar with theorem indicators
    └── apps/
        ├── shell.rs         # Built-in shell (ls, mv, find, etc.)
        ├── terminal.rs      # Terminal emulator
        ├── seal_ide.rs      # IDE with syntax highlighting
        ├── file_manager.rs  # Visual ManifoldFS browser
        └── theorem_viewer.rs # T1-T5 live dashboard
```

## License

MIT — Copyright (c) 2024 Teerth Sharma
