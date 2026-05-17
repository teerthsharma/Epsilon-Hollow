# Epsilon-Hollow

The largest open-source geometrical scientific Rust operating system.

## Seal OS — The Geometrical Operating System

**All data is geometry on S². File moves are O(1) topological surgery.**

Seal OS is a bare-metal x86_64 operating system built from scratch in Rust. Every kernel subsystem — memory, scheduling, filesystem, display — is driven by five topology theorems (T1-T5). It boots on QEMU and real hardware via USB ISO.

### Run it

```bash
# Docker (recommended)
cd kernel/seal-os && docker compose up --build

# QEMU (from ISO)
qemu-system-x86_64 -cdrom seal-os.iso -serial stdio -m 512M -vga std

# Real hardware
dd if=seal-os.iso of=/dev/sdX bs=4M status=progress
```

Download the latest ISO from [Releases](../../releases).

### What's inside

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
Layer 1   │ Memory: 16MB heap, bump allocator
Layer 0   │ Boot: Multiboot2, 32→64 trampoline, GRUB ISO
          └────── ALL LAYERS DRIVEN BY T1-T5 THEOREMS ──────
```

| Property | Value |
|----------|-------|
| Target | `x86_64-unknown-none` |
| Kernel size | ~260KB |
| ISO size | < 10MB |
| Display | 1024x768x32bpp (Multiboot2 framebuffer) |
| Dependencies | `aether-core` (T1-T5), `x86_64`, `spin`, `libm` |

### ManifoldFS

The filesystem where files are not byte sequences — they are 64-point clouds on the unit sphere S².

```
Write: data → trigram hash → JL project → L2 normalize → 64 pts on S² (1,536 bytes)
Move:  topological surgery → O(1) regardless of file size
Find:  encode query → Voronoi cell → O(n/K) content-addressable search
```

### Theorem Integration

| Subsystem | T1 (Voronoi) | T2 (SCM) | T3 (Entropy) | T4 (Governor) | T5 (Hyperbolic) |
|-----------|:---:|:---:|:---:|:---:|:---:|
| Memory | frame locality | — | — | heap growth | — |
| Scheduler | task groups | predict next | — | timeslice adapt | — |
| Filesystem | file lookup O(1) | file prefetch | dir reorganize | cache pressure | path resolution |
| Window Mgr | hit-test | — | — | FPS adaptation | — |

See [kernel/seal-os/README.md](kernel/seal-os/README.md) for full documentation, [ARCHITECTURE.md](kernel/seal-os/ARCHITECTURE.md) for internals, and [TESTING.md](kernel/seal-os/TESTING.md) for verification.

---

## The Repository

Three projects. One repository. Mathematically verified.

Epsilon-Hollow is built on 10 formally verified theorems (Lean 4 proofs), a custom
topological DSL (Aether-Lang), and a novel data teleportation primitive that transfers
context between agents in O(1) via topological surgery on S² manifolds.

### Epsilon-OS — Geometrical World Model

An AI assistant powered by a topological world model:

- Episodic memory stored as L2-normalized vectors on S^(D-1)
- O(1) retrieval via spherical Voronoi tessellation (TSS, Theorem 1)
- Forward dynamics via spectral contraction (SCM, Theorem 2)
- LLM synthesis for reasoning over retrieved context
- Betti-0 clustering for semantic grouping

```
/step <text>     Ingest observation into topological memory
/query <text>    Retrieve from memory + LLM synthesis
/dream <text>    Counterfactual rollout through latent dynamics
/memory          Memory stats: Betti-0, TSS state, cell distribution
/theorems        Run T1-T10 verification suite
```

### Epsilon-IDE — Local AI Coding Assistant

A standalone app (no cloud, no API keys) with three-tier model routing:

| Tier | Model Size | Latency | Use Case |
|------|-----------|---------|----------|
| Fast | 1.5B (~1 GB) | 1-2s | Simple queries |
| Balanced | 7B (~4 GB) | 5-15s | Moderate tasks |
| Deep | 33B (CPU/SSD) | 30-120s | Complex architecture |

### Aether-Lang — Topological DSL

All data processing runs through a manifold-native language with seal loops, tilde terminators, and topological convergence primitives. Supports tree-walking interpretation (Bio mode) and bytecode VM (Titan mode).

### aether-core — Math Foundation

The `no_std` mathematics library powering everything:

| Module | What |
|--------|------|
| `tss.rs` | `SphericalVoronoiIndex<K>` — O(1) cell lookup on S² |
| `scm.rs` | `SpectralContractionOperator<D>` — Banach fixed-point |
| `topology.rs` | Betti number computation (filtration-based TDA) |
| `governor.rs` | `GeometricGovernor` — PD adaptive threshold control |
| `manifold.rs` | Point cloud, sparse graph, topological surgery |

## Theorems

Ten formally verified theorems govern the system. Each is implemented in Rust,
verified at boot, and backed by Lean 4 proofs.

| ID | Name | What it governs | Status |
|----|------|-----------------|--------|
| T1 | TSS  | O(1) retrieval via spherical Voronoi tessellation | **Active** |
| T2 | SCM  | Spectral contraction toward attractor (Banach) | **Active** |
| T3 | GMC  | Renyi entropy bound on memory consolidation | Verified |
| T4 | AGCR | Governor convergence (eigenvalue-bounded PD control) | Verified |
| T5 | HCS  | Hyperbolic vs. Euclidean separation ratio | Verified |
| T6 | RGCS | Tangent deviation bound for sync frequency | Verified |
| T7 | PHKP | Betti-guided latency (topological persistence) | Verified |
| T8 | TEB  | Landauer energy bound per bit erasure | Verified |
| T9 | CMA  | Alignment error (Procrustes curvature + SVD) | Verified |
| T10 | WPHB | Predictive horizon (information + stability) | Verified |

## Build

Requires Rust 1.85+.

```bash
cargo build --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

Build Seal OS (requires nightly):

```bash
cd kernel/seal-os
cargo +nightly build --release
```

Run the world model shell:

```bash
cargo run -p epsilon-os
```

## CI

GitHub Actions runs: fmt, clippy, test, bench-compile, miri, cargo-audit, cargo-deny, docs, BOM validation, Python checks, Lean proofs. The [release workflow](.github/workflows/release.yml) builds the Seal OS ISO and publishes it to GitHub Releases on every tag.

## License

MIT License. Copyright (c) 2024 Teerth Sharma. See [LICENSE](LICENSE).
