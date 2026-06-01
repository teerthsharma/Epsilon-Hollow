# Seal OS / Epsilon-Hollow API Index

> Master index of workspace crates, their purpose, and where to find generated documentation.

---

## Workspace Layout

The repository contains two build systems:

1. **Unified Cargo Workspace** (`Cargo.toml` at repo root)
2. **Standalone crates** (`kernel/seal-os`, `kernel/seal-mkimage`) excluded from the unified workspace because they target `x86_64-unknown-uefi` or have conflicting profile requirements.

---

## Unified Workspace Crates

These crates share the workspace root (`Cargo.toml`) and build together.

### `aether-core`
- **Path:** `kernel/epsilon/epsilon/crates/aether-core`
- **Purpose:** Platform-agnostic mathematical foundation for topology, manifolds, and ML primitives. Supports both `no_std` (bare-metal kernel) and `std` (CLI/apps) via features.
- **Key Public Modules:**
  - `topology` — TDA, Betti numbers, shape verification
  - `manifold` — Time-delay embedding, sparse attention graphs
  - `tss` — Topological State Synchronization (Theorem 1): O(1) spherical Voronoi index
  - `scm` — Spectral Contraction Mapping (Theorem 2): latent predictor, contraction operator
  - `governor` — PD governor, gain-margin algebra (Theorem 4)
  - `hyperbolic_capacity` / `hyperbolic_geometry` — Hyperbolic clustering (Theorem 5)
  - `geodesic_consolidation` — GMC / entropy reduction (Theorem 3)
  - `world_model_horizon` — Predictive horizon bound (Theorem 10)
  - `cross_manifold_alignment` — CMA pipeline alignment (Theorem 9)
  - `persistent_kv_partition` — PHKP latency bound (Theorem 7)
  - `thermodynamic_plasticity` — TEB energy floor (Theorem 8)
  - `parallel_riemannian` — RGCS gradient coherence (Theorem 6)
  - `memory` — Chebyshev liveness, GC primitives
  - `ml` — Linear algebra, autograd, classifiers
- **Features:** `std` (default), `alloc`, `no_std`, `force-soft-floats`

### `epsilon`
- **Path:** `kernel/epsilon/epsilon/crates/epsilon`
- **Purpose:** Zero-Shot Context Transfer research implementation. Topological surgery on hollow S² manifolds.
- **Key Public Modules:**
  - `lib.rs` re-exports bridge, teleport, manifold, memory, governor, mock
- **Features:** `serde`

### `epsilon-os`
- **Path:** `kernel/epsilon/epsilon/crates/epsilon-os`
- **Purpose:** Topological OS user-space runtime — episodic memory with O(1) Voronoi retrieval, LLM bridge, and manifold FS frontend.
- **Key Public Modules:**
  - `main.rs` — CLI entry point
  - `manifold_fs.rs` — Manifold filesystem interface
  - `world.rs` — World model / episodic memory
  - `encoder.rs` — JL projection encoder pipeline
  - `llm.rs` — LLM bridge
  - `splash.rs` — Boot splash
- **Binary:** `epsilon-os`

### `aether-link`
- **Path:** `kernel/aether/aether-link`
- **Purpose:** Ultra-fast I/O super-kernel with adaptive POVM-based prefetching for HFT, DirectStorage, and WSL2.
- **Key Public Types:**
  - `AetherLinkKernel` — Core adaptive prefetch kernel (~18 ns/cycle)
  - `fast_math` — `fast_atan`, `fast_exp`, `fast_sigmoid`
- **Features:** `windows-directstorage`
- **Examples:** `basic_usage`, `streaming_io`, `world_model_demo`

### `aether_verified`
- **Path:** `kernel/aether/aether-verified`
- **Purpose:** Lean 4 → C → Rust verified kernels. Runtime checkers tied to Lean artifacts.
- **Key Public Modules:**
  - `aether_tss` — T1 boot gate
  - `aether_scm` — T2 boot gate
  - `aether_gmc` — T3 boot gate
  - `aether_agcr` — T4 boot gate
  - `aether_hcs` — T5 boot gate
  - `aether_world` — T6-T10 boot gate
  - `aether_betti` — Betti approximation bounds
  - `aether_chebyshev` — Chebyshev GC guard
  - `aether_governor` — PD Lyapunov stability
  - `aether_pruning` — Cauchy-Schwarz pruning bounds
- **Build:** `no_std`; uses `libm` for floating point

### `aether-lang`
- **Path:** `kernel/aether/Aether-Lang/crates/aether-lang`
- **Purpose:** AETHER Language compiler/interpreter — lexer, parser, AST, VM, and interpreter. Native scripting language for Seal OS.
- **Key Public Modules:**
  - `lexer` — Tokenizer (`Lexer`, `Token`, `TokenKind`)
  - `parser` — AST builder (`Parser`)
  - `ast` — Abstract syntax tree nodes
  - `interpreter` — Execution engine (`Interpreter`, callback registries)
  - `vm` — Virtual machine
  - `ascii_render` — ASCII visualization
  - `webgl_export` — WebGL export
- **Features:** `std` (default), `alloc`, `no_std`, `gpu`, `net`, `ml`, `pyo3`

### `aether-cli`
- **Path:** `kernel/aether/Aether-Lang/crates/aether-cli`
- **Purpose:** Cross-platform CLI for running `.aether` scripts.
- **Binary:** `aether`

### `aegis-core`
- **Path:** `kernel/aether/Aether-Lang/crates/aegis-core`
- **Purpose:** TitanClock lock-free sharded memory allocator primitives.
- **Key Public Modules:**
  - `memory` — Allocator scaffolding
  - `ml` — ML sub-modules
- **Features:** `std` (default), `alloc`

### `aegis-cli`
- **Path:** `kernel/aether/Aether-Lang/crates/aegis-cli`
- **Purpose:** Cross-platform CLI for AEGIS runtime.
- **Binary:** `aegis`

### `repl-core`
- **Path:** `kernel/aether/Aether-Lang/crates/repl-core`
- **Purpose:** Shared REPL, runner, and syntax checker used by `aether-cli` and `aegis-cli`.
- **Dependencies:** `aether-lang`, `clap`, `rustyline`

### `ubuntu-alloc-bench`
- **Path:** `tools/ubuntu-alloc-bench`
- **Purpose:** Benchmark harness comparing Seal allocator behavior against Ubuntu baseline.

---

## Standalone Crates (Excluded from Unified Workspace)

### `seal-os`
- **Path:** `kernel/seal-os`
- **Purpose:** The bare-metal x86_64 operating system. UEFI PE/COFF, Seal ABI syscall table, drivers, scheduler, ManifoldFS, graphics, and Aether-Lang runtime integration.
- **Target:** `x86_64-unknown-uefi`
- **Key Public Modules (internal crate structure):**
  - `syscall::table` — Seal ABI dispatch (native calls + Epsilon extensions)
  - `fs::manifold_fs` — ManifoldFS in-memory implementation
  - `fs::vfs` — Virtual filesystem trait
  - `process::scheduler` — ManifoldScheduler with Voronoi task groups
  - `memory::topo_ram` — TopoRAM (T1-T5 runtime)
  - `lang` — Aether-Lang runtime integration
  - `drivers` — NVMe, AHCI, e1000, xHCI, HDA, PCI, ACPI, RTC, entropy
  - `security` — ASLR, seccomp, KPTI, audit
  - `graphics` — Framebuffer, htek rendering engine, topo_render
  - `wm` — Window manager / compositor
  - `apps` — SealShell, terminal, IDE, games
- **Features:** `test-mode`

### `seal-mkimage`
- **Path:** `kernel/seal-mkimage`
- **Purpose:** Bootable UEFI disk image creator for Seal OS. Also hosts CLI audit gates:
  - `--check-doc-claim-contract`
  - `--check-theorem-log`
  - `--check-lean-proof-hygiene`
- **Target:** Host (`std`)
- **Binary:** `seal-mkimage`

---

## Generated Documentation

After building docs with Cargo:

### Unified Workspace
```bash
cargo doc --workspace --no-deps
```
Open:
```
target/doc/aether_core/index.html
target/doc/aether_link/index.html
target/doc/aether_verified/index.html
target/doc/aether_lang/index.html
target/doc/epsilon/index.html
target/doc/epsilon_os/index.html
target/doc/repl_core/index.html
target/doc/aegis_core/index.html
```

### Seal OS (Standalone)
```bash
cd kernel/seal-os
cargo doc --no-deps
```
Open:
```
kernel/seal-os/target/doc/seal_os/index.html
```

### seal-mkimage (Standalone)
```bash
cd kernel/seal-mkimage
cargo doc --no-deps
```
Open:
```
kernel/seal-mkimage/target/doc/seal_mkimage/index.html
```

---

## Quick Reference: Seal ABI → Syscall Docs

For syscall numbers and semantics, see [`docs/SYSCALLS.md`](SYSCALLS.md).

For theorem boot gates and proof strength, see [`docs/THEOREMS.md`](THEOREMS.md).

For O(1) claim discipline and runtime contracts, see [`docs/TOPOLOGICAL_OS_CONTRACT.md`](TOPOLOGICAL_OS_CONTRACT.md).
