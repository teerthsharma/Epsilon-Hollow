# MASTER PROMPT — Build Seal OS into a Working Topological Operating System

You are building **Seal OS** — a bare-metal x86_64 operating system where every byte is a point cloud on the unit sphere S², every file move is O(1) topological surgery, and every kernel decision is driven by five formally verified topology theorems. The programming language (Aether-Lang) treats 3D manifolds as first-class primitives. The filesystem (ManifoldFS) maps its entire structure to topological invariants. The data transfer layer (Epsilon) teleports context between agents via geometric state transfer on hollow manifolds. The I/O superkernel (Aether-Link) optimizes prefetch for HFT-grade latency where the only bottleneck is hardware.

This is one of its kind. No libc. No POSIX. No Linux. Pure Rust, pure math, pure topology.

---

## Project Identity — The Mathematical Substrate

**Core principle**: You aren't listening to pseudo-random numbers; you are forcing raw silicon to execute great-circle distances on S².

- **All data = geometry on S²**: Every file, every inode, every scheduler decision maps to a 64-point cloud on the unit sphere
- **File moves = O(1) topological surgery**: Teleport is a BTreeMap pointer swap + governor clutch, not a copy
- **Five theorems drive everything**:
  - T1 (TSS): Spherical Voronoi tessellation — partitions S² into cells for indexing
  - T2 (Spectral Contraction): Convergence guarantees for iterative refinement
  - T3 (Betti Number Entropy): Topology measurement for filesystem integrity
  - T4 (AGCR/Governor): PD control law ε(t+1) = ε(t) + α·e(t) + β·de/dt — adapts precision everywhere
  - T5 (Hyperbolic Curvature): Curvature-based routing for teleport path selection
- **Three pillars**:
  1. **Aether-Lang**: Topological DSL with manifold/block/point as first-class types
  2. **Epsilon**: Zero-latency data teleportation via JL projection ℝ^E → S² + void injection
  3. **Aether-Link**: I/O superkernel — 6D telemetry extraction, sigmoid prefetch, ~18ns/cycle overhead

---

## Repository Structure

```
kernel/
├── seal-os/src/          # The OS kernel (bare-metal, #![no_std])
│   ├── main.rs           # Entry point, boot sequence, event loop
│   ├── memory/           # Heap allocator
│   ├── drivers/          # PCI, serial, interrupts, net, wifi, bluetooth, gpu, usb
│   ├── fs/               # ManifoldFS — topological filesystem
│   ├── graphics/         # Framebuffer, console, splash, font
│   ├── wm/               # Window manager, compositor, desktop
│   ├── apps/             # 15 built-in applications
│   ├── process/          # Scheduler (topology-driven)
│   ├── syscall/          # Syscall table (POSIX + Epsilon extensions)
│   ├── boot/             # UEFI entry (pure Rust, zero assembly)
│   ├── async_rt/         # Async runtime
│   ├── pkg/              # ManifoldPkg package manager
│   └── lang/             # Aether-Lang kernel integration
├── epsilon/epsilon/crates/
│   ├── epsilon/src/      # Teleportation: bridge.rs, teleport.rs, governor.rs
│   └── aether-core/src/  # Math foundation: voronoi, spectral, governor, geodesic
├── aether/
│   ├── Aether-Lang/crates/aether-lang/src/  # Language: lexer, parser, interpreter, vm
│   └── aether-link/src/  # I/O superkernel: telemetry, decision engine, fast_math
proofs/                   # Lean 4 formal proofs (10 theorems)
```

---

## Execution Plan — 15 Plans in 7 Phases

Execute these plans in phase order. Plans within the same phase can run in parallel. Each phase gate requires: `cargo +nightly build --release` passes, existing tests pass, no new clippy warnings.

### Phase 1: Foundation Safety
| Plan | File | What it does |
|------|------|--------------|
| 01 | `01-kernel-safety.md` | Fix IRQ deadlocks (AtomicU64 for TICKS, try_lock in handlers), replace `static mut FRAMEBUFFER` with `spin::Once`, fix NaN unwrap panic in ManifoldFS, add SAFETY comments |

**Gate**: Kernel builds, boots in QEMU, no deadlock under keyboard/mouse input.

### Phase 2: Memory & Storage
| Plan | File | What it does |
|------|------|--------------|
| 02 | `02-memory-allocator.md` | Replace bump allocator with linked-list free-list (real dealloc, coalescing, heap diagnostics) |
| 03 | `03-manifold-fs.md` | Add persistence (block device trait, on-disk format, WAL journal, VirtIO-blk driver) |

**Gate**: Files survive QEMU reboot. Heap can alloc/dealloc 10,000 blocks without OOM.

### Phase 3: The Three Pillars
| Plan | File | What it does |
|------|------|--------------|
| 04 | `04-aether-lang-interpreter.md` | Implement for/fn/return/break/continue in Bio interpreter, fix TopoBetti, wire render |
| 05 | `05-aether-lang-vm.md` | Complete Titan VM: EMBED/PRUNE/ATTEND opcodes, function calls, remove `_ => {}` catch-all |
| 06 | `06-epsilon-local.md` | Harden local teleport: input validation, concurrent guard, fuzzing, edge cases |

**Gate**: `cargo test -p aether-lang` passes with control flow tests. `cargo test -p epsilon` passes with fuzz-found edge cases. No `Ok(Value::Unit)` stubs remain.

### Phase 4: Connectivity
| Plan | File | What it does |
|------|------|--------------|
| 07 | `07-epsilon-remote.md` | Implement RemoteVoid: wire protocol, transport trait, permit delegation, loopback tests |
| 08 | `08-aether-link-hardware.md` | Wire Aether-Link to block device I/O: prefetch cache, IoBackend trait, VirtIO integration |
| 09 | `09-network-stack.md` | Real network: virtio-net driver, Ethernet/ARP/IP/UDP/TCP, DHCP client, DNS resolver |

**Gate**: Remote teleport works over loopback. Aether-Link prefetch cache serves hits. DHCP obtains IP under QEMU.

### Phase 5: Hardware & UI
| Plan | File | What it does |
|------|------|--------------|
| 10 | `10-drivers-real.md` | Replace all `[Sim]` drivers: real xHCI init, honest WiFi/BT status, virtio-gpu 2D ops |
| 11 | `11-window-manager.md` | Window drag/resize/z-order/focus, close/minimize/maximize, Alt-Tab, taskbar integration |

**Gate**: No `[Sim]` strings in boot log. Windows draggable and resizable in QEMU.

### Phase 6: Applications & Security
| Plan | File | What it does |
|------|------|--------------|
| 12 | `12-applications.md` | Complete all 15 apps: shell commands, IDE editing, calculator, games, file manager |
| 13 | `13-security-hardening.md` | Syscall validation, page tables, NX bit, capability model |

**Gate**: `ls`, `cat`, `teleport` work in shell. Invalid syscall returns error, not panic.

### Phase 7: Polish
| Plan | File | What it does |
|------|------|--------------|
| 14 | `14-testing-ci.md` | Integration tests, fuzzing in CI, benchmark regression gates, coverage reporting |
| 15 | `15-documentation.md` | Honest README, ARCHITECTURE.md, API docs, CONTRIBUTING.md, theorem docs |

**Gate**: All CI jobs green. README matches reality. `cargo doc` builds clean.

---

## Rules for Implementation

1. **No stubs**: Every function either works or returns an explicit error. No `Ok(Value::Unit)` for unimplemented features. No `_ => {}` catch-alls. No `[Sim]` fake data.

2. **No overclaiming**: If a driver doesn't talk to hardware, it says "not detected", not "connected to [Sim] HomeNetwork".

3. **Mathematical integrity**: The S² substrate is real. Point clouds have 64 points. Voronoi cells partition the sphere. The governor PD law uses real error/derivative terms. Don't approximate these with random numbers.

4. **Safety first**: No `static mut` (use `spin::Once` or `AtomicXxx`). No `unwrap()` in kernel code (use `if let` or `match`). `// SAFETY:` comment on every `unsafe` block. IRQ handlers use `try_lock()`.

5. **Test everything**: Every plan has acceptance criteria. Write tests before or alongside implementation. Fuzzing for security-sensitive code. Benchmark gates for performance-sensitive code.

6. **Honest errors**: Every error path returns a typed error with a human-readable message. No silent failures. `serial_println!` for kernel diagnostics.

7. **Build always passes**: After every change, verify `cargo +nightly build --release` in `kernel/seal-os/`. Never break the build across plan boundaries.

---

## How to Read a Plan File

Each of the 15 `.md` files follows this structure:

- **Goal**: One sentence — what this plan achieves
- **Current State**: What exists now, with exact file paths and line numbers
- **Gap Analysis**: What's missing or broken, with severity
- **Implementation Steps**: Numbered steps an agent can execute sequentially
- **Dependencies**: Which other plans must complete first
- **Acceptance Criteria**: Checkboxes — all must pass before the plan is "done"
- **Files to Modify/Create**: Exact paths in the repository

Read the plan file completely before starting implementation. Follow the steps in order. Check every acceptance criterion before marking done.

---

## Verification Protocol

After each phase completes:

```bash
# Build kernel
cd kernel/seal-os && cargo +nightly build --release

# Run workspace tests
cd ../.. && cargo test --workspace

# Clippy (workspace + kernel)
cargo clippy --workspace -- -D warnings
cd kernel/seal-os && cargo clippy -- -D warnings

# QEMU smoke test
# (uses the CI script — boots kernel, checks serial output for milestones)

# Miri (workspace only, not kernel)
cd ../.. && cargo +nightly miri test --workspace
```

If any step fails, fix it before proceeding to the next phase.

---

## The Vision

When all 15 plans are complete, Seal OS will be:
- A real operating system that boots on real x86_64 hardware (or QEMU)
- Every file stored as a point cloud on S², retrievable by topological similarity
- Files teleported in O(1) via topological surgery with formally verified invariants
- Cross-machine context transfer via the Epsilon protocol
- I/O prefetched at HFT-grade latency by Aether-Link's 6D telemetry engine
- Programmed in Aether-Lang, a DSL where manifolds are first-class citizens
- All five topology theorems actively driving kernel decisions
- Honest about what it can and cannot do — no simulation pretending to be hardware

This is not a demo. This is not a showcase. This is a working operating system built on mathematical substrate.
