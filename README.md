<p align="center">
  <img src="assets/logo.svg" alt="Seal OS Logo" width="180">
</p>

<h1 align="center">Seal OS (Epsilon-Hollow)</h1>

<p align="center">
  A research operating system where core runtime decisions are modeled as topology on the unit sphere (S²).
</p>

<p align="center">
  <a href=".github/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/teerthsharma/epsilon-hollow/ci.yml?branch=main&label=CI&style=flat-square" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/teerthsharma/epsilon-hollow?style=flat-square&color=00aaff" alt="License: MIT"></a>
  <a href="#build-and-run"><img src="https://img.shields.io/badge/rust-stable-orange?style=flat-square&logo=rust" alt="Rust stable"></a>
  <a href="docs/THEOREMS.md"><img src="https://img.shields.io/badge/Lean%204-proofs-green?style=flat-square" alt="Lean 4 proofs"></a>
</p>

---

## Overview

Seal OS is a bare-metal `x86_64` research kernel written in Rust (`no_std`) with a small amount of assembly for hardware bring-up.

The project explores a geometric approach to systems design:
- memory allocation and layout with topological structures,
- filesystem metadata and lookup using spatial embeddings,
- scheduling with Voronoi-style partitioning,
- theorem-gated runtime checks for critical invariants.

This is not a POSIX-compatible OS and is not intended as a general-purpose Linux replacement.

---

## Current Status

### Working today
- UEFI boot path with SMP bring-up and APIC/ACPI initialization
- Memory management (frame allocation, slab allocator, paging)
- Filesystem stack (ManifoldFS + VFS + additional filesystem support)
- Process and syscall surface (fork/exec/signals/pipes and related primitives)
- Desktop pipeline (framebuffer, compositor/window manager, shell, apps)
- Network stack foundations (TCP/UDP/DHCP/DNS + constrained TLS path)
- CI proof gates for key runtime and theorem checkpoints

### Known limits
- Still a research kernel with active experimental subsystems
- No full POSIX/libc environment
- TLS and some hardware acceleration paths are intentionally incomplete
- Not production-ready

For a detailed status breakdown, see:
- `docs/SEAL_OS_GUIDE.md`
- `docs/THEOREMS.md`
- `docs/CI.md`

---

## Quick Start

### Prerequisites
- Rust stable (workspace crates)
- Rust nightly (kernel build)
- QEMU + OVMF/EDK2 for VM boot

### 1) Build workspace
```bash
cargo build --workspace
cargo test --workspace
```

### 2) Build kernel
```bash
cd kernel/seal-os
cargo +nightly build --release
```

### 3) Build image
```bash
cd ../seal-mkimage
cargo +stable run --release
```

### 4) Boot in QEMU
```bash
cd ../seal-os
./run-qemu.sh
```

For additional VM and local CI workflows, see `docs/VM_RUNBOOK.md` and `docs/LOCAL_CI.md`.

---

## Repository Layout

```text
kernel/          Kernel, boot path, low-level runtime
userspace/       Userland components and runtime-facing tools
apps/            Applications and higher-level interfaces
docs/            Design docs, guides, audits, and plans
tools/           Host-side tools and utilities
scripts/         Automation scripts for build/test flows
tests/           Test fixtures and validation helpers
```

---

## Documentation

Start here:
- `docs/SEAL_OS_GUIDE.md` — project guide
- `docs/BUILD_SYSTEM.md` — build system details
- `docs/API_INDEX.md` — API and interface references
- `docs/THREAT_MODEL.md` — security model and assumptions
- `docs/COMMUNITY.md` — contribution and community guidance

Legacy long-form README content is preserved at `docs/archive/README_OLD.md`.

---

## Contributing

Please read `CONTRIBUTING.md` before opening changes. For security reports, see `SECURITY.md`.

---

## License

MIT — see `LICENSE`.
