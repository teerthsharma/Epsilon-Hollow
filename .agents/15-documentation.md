# 15 — Documentation: Architecture Docs, API Reference, Contributor Guide

## Goal

Create honest, accurate documentation that matches the actual implementation — no overclaiming. Every feature documented as "working" must be verifiable by running the code.

## Current State

- `README.md` (1012 lines): comprehensive but overclaims in several areas (claims 15 codec support, TCP/UDP/TLS stack, full WiFi/Bluetooth)
- `SECURITY.md`: threat model, known limitations (honest about RemoteVoid stub)
- Lean 4 proofs: 10 theorems with Mathlib-backed proofs
- No API documentation beyond code comments
- No contributor guide
- No architecture decision records

## Implementation Steps

1. **Update README.md to match reality**
   - Add status badges: "Working", "Partial", "Planned" next to each feature
   - Driver table: mark PCI/Serial/Interrupts as "Real", WiFi/BT/USB as "Simulated" (or "Real" after plan 10)
   - Network stack: mark as "Simulated" (or "Real" after plan 09)
   - Media player: mark codec list as "Planned" not "Supported"
   - Aether-Lang: note interpreter stubs (or "Complete" after plan 04)

2. **Create ARCHITECTURE.md**
   - Boot sequence diagram (UEFI → kernel_main → layers 1-14)
   - Memory layout: heap (16MB bump → free-list after plan 02), framebuffer (GOP), stack
   - S² substrate: how all data maps to point clouds on the unit sphere
   - Theorem integration: which theorems drive which kernel subsystems
   - Dependency graph between crates

3. **Create API.md for each major crate**
   - `aether-core`: Governor, Voronoi, Spectral, Geodesic — public API surface
   - `epsilon`: Bridge, Teleport, Governor — public API with examples
   - `aether-link`: TelemetryExtractor, DecisionEngine — configuration and usage
   - `aether-lang`: Lexer, Parser, Interpreter, VM — how to evaluate a program
   - `seal-os`: Syscall table, ManifoldFS operations, driver APIs

4. **Create CONTRIBUTING.md**
   - Build prerequisites: Rust nightly, QEMU, Lean 4 (optional)
   - Build commands: `cargo build --release` (workspace), kernel build steps
   - Testing: `cargo test --workspace`, QEMU smoke test, Miri
   - Code style: no comments unless "why" is non-obvious, `// SAFETY:` for unsafe
   - PR process: branch naming, commit message format, CI must pass

5. **Create docs/THEOREMS.md**
   - T1 (TSS): Spherical Voronoi tessellation — what it does, where used (scheduler, ManifoldFS)
   - T2 (Spectral Contraction): convergence guarantees — where used (scheduler predictions)
   - T3 (Betti Number Entropy): topology measurement — where used (ManifoldFS integrity)
   - T4 (AGCR/Governor): PD control law — where used (ε adaptation everywhere)
   - T5 (Hyperbolic Curvature): curvature-based routing — where used (teleport path selection)
   - Link each theorem to its Lean 4 proof file

6. **Create docs/AETHER_LANG_SPEC.md**
   - Language grammar (EBNF)
   - Type system: Num, Bool, Str, Manifold, Block, Point, List, Function
   - Built-in modules: math, topology, ml, Seal
   - Example programs with expected output
   - Bio mode vs Titan mode: when to use which

7. **Inline documentation**
   - Add `//!` module docs to every `mod.rs` and `lib.rs`
   - Add `/// ` doc comments to all public functions and types
   - Generate with `cargo doc --workspace --no-deps`
   - Verify no broken doc links

8. **Update SECURITY.md**
   - Update known limitations based on completed plans
   - Document capability model (after plan 13)
   - Document syscall validation rules
   - Responsible disclosure process

## Dependencies

- Should run last (documents the state after all other plans complete)
- Can run in parallel with plan 14 (testing)

## Acceptance Criteria

- [ ] README status badges match actual implementation state
- [ ] ARCHITECTURE.md accurately describes boot sequence and memory layout
- [ ] API docs exist for all 5 major crates
- [ ] CONTRIBUTING.md enables a new contributor to build and test in <15 minutes
- [ ] THEOREMS.md links each theorem to its code usage and Lean 4 proof
- [ ] `cargo doc --workspace --no-deps` builds with zero warnings
- [ ] No claim in any document is contradicted by the actual code

## Files to Modify

- `README.md`
- `SECURITY.md`

## Files to Create

- `ARCHITECTURE.md`
- `CONTRIBUTING.md`
- `docs/THEOREMS.md`
- `docs/AETHER_LANG_SPEC.md`
- `docs/API_AETHER_CORE.md`
- `docs/API_EPSILON.md`
- `docs/API_AETHER_LINK.md`
- `docs/API_AETHER_LANG.md`
- `docs/API_SEAL_OS.md`
