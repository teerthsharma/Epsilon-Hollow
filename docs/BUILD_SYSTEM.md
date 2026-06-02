# Build System Guide

## Workspace Structure

The repository uses a Cargo workspace defined in the root `Cargo.toml`.

### Workspace Members

| Crate | Path | Description |
|-------|------|-------------|
| `aether-core` | `kernel/epsilon/epsilon/crates/aether-core` | Math foundation (topology, manifolds, ML) |
| `epsilon` | `kernel/epsilon/epsilon/crates/epsilon` | Context Teleportation research implementation |
| `epsilon-os` | `kernel/epsilon/epsilon/crates/epsilon-os` | Topological OS runtime |
| `aether-link` | `kernel/aether/aether-link` | Ultra-fast I/O super-kernel |
| `aether_verified` | `kernel/aether/aether-verified` | Lean 4 verified kernels |
| `aether-lang` | `kernel/aether/Aether-Lang/crates/aether-lang` | AETHER language compiler / interpreter |
| `aether-cli` | `kernel/aether/Aether-Lang/crates/aether-cli` | AETHER cross-platform CLI |
| `aegis-core` | `kernel/aether/Aether-Lang/crates/aegis-core` | TitanClock lock-free allocator |
| `aegis-cli` | `kernel/aether/Aether-Lang/crates/aegis-cli` | AEGIS cross-platform CLI |
| `repl-core` | `kernel/aether/Aether-Lang/crates/repl-core` | Shared REPL and runner |
| `ubuntu-alloc-bench` | `tools/ubuntu-alloc-bench` | Host allocation baseline benchmark |

### Excluded from Workspace

These crates are **excluded** from the root workspace because they require special toolchains, targets, or are experimental:

- `kernel/seal-os` — Bare-metal x86_64 OS (requires **nightly** + `rust-src` + custom target)
- `kernel/aether/Aether-Lang/crates/aether-kernel` — Kernel-space AETHER (depends on nightly bare-metal)
- `kernel/aether/aether-nexus` — Experimental nexus component
- `future/` — Future research tracks

## Toolchain Requirements

### Root Workspace (Stable)

- **Channel:** `1.85` (pinned in `rust-toolchain.toml`)
- **Components:** `rustfmt`, `clippy`, `rust-docs`
- **Edition:** `2021`
- **Target:** Host triple (e.g., `x86_64-pc-windows-msvc`, `x86_64-unknown-linux-gnu`)

### Seal OS (Nightly)

- **Location:** `kernel/seal-os/`
- **Channel:** `nightly` (pinned in `kernel/seal-os/rust-toolchain.toml`)
- **Components:** `rust-src`, `llvm-tools-preview`
- **Target:** Bare-metal `x86_64` (custom target JSON or `x86_64-unknown-none`)
- **Profile:** `panic = "abort"` is mandatory for both `dev` and `release` profiles.

> **Why nightly?** Seal OS is a bare-metal research OS. It needs `rust-src` for building `core`/`alloc` with a custom target, and `llvm-tools-preview` for `llvm-objcopy`/`llvm-readobj` in the boot-image build pipeline.

## How to Build

### Build the entire workspace

```bash
cargo build --workspace
```

### Run all workspace tests

```bash
cargo test --workspace
```

### Build Seal OS

```bash
cd kernel/seal-os
# Ensure nightly is active
rustup show
# Build with the custom bare-metal target
cargo build --target <path-to-your-target>.json
```

### Run benchmarks

```bash
cargo bench -p aether-link
```

## Dependency Policy

License checking is configured in `deny.toml`. Only **permissive** licenses are allowed:

- MIT
- Apache-2.0 / Apache-2.0 WITH LLVM-exception
- BSD-2-Clause / BSD-3-Clause
- ISC
- Unicode-DFS-2016 / Unicode-3.0
- Zlib, MPL-2.0, CC0-1.0, BSL-1.0, CDLA-Permissive-2.0

**Copyleft licenses (GPL, LGPL, AGPL, etc.) are banned.** If a dependency introduces a copyleft license, it must be replaced or isolated in a separate process.

To run the license audit (requires `cargo-deny`):

```bash
cargo deny check
```

## Cargo.lock Hygiene

`Cargo.lock` is **committed** to the repository (it is *not* ignored in `.gitignore`). This ensures reproducible CI builds and deterministic dependency resolution for all binaries in the workspace.

If you add or update a dependency, regenerate the lockfile:

```bash
cargo generate-lockfile
```

## Common Build Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `rustc` version mismatch | Wrong toolchain active | Run `rustup show` in the repo root; the `rust-toolchain.toml` will auto-select `1.85` |
| `cargo deny` fails on license | Dependency uses an unapproved license | Review `deny.toml`; add an exception only if the license is truly permissive, otherwise replace the crate |
| Seal OS build fails with linker error | Missing bare-metal target or `rust-src` | Install nightly components: `rustup component add rust-src llvm-tools-preview --toolchain nightly` |
| `no_std` feature not found | `aether-core` built with default `std` feature | Disable defaults: `aether-core = { default-features = false, features = ["no_std"] }` |
| Workspace member not found | Path in `Cargo.toml` is incorrect | Verify relative paths; remember that `kernel/seal-os/` paths are relative to `kernel/seal-os/Cargo.toml` |
| ` Blocking waiting for file lock on build directory` | Concurrent Cargo processes | Wait for the other process to finish, or run `cargo` from a single shell |

## Profiles

### Workspace Root (`Cargo.toml`)

```toml
[profile.dev]
opt-level = 1

[profile.release]
lto = "fat"
codegen-units = 1
opt-level = 3
panic = "abort"

[profile.bench]
lto = true
codegen-units = 1
```

### Seal OS (`kernel/seal-os/Cargo.toml`)

Seal OS sets `panic = "abort"` in **both** `dev` and `release` because unwinding is not supported in bare-metal environments without an unwinder.

## Aether Self-Hosting Roadmap

Seal OS is moving toward a build system where the Aether language drives its own compilation, reducing reliance on host Cargo/Rust toolchains. This is staged in three phases.

### Phase 0 — Rust Driver + Bootstrap Compiler *(Current)*

A Rust module (`kernel/seal-mkimage/src/aether_build.rs`) and a shell script (`scripts/build_aether.sh`) discover Aether sources (`.ae` / `.aether`), invoke the bootstrap `aether-cli` compiler for syntax verification, and emit placeholder **Aether Object Images** (`.aeo`) into `target/aether-build/`.

Rust remains responsible for:
- Directory traversal and build orchestration
- Bare-metal image linking (`seal-mkimage`)
- Host-side build utilities

**Key files:**
- `kernel/seal-mkimage/src/aether_build.rs` — Rust build driver
- `scripts/build_aether.sh` — Shell orchestrator
- `target/aether-build/aether-build-manifest.json` — Incremental-build manifest

### Phase 1 — Aether Builds Aether

The Aether bootstrap compiler gains a `compile` subcommand that emits serialized TitanVM bytecode directly to `.aeo` files. The Rust driver shrinks to a thin linker wrapper:

1. `aether-cli compile kernel/aether/*.ae -o target/aether-build/`
2. Rust driver reads `.aeo` blobs and embeds them into the OS disk image.

**Blockers to resolve:**
- Bytecode serialization format for `OpCode` vectors
- Bare-metal TitanVM loader in `kernel/seal-os`
- Incremental rebuild tracking inside Aether itself

### Phase 2 — Aether Builds Rust Subset

Aether compiler front-end learns enough of the Rust type system to translate a **restricted subset** of Rust (no `std`, no proc macros, no async) into Aether IR, then into the same bytecode/object pipeline.

This allows core kernel shims (allocators, serial drivers, GDT/IDT setup) to be expressed in Rust syntax but compiled by the Aether toolchain, proving the compiler can handle systems-level code.

**Scope of Rust subset:**
- `no_std` + `no_alloc` safe code
- Raw pointers and `unsafe` blocks (with Aether safety annotations)
- Inline assembly (passthrough to LLVM backend)
- Zero-cost abstractions (enums, structs, generics without trait objects)

### Phase 3 — Full Self-Host

The Aether compiler is rewritten in Aether and compiles itself. At this point:

- `cargo` is no longer required on the host.
- The only host dependency is a C compiler or assembler for the final linker step (or an Aether-native linker).
- Seal OS can rebuild itself from source inside a running Seal OS instance.

**Verification target:**
> Boot a Seal OS VM, mount the source partition, run `aether build --self-host`, and produce a byte-identical OS image.

## Notes

- Do **not** modify `kernel/seal-os/Cargo.toml` workspace membership. It is intentionally excluded and carries its own `[workspace]` header to remain a standalone bare-metal crate.
- Feature flags in `kernel/seal-os` are minimal (`default = []`, `test-mode = []`) to keep the kernel image small.
- The `scripts/build_aether.sh` script is additive: it does not replace `cargo build` or `scripts/build_iso.sh`. Run it independently when working on Aether kernel modules.
