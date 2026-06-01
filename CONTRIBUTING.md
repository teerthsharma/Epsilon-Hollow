# Contributing to Epsilon-Hollow

Thank you for your interest in Seal OS. This document will get you from zero to a working build, explain where to contribute, and set expectations for code quality, theorem gates, and benchmarks.

## Prerequisites

| Tool | Minimum version | Notes |
|------|-----------------|-------|
| Rust | 1.85+ | See `rust-toolchain.toml` |
| QEMU | 9.0+ | For UEFI + bare-metal emulation |
| OVMF | latest | UEFI firmware binaries (edk2-ovmf package) |
| Python | 3.11+ | For auxiliary scripts and kernel tests |

The `aether-kernel` crate requires **nightly Rust** and a bare-metal target (`x86_64-unknown-none` or equivalent). Install it with:

```bash
rustup target add x86_64-unknown-none
```

## Setup by operating system

### Windows

1. Install Rust via [rustup](https://rustup.rs/).
2. Install QEMU from the [official installer](https://www.qemu.org/download/#windows).
3. Ensure `qemu-system-x86_64.exe` is on your `PATH`.
4. OVMF firmware is usually bundled with QEMU on Windows; if not, copy `OVMF_CODE.fd` and `OVMF_VARS.fd` into the project root or set the path in your environment.

### macOS

```bash
brew install rustup qemu
rustup-init
rustup target add x86_64-unknown-none
```

OVMF on macOS can be installed via:

```bash
brew install ovmf
```

### Linux (Debian / Ubuntu)

```bash
sudo apt update
sudo apt install -y qemu-system-x86 ovmf build-essential curl
# Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup target add x86_64-unknown-none
```

## Quick build & test

From the repository root:

```bash
# Format, lint, and test the workspace
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

### Building the bare-metal kernel

`aether-kernel` is excluded from the default workspace because it requires nightly and a bare-metal target:

```bash
cargo +nightly build -p aether-kernel
```

### Running in QEMU

```bash
# Typical UEFI launch (adjust paths to your OVMF location)
qemu-system-x86_64 \
  -drive if=pflash,format=raw,readonly=on,file=OVMF_CODE.fd \
  -drive if=pflash,format=raw,file=OVMF_VARS.fd \
  -drive format=raw,file=target/x86_64/debug/seal-os.img \
  -serial stdio \
  -m 512M
```

See `docs/BOOT.md` for detailed boot options, headless modes, and GDB stub flags.

## Where to contribute

| Interest area | Directory | Key docs |
|---------------|-----------|----------|
| Core math (topology, manifolds, PD control, teleportation) | `kernel/epsilon/epsilon/crates/` | `docs/THEOREMS.md`, `docs/TOPOLOGICAL_OS_CONTRACT.md` |
| DSL / language work (Aether-Lang) | `kernel/aether/Aether-Lang/crates/` | `docs/SEAL_OS_GUIDE.md` |
| I/O super-kernel & benchmarks | `kernel/aether/aether-link/` | `BENCHMARKS.md` |
| Memory allocator | `kernel/seal-os/src/memory/` | `docs/MEMORY.md` |
| Filesystem (ManifoldFS) | `kernel/seal-os/src/fs/` | `docs/MANIFOLDFS.md` |
| Process scheduler | `kernel/seal-os/src/process/` | `docs/THEOREMS.md` |
| Drivers | `kernel/seal-os/src/drivers/` | `docs/AETHER_HARDWARE_CAPS.md` |
| Network stack | `kernel/seal-os/src/net/` | — |
| Security / theorem gates | `kernel/seal-os/src/security/` | `SECURITY.md` |
| Graphics / window manager | `kernel/seal-os/src/wm/` | `docs/VRAM_TOPOLOGY_FAST_PATH.md` |
| Build / CI | `.github/workflows/`, `scripts/` | `docs/CI.md`, `docs/LOCAL_CI.md` |

## Debugging

### QEMU GDB stub

Launch QEMU with `-s -S` (wait for GDB on `:1234`):

```bash
qemu-system-x86_64 ... -s -S
```

In another terminal:

```bash
gdb target/x86_64/debug/seal-os
(gdb) target remote :1234
(gdb) break main
(gdb) continue
```

### Serial logs

Seal OS prints boot and runtime diagnostics to the serial port. Always capture serial output when reporting bugs:

```bash
qemu-system-x86_64 ... -serial file:serial.log
```

### Headless proof mode

For CI-like verification without a display window, use the headless flags documented in `scripts/test_kernel.sh` and `docs/BOOT.md`.

## Style guide

- **Formatting**: `cargo fmt --all`
- **Linting**: `cargo clippy --workspace --all-targets -- -D warnings`
- All CI jobs must pass before a PR is merged.
- Keep commits focused: one logical change per commit.
- Add or update tests for any behavioural change.

## Commit message format

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <short summary>

<body: explain why, not just what>

<footer: breaking changes, issue refs>
```

Examples:

```
feat(scheduler): add Voronoi cell affinity for ML tasks

Reduces context-switch latency for pinned workloads by ~12%.
BREAKING CHANGE: `Task::affinity` now returns `Option<CellId>`.
```

```
docs(theorems): clarify T4/AGCR gate invariant

Closes #123
```

Common types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`.

## Benchmark protocol for performance changes

If your change is performance-sensitive, run Criterion benchmarks locally before opening a PR:

```bash
# Primary I/O cycle benchmark
cargo bench --bench io_cycle --manifest-path kernel/aether/aether-link/Cargo.toml

# Compile all benchmarks without running (CI parity)
cargo bench --workspace --no-run
```

Include before/after numbers in the PR description. See `BENCHMARKS.md` for expected ranges and how to read regression output.

## Theorem-gate requirements (T1–T5)

Seal OS kernel code is organized around ten theorems (T1–T10). Changes that touch **T1–T5 runtime paths** must:

1. Preserve the existing theorem gate (or extend it with a new proof / invariant).
2. Update `docs/THEOREMS.md` if the theorem statement, proof sketch, or runtime check changes.
3. Ensure the headless boot proof still prints `All T1-T10 theorems VERIFIED; T1-T5 ACTIVE in runtime paths`.

If you are unsure whether your change affects a theorem gate, open a draft PR and tag `@teerthsharma` for review.

## Good first issues

New contributors are welcome. Look for issues labelled:

- `good first issue` — small, self-contained changes with clear acceptance criteria.
- `help wanted` — larger tasks where maintainer bandwidth is limited.

Suggested starter areas:

- Documentation fixes or rustdoc improvements.
- Adding unit tests to `kernel/epsilon/epsilon/crates/` math utilities.
- Benchmark harness improvements in `kernel/aether/aether-link/`.
- Driver probe logging or error-message clarity in `kernel/seal-os/src/drivers/`.

## Questions?

- **Usage / open-ended ideas**: [GitHub Discussions](https://github.com/teerthsharma/Epsilon-Hollow/discussions)
- **Security issues**: See `SECURITY.md` — do **not** open a public issue.
- **Code of conduct**: See `CODE_OF_CONDUCT.md`.

Thank you for helping make Seal OS rigorous, fast, and geometrically sound.
