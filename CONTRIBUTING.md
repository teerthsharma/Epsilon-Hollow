# Contributing to Epsilon-Hollow

Thank you for your interest in Seal OS. This document will get you from zero to a working build, explain where to contribute, and set expectations for code quality, theorem gates, and benchmarks.

## Prerequisites

| Tool | Minimum version | Notes |
|------|-----------------|-------|
| Rust | 1.85+ | See `rust-toolchain.toml` |
| QEMU | 9.0+ | For UEFI + bare-metal emulation |
| OVMF | latest | UEFI firmware binaries (edk2-ovmf package) |
| Python | 3.11+ | For auxiliary scripts and kernel tests |

The `seal-os` kernel crate requires **nightly Rust** and a bare-metal target (`x86_64-unknown-none` or equivalent). Install it with:

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

`seal-os` is excluded from the default workspace because it requires nightly and a bare-metal target:

```bash
cargo +nightly build --manifest-path kernel/seal-os/Cargo.toml
```

### Running in QEMU

```bash
# Typical UEFI launch (adjust paths to your OVMF location)
qemu-system-x86_64 \
  -drive if=pflash,format=raw,readonly=on,file=OVMF_CODE.fd \
  -drive if=pflash,format=raw,file=OVMF_VARS.fd \
  -drive format=raw,file=target/x86_64-unknown-uefi/release/seal-os.img \
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
| Linux kernel replacement (ABI, boot, parity gates) | `tests/linux_parity/`, `kernel/seal-os/src/syscall/` | [`docs/design/LINUX-REPLACEMENT.md`](docs/design/LINUX-REPLACEMENT.md) |
| Ports of existing kernel code | `ports/` | [`PORTING.md`](PORTING.md), [`ports/README.md`](ports/README.md) |

## Debugging

### QEMU GDB stub

Launch QEMU with `-s -S` (wait for GDB on `:1234`):

```bash
qemu-system-x86_64 ... -s -S
```

In another terminal:

```bash
gdb target/x86_64-unknown-uefi/release/seal-os
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

## Theorem status lines (T1–T10)

Seal OS kernel code is organized around ten theorems (T1–T10), stated in `docs/THEOREMS.md`. The rule for every change, from D5 of [`docs/design/LINUX-REPLACEMENT.md`](docs/design/LINUX-REPLACEMENT.md):

1. Every theorem line the kernel prints reports exactly one of **certified**, **refused** with its reason, or **not checked**.
2. The status is computed from the running kernel's state when the line is printed, never from a string constant, a build flag, or a fixture.
3. A change may turn a line from certified to refused when the certificate was unearned. The commit that does so says so in its message body, naming the theorem and the reason.
4. Refusal is reported, not fatal: the M0 target boots with T4 refused.
5. A change to a theorem statement, proof sketch, or runtime check updates `docs/THEOREMS.md` in the same commit.
6. The headless boot proof prints `[BOOT] Theorems: 2 certified (T1/TSS T2/SCM), 1 not certified (T4/AGCR), 7 not checked (T3/GMC T5/HCS T6/RGCS T7/PHKP T8/TEB T9/CMA T10/WPHB)` and must pass `seal-mkimage --check-theorem-log`, which recomputes each verdict from the evidence on its line and rejects a verdict it cannot recompute.

`kernel/seal-os/src/theorems.rs` computes every line from the running kernel: T1 from the centroid tables the scheduler, compositor, firewall, router and ManifoldFS indexes were built from, at the scheduler governor's live epsilon; T2 from the gains of the running spectral contraction operators, measured through aether-core's own `apply`; T4 from the gains and step every runtime governor uses. T3, T5 and T6-T10 have no running instance that carries their parameters and read `not checked`. `seal-mkimage --check-runtime-theorems` fails a tree whose theorem module reads a literal where a running instance's parameter belongs.

## RED test first

Every behavioural change starts from a test that fails without it.

1. Write the test and run it on the parent commit. It fails, and for the reason it names. A setup failure (exit 2, a missing binary, a VM that never booted) is not RED.
2. Make the change. The same test, unmodified, passes.
3. The commit message or PR quotes both results: the failing assertion or serial line, then the passing run.

A test never seen failing is not a gate. For milestone work the test that ticks a box in [`FUTURE_PLAN.md`](FUTURE_PLAN.md) Phase 0 is a QEMU gate (D6); source-inspection tests are pre-checks. Ports follow the same order ([`PORTING.md`](PORTING.md)).

## Linux-parity gates

[`tests/linux_parity/`](tests/linux_parity/) pins each row of the starting-point table in [`docs/design/LINUX-REPLACEMENT.md`](docs/design/LINUX-REPLACEMENT.md). Each finding is bound to a test that fails on `9ebbe2e`, the commit the plan was measured at; tests named `*_control` pass there, proving the failures are property results rather than a broken setup. A change turns specific findings green; none is deleted or weakened to get there. The gate for each milestone item is listed in [`FUTURE_PLAN.md`](FUTURE_PLAN.md) Phase 0.

Host tests (Python 3.11+ with pytest):

```bash
python -m pytest tests/linux_parity -q
```

Most files inspect kernel source and need nothing else. `test_foreman_userland.py` boots the built image under QEMU and `test_foreman_image_userland.py` runs the built `seal-mkimage`; both need the image built first, and `tests/linux_parity/conftest.py` looks for QEMU, OVMF and `seal-mkimage.exe` at Windows paths.

QEMU gates need the image built first:

```bash
(cd kernel/seal-os && cargo +nightly build --release)
(cd kernel/seal-mkimage && cargo +stable run --release)
```

`chase_boot.sh` (modes `usermode`, `usermode-seal`, `ext4`, `wx`) runs on a Linux host as is. On Windows it runs from Git Bash and reaches `gcc`, `mke2fs` and `e2fsck` through WSL:

```bash
MSYS2_ARG_CONV_EXCL="PATH=" LINUX="wsl -e env PATH=/usr/sbin:/usr/bin" \
QEMU="/c/Program Files/qemu/qemu-system-x86_64.exe" \
OVMF="C:/Program Files/qemu/share/edk2-x86_64-code.fd" \
bash tests/linux_parity/chase_boot.sh usermode-seal
```

`MSYS2_ARG_CONV_EXCL="PATH="` stops Git Bash from rewriting the `PATH=/usr/sbin:/usr/bin` argument into a Windows path before WSL receives it.

`cameron_qemu_milestone.sh` (cases `ring3-seal`, `ring3-linux`, `ring3-glibc`, `ahci-root`, `virtio-root`, `ext4-root`) runs from a WSL shell at the repository root, with `gcc` and `mke2fs` installed in WSL and the Windows QEMU build reached through interop. It converts paths with `wslpath`, so it does not run on a plain Linux host:

```bash
bash tests/linux_parity/cameron_qemu_milestone.sh ring3-seal
```

Both scripts exit 0 when the property holds, 1 when it is violated (RED), and 2 on a setup failure. Exit 2 is no verdict and is never reported as RED or as a pass.

## Linux experience is welcome

People who know Linux internals, have read the Linux source, or maintain Linux code are welcome in every part of the tree. No clean-room restriction applies, because Linux code runs only inside separate driver-server programs (D3): LKL built from a pinned upstream tree, in user mode, reaching the kernel over virtio-style rings. The MIT kernel does not reimplement the in-kernel Linux driver API (D3 rejects that shim), so there is no derived copy of `include/linux` to keep clean. Linux source text is not copied into `kernel/`; implementing the documented userspace ABI (D1) is the work itself. Code ported from any upstream follows [`PORTING.md`](PORTING.md).

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
