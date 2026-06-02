# Local CI System

This repository includes a **local CI runner** that mirrors the checks run in `.github/workflows/ci.yml` so you can catch failures before pushing.

---

## Quick Start

### 1. Enable Git hooks

```bash
git config core.hooksPath .githooks
```

Once enabled, every `git push` will automatically run the full local CI suite. If any check fails, the push is blocked.

### 2. Push with CI (alias)

If you prefer not to use hooks, use the provided alias:

```bash
git config --local include.path ../.gitconfig.local
git push-ci
```

This runs the same checks and only pushes on success.

### 3. Run CI manually

```bash
./scripts/local_ci.sh
```

---

## Bypassing Checks

If you need to push urgently without running CI (e.g., to share a WIP branch):

```bash
git push --no-verify
```

> ⚠️ Only use `--no-verify` for temporary branches. All code merged to `main` must pass CI.

---

## What Checks Run

| Step | Command |
|------|---------|
| Format | `cargo fmt --all -- --check` |
| Lints | `cargo clippy --workspace --all-targets -- -D warnings` |
| Tests | `cargo test --workspace` |
| Aether no_std check | `cargo +stable check --manifest-path kernel/epsilon/epsilon/crates/aether-core/Cargo.toml --no-default-features --features no_std` |
| Aether migration gate | `cargo +stable test --manifest-path kernel/epsilon/epsilon/crates/aether-core/Cargo.toml --test legacy_migration_gate` |
| Kernel test-mode build | `cd kernel/seal-os && cargo +nightly build --release --features test-mode` |
| Kernel build | `cd kernel/seal-os && cargo +nightly build --release` |
| Image build | `cd kernel/seal-mkimage && cargo +stable run --release` |
| Image verify | `cargo +stable run --manifest-path kernel/seal-mkimage/Cargo.toml --release -- --verify kernel/seal-os/target/x86_64-unknown-uefi/release/seal-os.img kernel/seal-os/target/x86_64-unknown-uefi/release/seal-os.efi` |
| Seal ABI audit | `cargo +stable run --manifest-path kernel/seal-mkimage/Cargo.toml --release -- --check-seal-abi .` |
| Language hygiene audit | `cargo +stable run --manifest-path kernel/seal-mkimage/Cargo.toml --release -- --check-language-hygiene .` |
| Doc claim contract audit | `cargo +stable run --manifest-path kernel/seal-mkimage/Cargo.toml --release -- --check-doc-claim-contract .` |
| Release workflow contract audit | `cargo +stable run --manifest-path kernel/seal-mkimage/Cargo.toml --release -- --check-release-workflow-contract .` |
| Aether migration audit | `cargo +stable run --manifest-path kernel/seal-mkimage/Cargo.toml --release -- --check-aether-migration .` |
| O(1) allocator audit | `cargo +stable run --manifest-path kernel/seal-mkimage/Cargo.toml --release -- --check-o1-allocator .` |
| Runtime theorem coverage audit | `cargo +stable run --manifest-path kernel/seal-mkimage/Cargo.toml --release -- --check-runtime-theorems .` |
| Unsafe inventory | `cargo +stable run --manifest-path kernel/seal-mkimage/Cargo.toml --release -- --unsafe-inventory .` |
| Ubuntu allocator harness build | `cargo +stable build --manifest-path tools/ubuntu-alloc-bench/Cargo.toml --release` |
| VM proof audit | If `qemu-proof/serial.log` exists, `cargo +stable run --manifest-path kernel/seal-mkimage/Cargo.toml --release -- --check-vm-proof kernel/seal-os/target/x86_64-unknown-uefi/release/qemu-proof/serial.log` |
| Aether runtime audit | If `qemu-proof/serial.log` exists, `cargo +stable run --manifest-path kernel/seal-mkimage/Cargo.toml --release -- --check-aether-runtime kernel/seal-os/target/x86_64-unknown-uefi/release/qemu-proof/serial.log` |
| Desktop soak audit | If `qemu-proof/serial.log` exists, `cargo +stable run --manifest-path kernel/seal-mkimage/Cargo.toml --release -- --check-desktop-soak kernel/seal-os/target/x86_64-unknown-uefi/release/qemu-proof/serial.log` |
| Benchmark log audit | If `qemu-proof/serial.log` exists, `cargo +stable run --manifest-path kernel/seal-mkimage/Cargo.toml --release -- --check-benchmark-log kernel/seal-os/target/x86_64-unknown-uefi/release/qemu-proof/serial.log` |
| Ubuntu benchmark audit | If `tools/ubuntu-alloc-bench/ubuntu-alloc.log` exists, `cargo +stable run --manifest-path kernel/seal-mkimage/Cargo.toml --release -- --check-ubuntu-benchmark-log tools/ubuntu-alloc-bench/ubuntu-alloc.log` |
| Seal vs Ubuntu allocator comparison | If both logs exist, `cargo +stable run --manifest-path kernel/seal-mkimage/Cargo.toml --release -- --compare-benchmark-logs kernel/seal-os/target/x86_64-unknown-uefi/release/qemu-proof/serial.log tools/ubuntu-alloc-bench/ubuntu-alloc.log` |
| BOM check | `./scripts/check_no_bom.sh` |
| Docs | `cargo doc --workspace --no-deps` |

The VM proof audit is mandatory when a captured serial proof exists. It requires
the theorem gate, QEMU AHCI disk identity, block device `0x800`, readable disk,
persistent ManifoldFS root, TopoRAM allocation benchmark marker, physical
allocator benchmark marker, ManifoldFS teleport benchmark marker, scheduler
select benchmark marker, Aether runtime proof marker, desktop proof frame,
desktop live input marker, desktop soak marker, desktop readiness, and event
loop entry. The GUI pixel proof is manual/local because it needs a captured
`kernel/seal-os/target/x86_64-unknown-uefi/release/qemu-proof/screen.ppm` from
`run-qemu.ps1 -HeadlessProof`. After that capture, run
`cargo +stable run --manifest-path kernel/seal-mkimage/Cargo.toml --release -- --check-proof-screen kernel/seal-os/target/x86_64-unknown-uefi/release/qemu-proof/screen.ppm`.

The Ubuntu allocator audit is intentionally optional locally. It becomes a real
comparison only when `tools/ubuntu-alloc-bench/ubuntu-alloc.log` was captured on
native Ubuntu 26.04 LTS; the checker rejects non-Ubuntu, stale-Ubuntu, and WSL
kernel output.

Each step is timed and color-coded:
- 🟢 **PASS** — green
- 🔴 **FAIL** — red (stops the pipeline immediately)

---

## Files

| File | Purpose |
|------|---------|
| `scripts/local_ci.sh` | Main CI runner |
| `scripts/push_with_ci.sh` | Wrapper that runs CI then `git push` |
| `.githooks/pre-push` | Hook that blocks push on CI failure |
| `.githooks/post-push` | Hook that confirms successful push |
| `.gitconfig.local` | Git alias `push-ci` |
| `docs/LOCAL_CI.md` | This documentation |
