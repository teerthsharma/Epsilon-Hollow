# CI Pipeline Reference

> Source: `.github/workflows/ci.yml`

Triggered on: all `push` and `pull_request` events.  
Global env: `CARGO_TERM_COLOR=always`, `RUSTFLAGS="-D warnings"` (kernel jobs override RUSTFLAGS to `""`).

---

## Job 1: `fmt`

| Field | Value |
|-------|-------|
| Display Name | `cargo fmt --workspace` |
| Runner | `ubuntu-latest` |
| Toolchain | `dtolnay/rust-toolchain@1.85` + `rustfmt` |
| Command | `cargo fmt --all -- --check` |
| What It Checks | All Rust source files are formatted per `rustfmt` defaults |
| Failure Means | Unformatted code exists; run `cargo fmt` locally |

## Job 2: `clippy`

| Field | Value |
|-------|-------|
| Display Name | `cargo clippy --workspace` |
| Runner | `ubuntu-latest` |
| Toolchain | `dtolnay/rust-toolchain@1.85` + `clippy` |
| Cache | `Swatinem/rust-cache@v2` |
| Command | `cargo clippy --workspace --all-targets -- -D warnings` |
| What It Checks | Lint-free workspace (all warnings are errors) |
| Failure Means | Clippy lint violations |

## Job 3: `test`

| Field | Value |
|-------|-------|
| Display Name | `cargo test --workspace` |
| Runner | `ubuntu-latest` |
| Toolchain | `dtolnay/rust-toolchain@1.85` |
| Cache | `Swatinem/rust-cache@v2` |
| Command | `cargo test --workspace` |
| What It Checks | All unit and integration tests pass |
| Failure Means | Test failure in any workspace crate |

## Job 4: `bench-compile`

| Field | Value |
|-------|-------|
| Display Name | `cargo bench --no-run` |
| Runner | `ubuntu-latest` |
| Toolchain | `dtolnay/rust-toolchain@1.85` |
| Cache | `Swatinem/rust-cache@v2` |
| Command | `cargo bench --workspace --no-run` |
| What It Checks | All benchmarks compile without errors |
| Failure Means | Benchmark code has compilation errors |

## Job 5: `miri`

| Field | Value |
|-------|-------|
| Display Name | `miri (UB detection)` |
| Runner | `ubuntu-latest` |
| Toolchain | `dtolnay/rust-toolchain@nightly` + `miri` |
| Cache | `Swatinem/rust-cache@v2` |
| Env | `RUSTUP_TOOLCHAIN=nightly`, `MIRIFLAGS="-Zmiri-disable-isolation"` |
| Commands | `cargo +nightly miri test -p aether-core --features force-soft-floats --lib state::tests`, `...os::tests`, `...proptest_os` |
| What It Checks | No undefined behavior in aether-core state and os modules |
| Failure Means | Miri detected UB (uninitialized memory, data races, etc.) |

## Job 6: `bench-regression`

| Field | Value |
|-------|-------|
| Display Name | `bench regression gate (io_cycle_8_lbas)` |
| Runner | `ubuntu-latest` |
| Toolchain | `dtolnay/rust-toolchain@1.85` |
| Cache | `Swatinem/rust-cache@v2` |
| Command | `cargo bench --bench io_cycle --manifest-path kernel/aether/aether-link/Cargo.toml` |
| Threshold | Median must be <= 120 ns/iter |
| Artifact | `io-cycle-bench` (bench.txt) |
| What It Checks | AETHER-Link I/O cycle latency has not regressed beyond 120ns |
| Failure Means | Performance regression in the I/O prediction kernel |

## Job 7: `audit`

| Field | Value |
|-------|-------|
| Display Name | `cargo audit` |
| Runner | `ubuntu-latest` |
| Toolchain | `dtolnay/rust-toolchain@1.85` |
| Command | `cargo install cargo-audit --locked && cargo audit` |
| What It Checks | No known security vulnerabilities in dependencies |
| Failure Means | A dependency has a published CVE/advisory |

## Job 8: `deny`

| Field | Value |
|-------|-------|
| Display Name | `cargo deny` |
| Runner | `ubuntu-latest` |
| Action | `EmbarkStudios/cargo-deny-action@v2` |
| Command | `check --all-features` |
| What It Checks | License compliance, duplicate crates, banned crates |
| Failure Means | License violation, banned dependency, or duplicate crate |

## Job 9: `docs`

| Field | Value |
|-------|-------|
| Display Name | `cargo doc -D warnings` |
| Runner | `ubuntu-latest` |
| Toolchain | `dtolnay/rust-toolchain@1.85` |
| Env | `RUSTDOCFLAGS="-D warnings"` |
| Command | `cargo doc --workspace --no-deps` |
| What It Checks | All doc comments compile without warnings |
| Failure Means | Broken doc links, missing docs on public items |

## Job 10: `bom`

| Field | Value |
|-------|-------|
| Display Name | `no UTF-8 BOM` |
| Runner | `ubuntu-latest` |
| Command | `./scripts/check_no_bom.sh` |
| What It Checks | No source files contain a UTF-8 Byte Order Mark |
| Failure Means | A file has a BOM prefix (0xEF 0xBB 0xBF) |

## Job 11: `python`

| Field | Value |
|-------|-------|
| Display Name | `python compileall + unit tests` |
| Runner | `ubuntu-latest` |
| Python | 3.11 |
| Dependencies | `numpy` only (torch tests are `skipUnless`) |
| Commands | `python -m compileall -q infrastructure scripts tests kernel/epsilon/epsilon_core`, `python -m unittest discover -s tests -p 'test_*.py' -v` |
| What It Checks | All Python files compile; all Python unit tests pass |
| Failure Means | Python syntax errors or test failures |

## Job 12: `kernel-build`

| Field | Value |
|-------|-------|
| Display Name | `Build Seal OS kernel (nightly)` |
| Runner | `ubuntu-latest` |
| Toolchain | `dtolnay/rust-toolchain@nightly` + `rust-src`, `llvm-tools-preview` |
| Env | `RUSTFLAGS=""` (overrides global `-D warnings` for nightly) |
| Commands | `cargo +nightly build --release` (release), `cargo +nightly build` (debug) |
| Size Gate | Kernel binary must exist at `target/x86_64-unknown-uefi/release/seal-os.efi` and be <= 512KB |
| What It Checks | Kernel compiles for bare-metal x86_64 target; binary size is reasonable |
| Failure Means | Kernel build failure or binary exceeds 512KB |

## Job 13: `kernel-image`

| Field | Value |
|-------|-------|
| Display Name | `Build Seal OS UEFI Image` |
| Runner | `ubuntu-latest` |
| Depends On | `kernel-build` |
| Toolchain | `dtolnay/rust-toolchain@nightly` + `rust-src`, `llvm-tools-preview` |
| Commands | Build kernel, then run `cargo +stable run --release` in `kernel/seal-mkimage` to produce `seal-os.img` |
| Size Gate | Image must be <= 64MB |
| Verification | Checks UEFI PE/COFF header in `.efi`, GPT layout in `.img` |
| Artifact | `seal-os-img-ci` (seal-os.img) |
| What It Checks | Bootable UEFI disk image can be created from the kernel binary |
| Failure Means | `seal-mkimage` failed, missing PE/COFF header, or image too large |

## Job 14: `kernel-qemu-smoke`

| Field | Value |
|-------|-------|
| Display Name | `QEMU boot smoke test` |
| Runner | `ubuntu-latest` |
| Depends On | `kernel-image` |
| APT Packages | `qemu-system-x86`, `ovmf` |
| Command | `timeout 45 qemu-system-x86_64 -machine q35 -drive if=pflash,format=raw,readonly=on,file=/usr/share/OVMF/OVMF_CODE_4M.fd -drive file=seal-os.img,format=raw -nographic -m 4G -no-reboot -no-shutdown` |
| Artifact | `seal-os-boot-log` (/tmp/seal-os.log) |
| What It Checks | Kernel boots successfully in QEMU and reaches all 11 milestones |
| Failure Means | Kernel fails to boot or does not reach expected milestones |

### QEMU 11-Milestone Checklist

The smoke test (lines 303-329) verifies these serial output patterns:

| # | Pattern | Milestone |
|---|---------|-----------|
| 1 | `Seal OS v1.0.0-alpha` | Banner printed |
| 2 | `Heap initialized` | Heap initialized |
| 3 | `IDT + PIC initialized` | Interrupts configured |
| 4 | `Governor online` | T4 governor started |
| 5 | `Voronoi index: 8 cells` | T1 Voronoi active |
| 6 | `T1-T5 theorems ACTIVE` | All theorems active |
| 7 | `ManifoldFS` | ManifoldFS initialized |
| 8 | `Teleported.*O(1)` | O(1) teleportation demonstrated |
| 9 | `Scheduler` | Scheduler started |
| 10 | `Syscall` | Syscalls verified |
| 11 | `Shell` | Shell executed |

All 11 must pass (FAIL count must be 0) or the job fails.

## Job 15: `kernel-clippy`

| Field | Value |
|-------|-------|
| Display Name | `Seal OS clippy (nightly)` |
| Runner | `ubuntu-latest` |
| Toolchain | `dtolnay/rust-toolchain@nightly` + `rust-src`, `llvm-tools-preview`, `clippy` |
| Env | `RUSTFLAGS=""` |
| Command | `cargo +nightly clippy -- -A unused -A dead_code -A clippy::approx_constant` |
| What It Checks | Kernel code passes clippy (with unused/dead_code/approx_constant allowed) |
| Failure Means | Clippy lint violation in kernel code |

## Job 16: `lean`

| Field | Value |
|-------|-------|
| Display Name | `Lean 4 (aether-verified)` |
| Runner | `ubuntu-latest` |
| Action | `leanprover/lean-action@v1` |
| Package Directory | `kernel/aether/aether-verified/lean` |
| Cache Key | `lean-{os}-v4.7.0-{hash of lakefile.lean + lean-toolchain}` |
| Mathlib | `use-mathlib-cache: true` |
| What It Checks | All Lean 4 proofs in aether-verified compile (no `sorry`) |
| Failure Means | A Lean proof is broken, incomplete, or uses `sorry` |

---

## Pipeline Dependency Graph

```
fmt ─────────────────┐
clippy ──────────────┤
test ────────────────┤
bench-compile ───────┤
miri ────────────────┤
bench-regression ────┤
audit ───────────────┤  (all independent, run in parallel)
deny ────────────────┤
docs ────────────────┤
bom ─────────────────┤
python ──────────────┤
kernel-clippy ───────┤
lean ────────────────┘

kernel-build ──> kernel-image ──> kernel-qemu-smoke
```

The kernel pipeline is sequential: build depends on nothing, ISO depends on build, QEMU depends on ISO. All other jobs are independent and run in parallel.
