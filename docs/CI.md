# CI Pipeline Reference

> Source: `.github/workflows/ci.yml`

Triggered on: all `push` and `pull_request` events, plus manual `workflow_dispatch`
for the native Ubuntu 26.04 comparison lane.
Global env: `CARGO_TERM_COLOR=always`, `RUSTFLAGS="-D warnings"`, `EXPECTED_SEAL_OS_VERSION="0.4.5"` (kernel jobs override RUSTFLAGS to `""`).

> 💡 **Run these checks locally before pushing:** see [`docs/LOCAL_CI.md`](LOCAL_CI.md).
> To use the `git push-ci` alias, run: `git config --local include.path ../.gitconfig.local`

Current workflow jobs: `fmt`, `build`, `clippy`, `test`, `bench-compile`,
`miri`, `bench-regression`, `audit`, `deny`, `docs`, `bom`, `kernel-build`,
`kernel-image`, `kernel-qemu-smoke`, `ubuntu-alloc-baseline`, `kernel-clippy`,
`laamba-governor-check`, and `lean`.

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
| Extra Gates | `cargo +stable check --manifest-path kernel/epsilon/epsilon/crates/aether-core/Cargo.toml --no-default-features --features no_std`; `cargo +stable test --manifest-path kernel/epsilon/epsilon/crates/aether-core/Cargo.toml --test legacy_migration_gate`; `seal-mkimage --check-doc-claim-contract .` |
| What It Checks | All unit and integration tests pass, Aether core still compiles in the bare-metal no_std feature path, deleted legacy theorem modules have Rust public API coverage, and README/docs claims keep Ubuntu/O(1)/ManifoldFS assertions tied to proof gates |
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

## Removed: Host-Language CI

Seal OS GitHub-facing gates are Rust, assembly, and Aether-Lang DSL. Host
research-script compile/test jobs are removed from the OS CI path.

## Job 12: `kernel-build`

| Field | Value |
|-------|-------|
| Display Name | `Build Seal OS kernel (nightly)` |
| Runner | `ubuntu-latest` |
| Toolchain | `dtolnay/rust-toolchain@nightly` + `rust-src`, `llvm-tools-preview` |
| Env | `RUSTFLAGS=""` (overrides global `-D warnings` for nightly) |
| Commands | `cargo +nightly build --release --features test-mode`, then `cargo +nightly build --release` so the shipped EFI is the normal kernel |
| Size Check | Kernel binary must exist at `target/x86_64-unknown-uefi/release/seal-os.efi`; CI prints size but does not enforce a ceiling |
| What It Checks | Kernel compiles for bare-metal x86_64 target, the in-kernel test-mode path compiles, and the release build emits a PE/COFF EFI binary |
| Failure Means | Kernel build failure or missing/invalid EFI binary |

## Job 13: `kernel-image`

| Field | Value |
|-------|-------|
| Display Name | `Build Seal OS UEFI Image` |
| Runner | `ubuntu-latest` |
| Depends On | `kernel-build` |
| Toolchain | `dtolnay/rust-toolchain@nightly` + `rust-src`, `llvm-tools-preview` |
| Commands | Build kernel, then run `cargo +stable run --release` in `kernel/seal-mkimage` to produce `seal-os.img` |
| Size Check | CI prints image size; `seal-mkimage` currently creates a 128MB raw GPT image |
| Verification | Checks UEFI PE/COFF header in `.efi`, GPT layout in `.img` |
| Artifact | `seal-os-img-ci` (seal-os.img) |
| What It Checks | Bootable UEFI disk image can be created from the kernel binary |
| Failure Means | `seal-mkimage` failed, missing PE/COFF header, or invalid GPT/ESP layout |

## Job 14: `kernel-qemu-smoke`

| Field | Value |
|-------|-------|
| Display Name | `QEMU boot smoke test` |
| Runner | `ubuntu-latest` |
| Depends On | `kernel-image` |
| APT Packages | `qemu-system-x86`, `ovmf` |
| Command | `timeout 240 qemu-system-x86_64 -machine q35 -drive if=pflash,format=raw,readonly=on,file=/usr/share/OVMF/OVMF_CODE_4M.fd -device ahci,id=seal_sata -drive if=none,id=seal_disk,file=seal-os.img,format=raw,media=disk -device ide-hd,drive=seal_disk,bus=seal_sata.0,unit=0 -nographic -m 4G -no-reboot -no-shutdown` |
| Artifact | `seal-os-boot-log` (/tmp/seal-os.log) |
| What It Checks | Kernel boots successfully in QEMU and reaches hard boot gates; soft milestones are reported separately |
| Failure Means | Kernel fails to boot, misses a hard gate, or reports a panic/fault/watchdog/theorem failure |

### QEMU Hard Gates

The smoke test must verify these serial output patterns:

| # | Pattern | Milestone |
|---|---------|-----------|
| 1 | `Seal OS v0.4.5` | Exact release banner printed |
| 2 | `Heap initialized` | Heap initialized |
| 3 | `IDT + PIC initialized` | Interrupts configured |
| 4 | `SYSCALL/SYSRET MSRs programmed` | Syscall entry configured |
| 5 | `Governor online` | T4 governor started |
| 6 | `Voronoi index: 8 cells` | T1 Voronoi active |
| 7 | `[THEOREM] T1/TSS VERIFIED` through `[THEOREM] T10/WPHB VERIFIED` | Every theorem line verified |
| 8 | `All T1-T10 theorems VERIFIED` | Theorem summary verified |
| 9 | `[BENCH] toporam-alloc` | Hint-biased TopoRAM target-cell hot path measured |
| 10 | `[BENCH] alloc-frame` | Single-frame allocator hot path measured |
| 11 | `[BENCH] manifold-teleport` | ManifoldFS same-inode metadata teleport measured, with write-through persistence explicitly accounted |
| 12 | `[BENCH] scheduler-select-next` | Live scheduler selector requeue hot path measured |
| 13 | `[Aether-Lang] runtime proof` | Parser, interpreter, and app host executed the boot probe |
| 14 | `Device model: QEMU HARDDISK` | AHCI disk identified |
| 15 | `Registered as block device 0x800` | AHCI block device registered |
| 16 | `ManifoldFS mounted from disk` | Persistent ManifoldFS root mounted |
| 17 | `Desktop proof frame blit done` | First desktop frame blitted |
| 18 | `[GFX] desktop-soak` | Deterministic compositor compose+blit exercise ran |
| 19 | `Seal OS desktop ready.` | Desktop reached |
| 20 | `Entering real event loop` | Event loop active |

The job must fail if any theorem line contains `FAILED`, or if panic/fault/watchdog markers appear.

Additional Rust-native audit commands run against the same OS surface:

| Command | Purpose |
|---|---|
| `seal-mkimage --check-theorem-log /tmp/seal-os.log` | Requires all T1-T10 theorem lines, desktop proof frame blit, desktop soak marker, desktop readiness, event loop entry, and rejects panic/fault/watchdog fatal markers |
| `seal-mkimage --check-vm-proof /tmp/seal-os.log` | Requires theorem proof, QEMU AHCI disk identity, block device `0x800`, readable disk, persistent ManifoldFS root, desktop frame, desktop soak marker, desktop ready, and event loop; rejects ramfs fallback and missing AHCI |
| `seal-mkimage --check-aether-runtime /tmp/seal-os.log` | Requires the Aether runtime boot marker proving parser, interpreter, and app host executed `aether_boot_probe` inside the kernel runtime |
| `seal-mkimage --check-desktop-soak /tmp/seal-os.log` | Parses the `[GFX] desktop-soak` marker and requires frame count, monotonic cycle percentiles, input-events field, and full-frame dirty-pixel coverage |
| `seal-mkimage --check-benchmark-log /tmp/seal-os.log` | Parses `[BENCH] toporam-alloc`, `[BENCH] alloc-frame`, `[BENCH] manifold-teleport`, and `[BENCH] scheduler-select-next`; requires TopoRAM target-cell hits with zero fallbacks, physical allocator fast-path invariants, same-inode ManifoldFS metadata teleport across 8-256 entries, bounded metadata ops, explicit write-through byte accounting, and live scheduler select requeue with fixed 8-cell/256-bucket bounds and zero context switches |
| `seal-mkimage --check-proof-manifest <proof-manifest.txt>` | Verifies proof bundle provenance. QEMU manifests require image/EFI/log/screen byte counts, CRC32/SHA-256 fingerprints, backend, commit/dirty flag, proof-screen status, and hard gate statuses. VirtualBox manifests require image/EFI/log/screenshot fingerprints, headless backend, commit/dirty flag, vbox proof status, and hard gate statuses without claiming QEMU PPM parity |
| `seal-mkimage --check-ubuntu-benchmark-log ubuntu-alloc.log` | Parses a same-machine `[UBUNTU-BENCH] alloc-frame` artifact and rejects it unless `os=ubuntu`, `version_id=26.04`, `kernel=` is native rather than WSL, iterations are complete, bytes are 4096, and cycle percentiles are monotonic |
| `seal-mkimage --compare-benchmark-logs /tmp/seal-os.log ubuntu-alloc.log` | Fails unless Seal OS p50, p95, and max allocation cycles beat the validated Ubuntu artifact |
| `seal-mkimage --check-current-benchmark-proof qemu-proof/proof-manifest.txt ubuntu-alloc.log .` | Fails unless the Seal allocator comparison is bound to the serial log inside a current VM proof manifest whose commit and dirty flag match the checkout; this is allocator-only evidence, not a global Ubuntu win |
| `seal-mkimage --check-seal-abi .` | Scans `kernel/seal-os/src` for banned ABI imports and calls such as `extern crate libc`, `use libc`, `std::os::unix`, `std::os::linux`, `posix_spawn`, `pthread_`, `forkpty`, and `termios` |
| `seal-mkimage --check-language-hygiene .` | Keeps the public OS CI/docs surface on Rust, assembly, and Aether-Lang; also requires `.gitattributes` to mark scripts, JS/TS app scaffolding, Dockerfiles, requirements text, configs, assets, docs, and generated proof output as vendored/generated/documentation, and requires `.gitignore` to ignore VM/proof/trace/profile artifacts |
| `seal-mkimage --check-doc-claim-contract .` | Fails if README/docs drop the guardrails that tie Ubuntu wins, O(1) allocation, ManifoldFS teleport, Aether runtime, and benchmark claims to hard audit gates or recorded artifacts |
| `seal-mkimage --check-aether-migration .` | Requires Rust replacement symbols for deleted epsilon theorem modules, proves the deleted legacy theorem files are absent, requires restored Rust APIs in `legacy_migration_gate`, and rejects stale root or legacy imports of deleted modules |
| `seal-mkimage --check-o1-allocator .` | Requires physical and TopoRAM allocation paths to use bounded topological probes, interval-gated entropy sampling, bounded reseed refresh, and no RAM-wide runtime bitmap/frame scans |
| `seal-mkimage --check-runtime-theorems .` | Source-gates T1-T5 runtime topology callsites in memory, scheduler, ManifoldFS, compositor, ACPI power, taskbar status, and boot theorem state |
| `seal-mkimage --unsafe-inventory .` | Reports unsafe Rust totals and hotspots for audit review |

The Ubuntu benchmark gates are artifact-dependent. CI can compile
`tools/ubuntu-alloc-bench` as part of the Rust workspace, but it must not claim
an Ubuntu win unless a real Ubuntu runner emits the `ubuntu-alloc.log` artifact
for the same benchmark environment.

The `ubuntu-alloc-baseline` job is manual-only (`workflow_dispatch`) and requires
a self-hosted runner labeled `ubuntu-26.04`. It rejects non-Ubuntu, stale Ubuntu,
and WSL kernels, boots the Seal image on that same runner, captures
`ubuntu-alloc.log`, runs `--check-benchmark-log`, `--check-ubuntu-benchmark-log`,
`--compare-benchmark-logs`, and, when a proof bundle is available for the same
checkout, `--check-current-benchmark-proof`, then uploads the Seal log, Ubuntu
log, and host manifest as `ubuntu-26-alloc-comparison`.

The pixel and manifest gates, `seal-mkimage --check-proof-screen
qemu-proof/screen.ppm` and `seal-mkimage --check-proof-manifest
qemu-proof/proof-manifest.txt`, are local WSL2/QEMU proof gates because the CI
smoke job runs with `-nographic` and does not capture a framebuffer dump. They
must pass before claiming a GUI desktop proof.

### QEMU Soft Milestones

These are useful observations, but not hard proof unless CI explicitly requires them:

| Pattern | Meaning |
|---|---|
| `ManifoldFS` | Filesystem initialized or ramfs fallback mounted |
| `Scheduler` | Scheduler initialized |
| `Syscall` | Syscall filesystem initialized |
| `Shell` | Shell or desktop control surface loaded |
| `Seal OS desktop ready` | GUI path reached ready sentinel |

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
| What It Checks | The Lean package compiles; placeholder/proof closure is tracked separately in `docs/THEOREMS.md` |
| Failure Means | The Lean package no longer builds |

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
kernel-clippy ───────┤
lean ────────────────┘

kernel-build ──> kernel-image ──> kernel-qemu-smoke
```

The kernel pipeline is sequential: kernel build produces the UEFI binary, the disk image job packages `seal-os.img`, and QEMU boots that disk image. All other jobs are independent and run in parallel.
