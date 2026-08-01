# Build and Verification

The repository has two primary Cargo flows. Run the flow that matches the
component you changed.

## Prerequisites

| Flow | Toolchain | Target |
| --- | --- | --- |
| Root workspace | Stable, pinned by root `rust-toolchain.toml` | Host target |
| Seal OS | Nightly, `rust-src`, `llvm-tools-preview` | Bare-metal x86_64 UEFI |
| Lean proofs | Lean toolchain in `kernel/aether/aether-verified/lean` | Host proof checker |
| QEMU proof | Host `qemu-system-x86_64`, OVMF, and serial tooling | UEFI VM |

The kernel has a separate `rust-toolchain.toml`; do not assume a successful
root build means the kernel target is installed.

## Root workspace checks

Run from the repository root:

```powershell
cargo build --workspace
cargo test --workspace
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo doc --workspace --no-deps
```

Additional configured lanes include no-std `aether-core` compilation, legacy
migration coverage, Miri for selected modules, benchmarks, `cargo audit`, and
`cargo deny`. The full CI command matrix is in [`CI.md`](CI.md).

## Kernel build

The kernel is not a root workspace member. From `kernel/seal-os` use the pinned
nightly toolchain and the target expected by the repository scripts:

```powershell
cargo +nightly build --release
cargo +nightly build --release --features test-mode
```

The exact target and output layout may be selected by the repository build
scripts or CI environment. The expected release artifact is
`target/x86_64-unknown-uefi/release/seal-os.efi` when the standard UEFI lane is
used.

## Image construction

`seal-mkimage` consumes the kernel EFI artifact and creates a raw GPT image
with an EFI system partition and the configured data partition layout. The
image builder is a host tool and uses stable Rust:

```powershell
cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml --release
```

Before a VM run, verify the image and EFI payload with the image tool's
verification command or the corresponding script in `scripts/`. Image validity
is necessary but does not prove a successful kernel boot.

## Proof gates

The audit CLI in `kernel/seal-mkimage` owns repository checks such as:

```text
--check-doc-claim-contract
--check-theorem-log
--check-lean-proof-hygiene
--check-aether-runtime
--check-vm-proof
--check-vbox-proof
--check-proof-manifest
--check-ubuntu-benchmark-log
--compare-benchmark-logs
```

Use `--help` on the current binary for the exact arguments. The names are
listed here because the documentation must not invent a second verifier.

## QEMU proof path

The canonical helper is `kernel/seal-os/run-qemu.ps1` on Windows and
`kernel/seal-os/run-qemu.sh` on Linux/macOS. The CI smoke path builds an image,
boots QEMU with UEFI and AHCI, captures serial output, and runs the proof
checker. The captured-pixel lane additionally records a PPM proof screen.

Hard proof categories include:

- exact version banner and early memory/interrupt/syscall markers;
- all T1-T10 theorem lines and the theorem summary;
- allocator, scheduler, filesystem, networking, TLS, and renderer markers;
- AHCI identity, block registration, sector-zero readability, and persistent
  ManifoldFS mounting;
- Aether runtime proof, desktop frame/input/soak/ready markers, and event-loop
  entry;
- absence of fatal boot markers such as panic, fault, watchdog, or theorem
  failure.

The current contract is maintained in [`VM_RUNBOOK.md`](VM_RUNBOOK.md) and
`docs/CI.md`. If source behavior changes, update the marker-producing code,
the Rust checker, and the runbook as one change.

## Local proof discipline

1. Build the exact artifact you intend to boot.
2. Record the commit, toolchain, target, image path, and command.
3. Capture serial output and visual output from the same image.
4. Run the image verifier before the VM.
5. Run the VM proof checker against the fresh serial log and manifest.
6. Report unsupported hardware or roadmap claims as unsupported; do not infer
   them from a QEMU fixture.

## Common false positives

| Looks like success | Why it is insufficient |
| --- | --- |
| `cargo build --workspace` passes | The kernel is excluded from the root workspace |
| EFI file exists | The image may have a bad GPT/ESP layout or stale payload |
| QEMU prints a banner | Later memory, theorem, storage, or desktop gates may fail |
| A screenshot looks correct | Pixels do not prove serial milestones or persistence |
| A theorem line says `VERIFIED` | Formal Lean strength and runtime activation are tracked separately |
| A benchmark is fast on Windows/WSL | It is not the required native Ubuntu comparison artifact |

## Documentation build

The root site is built from `mkdocs.yml` and `docs/`:

```powershell
mkdocs build --strict
```

The nested Aether site is built from
`kernel/aether/Aether-Lang/mkdocs.yml` and has its own dependency setup. Build
the site you changed; do not treat one site’s successful build as validation of
the other.
