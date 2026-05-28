# Generated Files Policy

Seal OS keeps source, specs, lockfiles, scripts, and small reproducible metadata in git.
Seal OS ignores build products, VM images, firmware blobs, debug symbols, logs,
screenshots, caches, temporary reports, and raw benchmark output.

This repo is building a bare-metal OS. Generated files can become huge fast.
Big artifacts belong in CI artifacts or releases, not in normal source commits.

## Checked In

- Rust source, Aether-Lang source, Lean proof source, and documentation.
- `Cargo.toml` and `Cargo.lock` files needed for reproducible kernel, tool, and app builds.
- Local CI, QEMU, VirtualBox, image creation, and image verification scripts.
- Small generated API/schema files only when a build needs them and regeneration would require an extra host toolchain.
- Benchmark harness source and summarized benchmark reports.

## Ignored

- Rust, Node, Tauri, and documentation build directories.
- OS and VM artifacts: `*.img`, `*.iso`, `*.vdi`, `*.vmdk`, `*.qcow2`, `*.efi`.
- Firmware blobs and local firmware copies: `*.fd`, `*.ovmf`.
- Debug symbols and linker byproducts: `*.pdb`, `*.map`, `*.sym`, `*.dSYM/`.
- Serial logs, screenshots, local smoke-test output, and raw benchmark output.
- ML model weights, tensor dumps, local databases, caches, and temporary reports.

## Seal OS Regeneration Commands

From the repository root:

```powershell
cargo +nightly build --manifest-path kernel\seal-os\Cargo.toml --release
cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml --release
```

From `kernel\seal-os`, build the VirtualBox disk:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\build-vbox.ps1
```

Verify a generated disk image:

```powershell
cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml --release -- --verify kernel\seal-os\target\x86_64-unknown-uefi\release\seal-os.img kernel\seal-os\target\x86_64-unknown-uefi\release\seal-os.efi
```

Run the VirtualBox smoke test from `kernel\seal-os`:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\smoke-vbox.ps1 -Seconds 240
```

## LAAMBA Governor Schemas

`apps/laamba-governor/src-tauri/gen/schemas/*.json` are small generated schema
files. Keep them in git only if the LAAMBA Governor build needs them without a
host-side schema regeneration step. If the app can regenerate them reliably,
move them under the ignored generated-output policy and document the regeneration
command here.

## Benchmark Artifacts

Raw benchmark output stays out of git. Commit only:

- the benchmark harness
- exact commands
- host and guest configuration
- summarized tables
- short logs needed to reproduce a claim

Claims that Seal OS surpasses Ubuntu are valid only for benchmark rows with
fresh Seal OS and Ubuntu measurements under the same constraints.
