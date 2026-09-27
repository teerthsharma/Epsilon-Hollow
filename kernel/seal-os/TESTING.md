# Seal OS Testing Guide

This document describes the current supported proof path for Seal OS 0.4.7.5.
The kernel crate is a UEFI `no_std` target, so raw host unit tests are not the
kernel proof. The proof is: build the UEFI kernel, build the UEFI test-mode
variant, package the image, boot it in a VM, then run the Rust audit gates.

## Prerequisites

- Rust nightly with `rust-src`
- Rust stable for `seal-mkimage`
- QEMU with OVMF, or Oracle VM VirtualBox for GUI smoke
- PowerShell on Windows for the current WSL2 proof runner

## Build Gates

From `kernel\seal-os`:

```powershell
cargo +nightly build --release --features test-mode
cargo +nightly build --release
```

Expected artifact:

```text
target\x86_64-unknown-uefi\release\seal-os.efi
```

The `test-mode` build compiles the in-kernel test runner for the UEFI target.
Run the normal release build after it so `seal-os.efi` is the shipped kernel.
It is a compilation gate, not a host `cargo test` substitute.

## Image Build

From `kernel\seal-mkimage`:

```powershell
cargo +stable run --release
```

Expected artifacts:

```text
kernel\seal-os\target\x86_64-unknown-uefi\release\seal-os.img
kernel\seal-os\target\x86_64-unknown-uefi\release\seal-os-efi.img
```

## WSL2 QEMU Proof

From `kernel\seal-os`:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run-qemu.ps1 -HeadlessProof -ProofSeconds 240
```

Expected proof artifacts:

```text
target\x86_64-unknown-uefi\release\qemu-proof\serial.log
target\x86_64-unknown-uefi\release\qemu-proof\screen.ppm
target\x86_64-unknown-uefi\release\qemu-proof\screen.png
target\x86_64-unknown-uefi\release\qemu-proof\proof-manifest.txt
```

## Required Serial Markers

The log must contain:

```text
[THEOREM] T1/TSS CERTIFIED: eps=0.1000 theta_min=0.1000 cells=8 p_max=1600.0 min_sep=0.7854 covers=scheduler+compositor+firewall+route,manifoldfs
[THEOREM] T2/SCM CERTIFIED: operators=manifoldfs:0.70,firewall:0.30,route:0.30 max_lip=0.70 max_ratio=0.7000 unread=scheduler
[THEOREM] T3/GMC NOT CHECKED: <reason>
[THEOREM] T4/AGCR NOT CERTIFIED: alpha+beta/dt=5.01 >= 1 at dt=0.01
[THEOREM] T5/HCS NOT CHECKED: <reason>
[THEOREM] T6/RGCS NOT CHECKED: no kernel subsystem runs it
[THEOREM] T7/PHKP NOT CHECKED: no kernel subsystem runs it
[THEOREM] T8/TEB NOT CHECKED: no kernel subsystem runs it
[THEOREM] T9/CMA NOT CHECKED: no kernel subsystem runs it
[THEOREM] T10/WPHB NOT CHECKED: no kernel subsystem runs it
[BOOT] Theorems: 2 certified (T1/TSS T2/SCM), 1 not certified (T4/AGCR), 7 not checked (T3/GMC T5/HCS T6/RGCS T7/PHKP T8/TEB T9/CMA T10/WPHB)
[BOOT] Desktop proof frame blit done
[BOOT] Seal OS desktop ready.
[EVENT] Entering real event loop
```

The log must not contain panic, fault, watchdog, or QEMU fatal markers.
Every theorem line is computed by `kernel/seal-os/src/theorems.rs` from the
running kernel: the scheduler governor's epsilon, the centroid tables the
running Voronoi indexes were built from, and the gains of the running
spectral contraction operators. `--check-theorem-log` recomputes each verdict
from the evidence on its line: T1's `eps` must equal the
`[T4/AGCR] Governor online:` epsilon, `theta_min` must be `2 asin(eps/2)` and
`min_sep` must exceed it; every T2 gain must lie in (0, 1]; T4 is certified
exactly when `alpha + beta/dt < 1` at the governor line's gains and step. T3,
T5 and T6-T10 have no running instance and must read `NOT CHECKED`; a bare
`VERIFIED` line, or any verdict the gate cannot recompute, fails.

## Rust Audit Gates

From the repository root:

```powershell
cargo +stable test --workspace
cargo +stable check --manifest-path kernel\epsilon\epsilon\crates\aether-core\Cargo.toml --no-default-features --features no_std
cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml --release -- --verify kernel\seal-os\target\x86_64-unknown-uefi\release\seal-os.img kernel\seal-os\target\x86_64-unknown-uefi\release\seal-os.efi
cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml --release -- --check-theorem-log kernel\seal-os\target\x86_64-unknown-uefi\release\qemu-proof\serial.log
cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml --release -- --check-proof-screen kernel\seal-os\target\x86_64-unknown-uefi\release\qemu-proof\screen.ppm
cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml --release -- --check-proof-manifest kernel\seal-os\target\x86_64-unknown-uefi\release\qemu-proof\proof-manifest.txt
cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml --release -- --check-proof-manifest kernel\seal-os\target\x86_64-unknown-uefi\release\vbox-smoke\proof-manifest.txt
cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml --release -- --check-seal-abi .
cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml --release -- --check-language-hygiene .
cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml --release -- --check-o1-allocator .
cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml --release -- --unsafe-inventory .
```

The `aether-core` `no_std` command is intentionally manifest-pinned so it can
run from the repository root.

## GUI Proof

The QEMU pixel proof is machine checked by:

```powershell
cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml --release -- --check-proof-screen kernel\seal-os\target\x86_64-unknown-uefi\release\qemu-proof\screen.ppm
```

It verifies a nonblank 1024x768 first desktop frame with icon lane, control
region, and primary terminal titlebar visible. The PNG is for human inspection.

## Oracle VM VirtualBox

Convert the raw image to VDI:

```powershell
powershell -File .\build-vbox.ps1
```

Recommended VM settings:

- Type: Other/Unknown 64-bit
- EFI: enabled
- RAM: 4096 MB
- Video: VMSVGA, 128 MB video memory
- Disk: generated VDI from `seal-os.img`
- Optional NIC: Intel PRO/1000 MT Desktop

VirtualBox currently produces visual smoke evidence. QEMU is the canonical
machine-checked pixel proof path.

## Host Test Status

Do not use this as a Seal OS kernel proof:

```powershell
cargo +nightly test --lib
```

The kernel crate is configured for `x86_64-unknown-uefi` with `build-std`.
That is correct for the OS image, but it makes raw host tests mix incompatible
`core` metadata. The long-term fix is a separate host-harness crate or Cargo
config that imports only hardware-free modules.

## Current Known Gaps

- Raw host unit tests for `kernel/seal-os` need a dedicated host harness.
- Unsafe Rust inventory is reported but not yet threshold-gated.
- Vendor GPU acceleration is not present; GUI proof uses framebuffer rendering.
- Long-running VM soak tests are still needed.
- Ubuntu comparison benchmarks are still pending.
