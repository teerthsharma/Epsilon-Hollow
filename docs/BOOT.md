# Boot Sequence Technical Reference

Seal OS boots as a pure Rust UEFI application. The supported VM path is a GPT raw disk image with a FAT EFI System Partition containing `EFI/BOOT/BOOTX64.EFI`. Use [VM_RUNBOOK.md](VM_RUNBOOK.md) for the full QEMU and Oracle VM VirtualBox guide.

No GRUB. No Multiboot. No 32-bit trampoline. Firmware enters the kernel through UEFI, then Seal OS exits boot services and owns the machine.

## Current Boot Flow

```text
UEFI firmware
  -> load EFI/BOOT/BOOTX64.EFI
  -> kernel/seal-os/src/boot/uefi_entry.rs
  -> initialize serial logging
  -> read ACPI RSDP from UEFI config tables
  -> obtain framebuffer through UEFI GOP
  -> exit boot services with the UEFI memory map
  -> pass BootInfo to kernel_main()
  -> initialize memory, CPU, interrupts, theorem core, drivers, desktop
```

## Build Artifacts

| Artifact | Path | Purpose |
|---|---|---|
| EFI kernel | `kernel/seal-os/target/x86_64-unknown-uefi/release/seal-os.efi` | PE/COFF UEFI executable |
| Raw disk | `kernel/seal-os/target/x86_64-unknown-uefi/release/seal-os.img` | GPT disk used by QEMU and conversion tools |
| ESP image | `kernel/seal-os/target/x86_64-unknown-uefi/release/seal-os-efi.img` | FAT EFI System Partition payload |
| VirtualBox disk | `kernel/seal-os/target/x86_64-unknown-uefi/release/seal-os.vdi` | Created from raw image with `build-vbox.ps1` or `VBoxManage convertfromraw` |

## Build Commands

```bash
cd kernel/seal-os
cargo +nightly build --release

cd ../seal-mkimage
cargo +stable run --release
```

The image builder copies the EFI binary to the standard removable-media path:

```text
EFI/BOOT/BOOTX64.EFI
```

## QEMU

Linux/macOS:

```bash
cd kernel/seal-os
./run-qemu.sh
```

Windows PowerShell:

```powershell
cd kernel\seal-os
powershell -File .\run-qemu.ps1
```

QEMU needs OVMF/EDK2 firmware. The scripts search common OVMF paths and use the raw `seal-os.img` disk.

## Oracle VM VirtualBox

VirtualBox should boot a VDI converted from the raw image:

```powershell
cd kernel\seal-os
powershell -NoProfile -ExecutionPolicy Bypass -File .\build-vbox.ps1
```

Automated smoke test:

```powershell
cd kernel\seal-os
powershell -NoProfile -ExecutionPolicy Bypass -File .\smoke-vbox.ps1 -Seconds 240 -VBoxCommandTimeoutSeconds 30
```

The smoke helper uses an isolated VirtualBox home and a unique temporary VM name.
If `VBoxManage createvm` times out before the VM starts, repair the host
VirtualBox service first; that is not a Seal OS boot failure.

Manual conversion:

```powershell
VBoxManage convertfromraw --format VDI target\x86_64-unknown-uefi\release\seal-os.img seal-os.vdi
```

Recommended VM settings:

| Setting | Value |
|---|---|
| Type | Other |
| Version | Other/Unknown (64-bit) |
| EFI | Enabled |
| RAM | 4096 MB |
| CPU | 1 first, 2 after boot is stable |
| Display | VMSVGA, 128 MB video memory |
| Storage | SATA/AHCI, attach `seal-os.vdi` |
| Network | Intel PRO/1000 MT Desktop |

## Boot Theorem Gate

During `init_theorems()`, Seal OS runs cheap no_std checks from `aether_verified` for T1-T10:

| Range | Boot status | Runtime role |
|---|---|---|
| T1-T5 | Verified at boot | Active in filesystem, scheduler, memory, compositor/power paths |
| T6-T10 | Verified at boot | HFT/ML world-model bounds exposed through theorem status |

If any theorem check fails, the kernel panics instead of booting a false theorem state.

## Aether Runtime Gate

After the desktop app host is created, Seal OS runs a tiny Aether boot probe
through the kernel runtime. A passing VM proof must contain:

```text
[Aether-Lang] runtime proof: parser=ok interpreter=ok app_host=ok script=aether_boot_probe result=seal-topology-ok
```

`seal-mkimage --check-aether-runtime <serial.log>` is the audit gate for this
marker.
