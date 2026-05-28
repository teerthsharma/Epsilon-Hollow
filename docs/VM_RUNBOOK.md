# Seal OS VM Runbook

This is the clean path from checkout to VM boot proof. It covers QEMU and Oracle
VM VirtualBox. The proof target is serial output, not vibes.

## Boot Proof Rule

A VM run passes only when the serial log reaches:

```text
[BOOT] Seal OS desktop ready.
```

A VM run fails on:

- `!!! SEAL OS KERNEL PANIC !!!`
- `[FAULT]`
- `[WATCHDOG]`
- `gurumeditation`
- VM poweroff before the success sentinel
- timeout before the success sentinel

## Expected Serial Sentinels

The exact log can include extra driver messages, but these lines are the core
proof spine:

```text
Seal OS - UEFI entry reached, serial online
[BOOT] GOP mode set
[BOOT] Framebuffer accessibility test passed
Seal OS - UEFI boot complete, boot services exited
[BOOT] Heap initialized
[BOOT] IDT + PIC initialized
[BOOT] SYSCALL/SYSRET MSRs programmed
[T4/AGCR] Governor online
[T1/TSS]  Voronoi index: 8 cells
[THEOREM] T1/TSS VERIFIED
[THEOREM] T2/SCM VERIFIED
[THEOREM] T3/GMC VERIFIED
[THEOREM] T4/AGCR VERIFIED
[THEOREM] T5/HCS VERIFIED
[THEOREM] T6/RGCS VERIFIED
[THEOREM] T7/PHKP VERIFIED
[THEOREM] T8/TEB VERIFIED
[THEOREM] T9/CMA VERIFIED
[THEOREM] T10/WPHB VERIFIED
[BOOT] All T1-T10 theorems VERIFIED; T1-T5 ACTIVE in runtime paths
[BOOT] All layers initialized.
[BOOT] Seal OS desktop ready.
```

## Windows Build

From repo root:

```powershell
cargo +nightly build --manifest-path kernel\seal-os\Cargo.toml --release
cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml --release
```

Expected artifacts:

```text
kernel\seal-os\target\x86_64-unknown-uefi\release\seal-os.efi
kernel\seal-os\target\x86_64-unknown-uefi\release\seal-os.img
kernel\seal-os\target\x86_64-unknown-uefi\release\seal-os-efi.img
```

Verify the raw image:

```powershell
cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml --release -- --verify kernel\seal-os\target\x86_64-unknown-uefi\release\seal-os.img kernel\seal-os\target\x86_64-unknown-uefi\release\seal-os.efi
```

Expected:

```text
[mkimage] VERIFY OK
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
powershell -NoProfile -ExecutionPolicy Bypass -File .\run-qemu.ps1
```

QEMU needs OVMF/EDK2 firmware. The helper scripts search common OVMF paths and
boot the raw GPT disk image.

## Oracle VM VirtualBox: Automated Smoke

From `kernel\seal-os`:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\smoke-vbox.ps1 -Seconds 240
```

The script:

- builds `seal-os.vdi` if needed
- copies it to `target\x86_64-unknown-uefi\release\vbox-smoke\seal-os-smoke.vdi`
- creates a temporary VM named `SealOS-Codex-Smoke`
- enables EFI
- attaches the copied disk through SATA/AHCI
- writes COM1 serial output to `vbox-smoke\serial.log`
- captures `vbox-smoke\screenshot.png`
- removes the temporary VM on exit

PASS requires:

```text
Smoke verdict: PASS - success sentinel matched: \[BOOT\] Seal OS desktop ready\.
```

## Oracle VM VirtualBox: Manual GUI Run

From `kernel\seal-os`, create the VDI:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\build-vbox.ps1
```

Manual conversion alternative:

```powershell
VBoxManage convertfromraw --format VDI target\x86_64-unknown-uefi\release\seal-os.img target\x86_64-unknown-uefi\release\seal-os.vdi
```

Recommended VM settings:

| Setting | Value |
| --- | --- |
| Type | Other |
| Version | Other/Unknown (64-bit) |
| EFI | Enabled |
| RAM | 4096 MB |
| CPU | 1 first, 2 after boot is stable |
| Display | VMSVGA |
| Video memory | 128 MB |
| Storage controller | SATA/AHCI |
| Disk | `target\x86_64-unknown-uefi\release\seal-os.vdi` |
| Audio | Disabled |
| Network | Disabled for smoke; Intel PRO/1000 MT Desktop for network tests |

Serial capture for a manual VM:

```powershell
VBoxManage modifyvm SealOS --uart1 0x3F8 4 --uartmode1 file C:\tmp\seal-os-serial.log
```

## Known VirtualBox Quirks

- VirtualBox can offer very large GOP modes. Seal OS caps the selected GOP mode
  to a sane boot resolution so the first desktop render does not waste boot time
  on oversized framebuffers.
- VirtualBox 7.x prefers `--audio-driver none`; older `--audio none` prints a
  warning.
- If VBoxManage reports a COM registry error, close VirtualBox GUI and retry the
  smoke script. If it repeats, rebooting the host VirtualBox service may be
  required.
- Always use a copied VDI for smoke tests. Do not boot the source build artifact
  directly when the script can copy it.

## Troubleshooting

| Symptom | Likely cause | Next action |
| --- | --- | --- |
| No serial log | UART not configured | Use `smoke-vbox.ps1` or add `--uart1 0x3F8 4 --uartmode1 file <log>` |
| UEFI shell opens | EFI path missing | Rebuild image and verify `EFI/BOOT/BOOTX64.EFI` with `seal-mkimage --verify` |
| Panic before theorem gate | memory/UEFI/relocation bug | inspect first `[FAULT]` or panic line |
| Theorem failure | theorem gate input drift | inspect `[THEOREM] ... FAILED` line and Rust verifier |
| Timeout after desktop render | GUI path too slow or stuck | inspect last serial lines around `[Desktop]` |
| Guru meditation | VirtualBox CPU fault | inspect `serial.log`, VM log, and last boot milestone |

## What Counts As Evidence

Good evidence:

- fresh serial log from the current image
- image verifier output from the current image
- screenshot from the same VM run
- CI artifact from the current commit

Weak evidence:

- old serial logs
- manual screenshot without image hash
- "it booted once" without sentinels
- QEMU proof used to claim VirtualBox proof

