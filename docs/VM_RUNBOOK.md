# Seal OS VM Runbook

This is the clean path from checkout to VM boot proof. It covers QEMU and Oracle
VM VirtualBox. The proof target is serial output, a machine-checked
`screen.ppm` pixel gate for QEMU GUI proof, and a `proof-manifest.txt` bundle
that ties artifacts back to the exact image, EFI, commit, backend, and gates.
Because this is a solo-researcher topological OS, the runbook treats every VM
claim as a proof contract, not a vibe.

## Boot Proof Rule

A VM run passes only when the serial log reaches:

```text
[Aether-Lang] runtime proof: parser=ok interpreter=ok app_host=ok script=aether_boot_probe result=seal-topology-ok
[LAAMBA] app proof: version=1 native_app=kernel window=LAAMBA_Governor window_id=<n> launcher_id=10 desktop_icon=1 start_menu=1 aether_host_window_id=<n> runtime_bridge=rust_native_manifest python_runtime=0 result=pass
[GFX] desktop-proof version=1 surface=framebuffer width=1024 height=768 bpp=32 pitch=<n> back_buffer=1 window_count=12 focused_window_id=<n> scanned_pixels=786432 nonblack_px=<n> visible_icons=10 icon_region_signal=<n> icon_color_buckets=<n> control_region_signal=<n> primary_titlebar_signal=<n> start_button_signal=<n> theorem_indicator_signal=<n> minimized_app_lane_signal=<n> power_button_signal=<n> sampled_pixels=<n> nonblack_samples=<n> sample_hash=<n> result=pass
[BOOT] Desktop proof frame blit done
[GFX] desktop-live-proof version=1 route=desktop_handle_input action=desktop_icon_launch app=Files app_id=3 events=2 handled=1 icon_hit=1 launched_app_id=3 pre_focused=<n> post_focused=<n> post_window_id=<n> window_count=12 pre_hash=<n> post_hash=<n> changed_samples=<n> blit=1 result=pass
[GFX] desktop-soak frames=24 p50_cycles=<n> p95_cycles=<n> max_cycles=<n> missed_16ms=unscaled input_events=0 dirty_px_max=786432
[BOOT] Seal OS desktop ready.
[EVENT] Entering real event loop
```

For QEMU GUI proof, `seal-mkimage --check-proof-screen` must also pass against
the captured `qemu-proof\screen.ppm`. The script also writes `screen.png`; the
checker accepts that path when the sibling PPM proof capture is present.
`seal-mkimage --check-laamba-app-proof` separately parses the LAAMBA line and
rejects Python-backed or missing-window app proofs.
The desktop soak gate now also parses `[GFX] desktop-live-proof`, which proves
the real desktop input path launched and focused Files from an icon click before
the proof bundle accepts the desktop as live.

For QEMU provenance proof, `run-qemu.ps1 -HeadlessProof` writes a timestamped
`qemu-proof-runs\<stamp>` staging directory, runs every hard gate, writes
`proof-manifest.txt`, verifies that manifest with
`seal-mkimage --check-proof-manifest`, and only then publishes the canonical
`qemu-proof` directory. To prove the bundle belongs to the checkout in front of
you, also run `seal-mkimage --check-current-proof-manifest`; it compares the
manifest commit and dirty flag against local Git state instead of accepting an
older proof bundle as current. Failed proof runs are kept in the staging
directory.

If a staging run already has `serial.log`, `screen.ppm`, `screen.png`,
`seal-os.img`, and `seal-os.efi` but the wrapper exits before writing the
manifest, regenerate the manifest with the Rust verifier instead of hand-writing
checksums:

```powershell
cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml --release -- --write-qemu-proof-manifest kernel\seal-os\target\x86_64-unknown-uefi\release\qemu-proof-runs\<stamp> kernel\seal-os\target\x86_64-unknown-uefi\release\qemu-proof-runs\<stamp>\seal-os.img native 240 .
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
[ALLOC] O(1) proof: topo_cells=8, l3_word_probes_per_cell=2, single_word_probes_per_cell=8192, contiguous_candidate_probes=128, contiguous_max_run_pages=64, toporam_max_run_pages=64, marking=bounded_by_contiguous_max_run_pages
[ALLOC] runtime counters: fast_hits=1, bounded_misses=0, max_contiguous_probes_seen=0
[BENCH] toporam-alloc iterations=64 ok=64 p50_cycles=<n> p95_cycles=<n> max_cycles=<n> target_cell_hits_delta=64 target_cell_fallbacks_delta=0 low_to_high_fallbacks_delta=0 high_to_low_fallbacks_delta=0 pcie_to_high_fallbacks_delta=0 pcie_to_low_fallbacks_delta=0 free_before=<n> free_after=<n>
[BENCH] alloc-frame iterations=64 ok=64 p50_cycles=<n> p95_cycles=<n> max_cycles=<n> fast_hits_delta=64 bounded_misses_delta=0 max_contiguous_probes_seen_delta=0 free_before=<n> free_after=<n>
[BENCH] manifold-teleport api=teleport fs_mode=mock_block persistence=metadata_only samples=3 ok=3 same_inode=3 src_gone=3 dst_present=3 entries_min=8 entries_max=256 payload_bytes=64 p50_cycles=<n> p95_cycles=<n> max_cycles=<n> ticks_max=<n> metadata_ops_max=7 persistence_bytes_per_move=0 payload_points=<n>
[BENCH] scheduler-select-next selector=select_next_task mode=live_requeue clock=rdtsc iterations=64 ok=64 ready_before=3 ready_after=3 cells=8 priority_buckets=256 voronoi_locate_probes=8 max_cell_bitmap_tests=9 max_priority_bucket_scan=256 context_switches=0 selected_priority_max=<n> p50_cycles=<n> p95_cycles=<n> max_cycles=<n>
[BENCH] tcp-packet-demux api=handle_tcp_packet fixture=listener_first accepted_state=established ok=1 listener_first=1 exact_flow=1 decoy_rx_bytes=0 listener_fallback=1 payload_bytes=4 rx_bytes=4 cleanup=ok
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
[Aether-Lang] runtime proof: parser=ok interpreter=ok app_host=ok script=aether_boot_probe result=seal-topology-ok
[LAAMBA] app proof: version=1 native_app=kernel window=LAAMBA_Governor window_id=<n> launcher_id=10 desktop_icon=1 start_menu=1 aether_host_window_id=<n> runtime_bridge=rust_native_manifest python_runtime=0 result=pass
[GFX] desktop-proof version=1 surface=framebuffer width=1024 height=768 bpp=32 pitch=<n> back_buffer=1 window_count=12 focused_window_id=<n> scanned_pixels=786432 nonblack_px=<n> visible_icons=10 icon_region_signal=<n> icon_color_buckets=<n> control_region_signal=<n> primary_titlebar_signal=<n> start_button_signal=<n> theorem_indicator_signal=<n> minimized_app_lane_signal=<n> power_button_signal=<n> sampled_pixels=<n> nonblack_samples=<n> sample_hash=<n> result=pass
[BOOT] Desktop proof frame blit done
[GFX] desktop-live-proof version=1 route=desktop_handle_input action=desktop_icon_launch app=Files app_id=3 events=2 handled=1 icon_hit=1 launched_app_id=3 pre_focused=<n> post_focused=<n> post_window_id=<n> window_count=12 pre_hash=<n> post_hash=<n> changed_samples=<n> blit=1 result=pass
[GFX] desktop-soak frames=24 p50_cycles=<n> p95_cycles=<n> max_cycles=<n> missed_16ms=unscaled input_events=0 dirty_px_max=786432
[BOOT] Seal OS desktop ready.
[EVENT] Entering real event loop
```

## Windows Build

From repo root:

```powershell
cd kernel\seal-os
cargo +nightly build --release
cd ..\seal-mkimage
cargo +stable run --release
cd ..\..
```

Expected artifacts:

```text
kernel\seal-os\target\x86_64-unknown-uefi\release\seal-os.efi
kernel\seal-os\target\x86_64-unknown-uefi\release\seal-os.img
kernel\seal-os\target\x86_64-unknown-uefi\release\seal-os-efi.img
```

`seal-os.img` is a raw GPT disk. Partition 1 is the FAT EFI System Partition.
Partition 2 is `ManifoldFS` and starts at LBA 133120; it contains the `MNFD`
superblock used by the persistent root filesystem.

Verify the raw image:

```powershell
cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml --release -- --verify kernel\seal-os\target\x86_64-unknown-uefi\release\seal-os.img kernel\seal-os\target\x86_64-unknown-uefi\release\seal-os.efi
```

Expected:

```text
[mkimage] VERIFY OK
```

Verify the full VM serial proof after QEMU:

```powershell
cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml --release -- --check-vm-proof kernel\seal-os\target\x86_64-unknown-uefi\release\qemu-proof\serial.log
cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml --release -- --check-aether-runtime kernel\seal-os\target\x86_64-unknown-uefi\release\qemu-proof\serial.log
cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml --release -- --check-laamba-app-proof kernel\seal-os\target\x86_64-unknown-uefi\release\qemu-proof\serial.log
cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml --release -- --check-desktop-soak kernel\seal-os\target\x86_64-unknown-uefi\release\qemu-proof\serial.log
cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml --release -- --check-benchmark-log kernel\seal-os\target\x86_64-unknown-uefi\release\qemu-proof\serial.log
cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml --release -- --check-proof-screen kernel\seal-os\target\x86_64-unknown-uefi\release\qemu-proof\screen.ppm
cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml --release -- --check-proof-manifest kernel\seal-os\target\x86_64-unknown-uefi\release\qemu-proof\proof-manifest.txt
cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml --release -- --check-current-proof-manifest kernel\seal-os\target\x86_64-unknown-uefi\release\qemu-proof\proof-manifest.txt .
```

Verify the full Oracle/VirtualBox serial proof after `smoke-vbox.ps1`:

```powershell
cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml --release -- --verify kernel\seal-os\target\x86_64-unknown-uefi\release\seal-os.img kernel\seal-os\target\x86_64-unknown-uefi\release\seal-os.efi
cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml --release -- --check-vbox-proof kernel\seal-os\target\x86_64-unknown-uefi\release\vbox-smoke\serial.log
cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml --release -- --check-theorem-log kernel\seal-os\target\x86_64-unknown-uefi\release\vbox-smoke\serial.log
cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml --release -- --check-aether-runtime kernel\seal-os\target\x86_64-unknown-uefi\release\vbox-smoke\serial.log
cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml --release -- --check-laamba-app-proof kernel\seal-os\target\x86_64-unknown-uefi\release\vbox-smoke\serial.log
cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml --release -- --check-desktop-soak kernel\seal-os\target\x86_64-unknown-uefi\release\vbox-smoke\serial.log
cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml --release -- --check-benchmark-log kernel\seal-os\target\x86_64-unknown-uefi\release\vbox-smoke\serial.log
cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml --release -- --check-proof-manifest kernel\seal-os\target\x86_64-unknown-uefi\release\vbox-smoke\proof-manifest.txt
cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml --release -- --check-current-proof-manifest kernel\seal-os\target\x86_64-unknown-uefi\release\vbox-smoke\proof-manifest.txt .
```

The benchmark log gate requires all five current markers:
`[BENCH] toporam-alloc`, `[BENCH] alloc-frame`,
`[BENCH] manifold-teleport`, `[BENCH] scheduler-select-next`, and
`[BENCH] tcp-packet-demux`.
The Aether runtime gate requires the parser/interpreter/app-host boot marker.

Expected:

```text
[seal-audit] VM PROOF OK
[seal-audit] PROOF MANIFEST OK
[seal-audit] CURRENT PROOF MANIFEST OK
```

## Ubuntu Allocation Comparison Gate

Seal OS allocator logs prove the bare-metal hot path shape. A claim that Seal
beats Ubuntu additionally requires a same-machine Ubuntu 26.04 LTS artifact:

```bash
# Run these inside native Ubuntu 26.04 VM/bare metal or a self-hosted Ubuntu 26.04 runner.
# WSL output is rejected by the audit gate.
grep -E '^(ID|VERSION_ID)=' /etc/os-release
uname -r
cargo +stable run --manifest-path tools/ubuntu-alloc-bench/Cargo.toml --release -- 64 > tools/ubuntu-alloc-bench/ubuntu-alloc.log
cargo +stable run --manifest-path kernel/seal-mkimage/Cargo.toml --release -- --check-ubuntu-benchmark-log tools/ubuntu-alloc-bench/ubuntu-alloc.log
cargo +stable run --manifest-path kernel/seal-mkimage/Cargo.toml --release -- --compare-benchmark-logs kernel/seal-os/target/x86_64-unknown-uefi/release/qemu-proof/serial.log tools/ubuntu-alloc-bench/ubuntu-alloc.log
```

`--check-ubuntu-benchmark-log` rejects artifacts that do not report
`os=ubuntu version_id=26.04 kernel=<native-kernel>` and rejects kernels
containing `microsoft` or `wsl`. That keeps Windows, WSL, older Ubuntu, or
non-Ubuntu smoke output from becoming fake current-Ubuntu evidence.

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

Current QEMU storage proof must include:

```text
[AHCI] Device model: QEMU HARDDISK
[AHCI] Registered as block device 0x800
[disk::ahci] First disk readable (sector 0 OK)
[VFS] ManifoldFS mounted from disk
```

The Rust gate `--check-vm-proof` requires those QEMU storage lines, the T1-T10
theorem gate, allocator benchmark marker, ManifoldFS teleport benchmark marker,
desktop proof-frame blit, live desktop input marker, desktop soak marker,
desktop-ready sentinel, event-loop entry, and absence of fatal boot markers. For
Oracle/VirtualBox, `--check-vbox-proof`
applies the same proof contract but requires `VBOX HARDDISK` instead of
`QEMU HARDDISK`. Both gates reject ramfs fallback and missing-AHCI sentinels.

## Oracle VM VirtualBox: Automated Smoke

From `kernel\seal-os`:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\smoke-vbox.ps1 -Seconds 240
```

Use `-SkipBuild` only after `seal-os.vdi` was regenerated from the current
`seal-os.img`. A stale VDI can boot an old kernel and fail the current allocator
proof gate.

The script:

- builds `seal-os.vdi` if needed
- copies it to `target\x86_64-unknown-uefi\release\vbox-smoke\seal-os-smoke.vdi`
- uses an isolated `VBOX_USER_HOME` by default so stale user VM registry state
  does not poison the smoke run
- creates a uniquely named temporary VM such as `SealOS-Codex-Smoke-20260529-091745`
- enables EFI
- attaches the copied disk through SATA/AHCI
- writes COM1 serial output to `vbox-smoke\serial.log`
- captures `vbox-smoke\screenshot.png` as a smoke artifact
- removes the temporary VM on exit
- applies `-VBoxCommandTimeoutSeconds` to each `VBoxManage` command so a broken
  host VirtualBox service produces a clear failure instead of an endless hang

PASS requires:

```text
Smoke verdict: PASS - all success sentinels matched
```

The required success sentinels are the allocator O(1) proof marker, every
`[THEOREM] T1` through `T10` verified line, `VBOX HARDDISK`, block device
`0x800`, readable sector 0, persistent ManifoldFS, the desktop proof-frame blit,
desktop ready, and event-loop entry. The script runs
`seal-mkimage --verify`, `--check-vbox-proof`, theorem/Aether/desktop/benchmark
gates, and `--check-proof-manifest` before returning success.

VirtualBox smoke currently captures PNG only. It is visual evidence, not the same
machine-checked PPM gate used by QEMU. The VirtualBox manifest therefore
records `screenshot.png` as visual evidence and deliberately refuses a
`screen_ppm` parity claim.

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
- If `VBoxManage showvminfo` or `VBoxManage createvm` times out before the VM is
  created, the OS has not failed yet. The host VirtualBox service/COM layer is
  stuck. Check `Get-Service VBoxSDS`, stop stale `VBoxSVC`, start `VBoxSDS`, or
  reboot the host service before retrying.
- Always use a copied VDI for smoke tests. Do not boot the source build artifact
  directly when the script can copy it.

## Troubleshooting

| Symptom | Likely cause | Next action |
| --- | --- | --- |
| No serial log | UART not configured | Use `smoke-vbox.ps1` or add `--uart1 0x3F8 4 --uartmode1 file <log>` |
| UEFI shell opens | EFI path missing | Rebuild image and verify `EFI/BOOT/BOOTX64.EFI` with `seal-mkimage --verify` |
| `VBoxManage createvm` times out | Host VirtualBox service or COM registry stuck | restart `VBoxSDS`/`VBoxSVC`; retry with isolated default `VBOX_USER_HOME` |
| Panic before theorem gate | memory/UEFI/relocation bug | inspect first `[FAULT]` or panic line |
| Theorem failure | theorem gate input drift | inspect `[THEOREM] ... FAILED` line and Rust verifier |
| Timeout after desktop render | GUI path too slow or stuck | inspect last serial lines around `[Desktop]` |
| Guru meditation | VirtualBox CPU fault | inspect `serial.log`, VM log, and last boot milestone |
| `No persistent disk found` | AHCI or `MNFD` partition mount failed | check `[AHCI]`, `[disk::ahci]`, and `[VFS]` lines; rebuild image with `seal-mkimage` |

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
