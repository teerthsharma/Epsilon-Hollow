# Seal OS Testing Guide

This document describes how to build, run, and verify Seal OS. It is intended for automated testing by AI agents or human testers.

## Prerequisites

- Docker Desktop (recommended) OR:
  - Rust nightly (`rustup toolchain install nightly`)
  - `rust-src` component (`rustup component add rust-src --toolchain nightly`)
- Optional ISO tools: `xorriso`, `grub-mkrescue`, or `mkisofs`/`genisoimage`
  - `qemu-system-x86_64`

## Method 1: Docker (Recommended)

```bash
cd kernel/seal-os
docker compose up --build
```

This builds the kernel, creates `seal-os.img`, and runs it in QEMU with OVMF/UEFI firmware.

**Expected serial output** (appears in terminal/stdout):

```
Seal OS v0.4.5 — The Geometrical Operating System
All data = geometry on S^2. File moves = O(1) topological surgery.

[BOOT] Heap initialized (16 MB)
[BOOT] IDT + PIC initialized
[FB] 1024x768 @ 0xFD000000, pitch=4096, bpp=32
[BOOT] Framebuffer: 1024x768x32bpp
[T4/AGCR] Governor online: epsilon = 0.1000
[T1/TSS]  Voronoi index: 8 cells, test lookup -> cell N
[BOOT] All T1-T10 theorems VERIFIED; T1-T5 ACTIVE in runtime paths
[ManifoldFS] Initializing
[ManifoldFS] Teleported 'hello.txt' (19 bytes) in N ticks — O(1)
[ManifoldFS] N files, N dirs, governor e=X.XXXX
[Scheduler] 4 tasks, running 'kernel', epsilon=X.XXXX
[Syscall] getpid=1, theorems=T1:ACTIVE T2:ACTIVE T3:ACTIVE T4:ACTIVE T5:ACTIVE, query=theorem_1: ACTIVE
[Shell] Seal OS Theorem Status
[Shell] ======================
[Shell]   T1/TSS = ACTIVE
[Shell] Seal OS v0.4.5 (x86_64)
[Desktop] Rendering desktop environment
[Desktop] 3 windows active
[BOOT] Seal OS desktop ready.
[BOOT] Entering event loop
```

## Method 2: Local Build

```bash
cd kernel/seal-os
cargo +nightly build --release
```

**Expected**: Build succeeds. Binary at `target/x86_64-unknown-uefi/release/seal-os.efi`.

### Create VM disk image and run

```bash
cd ../seal-mkimage
cargo +stable run --release

cd ../seal-os
./run-qemu.sh
```

### Oracle VM VirtualBox

Convert the raw image to VDI, then attach the VDI as a SATA hard disk:

```powershell
powershell -File .\build-vbox.ps1
```

Manual conversion:

```powershell
VBoxManage convertfromraw --format VDI target\x86_64-unknown-uefi\release\seal-os.img seal-os.vdi
```

- Enable EFI.
- Use `Other/Unknown (64-bit)`.
- Set RAM to 4096 MB.
- Use VMSVGA with 128 MB video memory.
- Use an Intel PRO/1000 MT Desktop NIC if networking is needed.

## Verification Checklist

### Build Verification
- [ ] `cargo +nightly build --release` succeeds with no errors
- [ ] `seal-os.efi` exists
- [ ] `seal-os.img` exists
- [ ] Optional ISO builder produces `seal-os.iso`

### Boot Verification (Serial Output)
- [ ] "Seal OS v0.4.5" banner appears
- [ ] "Heap initialized (16 MB)" — memory works
- [ ] "IDT + PIC initialized" — interrupts work
- [ ] "Governor online: epsilon = 0.1000" — T4 active
- [ ] "Voronoi index: 8 cells" — T1 active
- [ ] "All T1-T10 theorems VERIFIED"
- [ ] ManifoldFS initializes, teleport succeeds with "O(1)"
- [ ] Scheduler reports 4 tasks
- [ ] Syscall dispatch returns correct values
- [ ] Shell executes commands (theorems, uname)
- [ ] Desktop reports windows active
- [ ] No kernel panics

### Graphical Verification (QEMU Display)
- [ ] Boot splash appears (ASCII seal art + progress bar)
- [ ] Progress bar fills from 0% to 100%
- [ ] Desktop wallpaper renders (dark background with visual elements)
- [ ] Taskbar visible at bottom of screen
- [ ] 3 windows visible: Terminal, Seal IDE, Theorems
- [ ] Windows have title bars and close buttons
- [ ] Mouse cursor visible (if mouse is moved)

### ManifoldFS Verification
- [ ] Files stored successfully (store_text)
- [ ] O(1) teleport demonstrated (elapsed_ticks is small)
- [ ] Directory operations (`create`, `look`) work
- [ ] Theorem status shows all 5 ACTIVE
- [ ] Governor epsilon adapts (changes from initial value)

### Theorem Verification
All five theorems must be demonstrably active:

| Theorem | Evidence |
|---------|----------|
| T1/TSS | Voronoi cell assignment in FS operations, scheduler task grouping |
| T2/SCM | SpectralContractionOperator used in scheduler prediction, FS prefetch |
| T3/GMC | Entropy calculation in ManifoldFS, Betti-0 in encoder |
| T4/AGCR | Governor epsilon in scheduler, compositor, FS governor_tick |
| T5/HCS | Hyperbolic ratio updated on mkdir, depth tracking |

## Source Code Map

For code review, the key files are:

| What | File | Lines |
|------|------|-------|
| Kernel entry | `src/main.rs` | Full boot sequence L0-L10 |
| UEFI entry | `src/boot/uefi_entry.rs` | Firmware handoff and BootInfo |
| ManifoldFS | `src/fs/manifold_fs.rs` | T1-T5 runtime filesystem |
| Encoder | `src/fs/encoder.rs` | Data → S² projection |
| Scheduler | `src/process/scheduler.rs` | T1+T2+T4 scheduling |
| Compositor | `src/wm/compositor.rs` | T1+T4 window manager |
| Shell | `src/apps/shell.rs` | All commands |
| Terminal | `src/apps/terminal.rs` | Text rendering |
| IDE | `src/apps/seal_ide.rs` | Syntax highlighting |
| File Manager | `src/apps/file_manager.rs` | S² projection view |
| Theorem Viewer | `src/apps/theorem_viewer.rs` | T1-T10 dashboard |

## Known Limitations

1. **No true userspace**: All "processes" run in kernel space (ring 0). True ring 3 execution requires per-process page tables.
2. **Bump allocator**: Memory is never freed. A slab allocator would be needed for long-running systems.
3. **No USB HID**: Only PS/2 keyboard/mouse via IRQ. USB requires xHCI driver.
4. **No persistent storage**: ManifoldFS is in-memory only. Disk I/O requires ATA/NVMe driver.
5. **Interactive input limited**: Keyboard input in graphical mode routes to serial. Full event routing requires the WM event loop to process IRQs.
6. **Single core**: No SMP/multi-core support.

## Automated Testing

For CI, the build verification alone is sufficient:

```bash
cd kernel/seal-os
cargo +nightly build --release
test $? -eq 0 && echo "PASS" || echo "FAIL"
```

For QEMU-based testing, capture serial output and grep:

```bash
timeout 30 qemu-system-x86_64 \
  -machine q35 \
  -drive if=pflash,format=raw,readonly=on,file=/usr/share/OVMF/OVMF_CODE_4M.fd \
  -drive file=target/x86_64-unknown-uefi/release/seal-os.img,format=raw \
  -serial stdio \
  -m 512M \
  -display none \
  -no-reboot 2>&1 | tee /tmp/seal-os.log

# Verify key milestones
grep -q "Seal OS v0.4.5" /tmp/seal-os.log && echo "PASS: Boot"
grep -q "All T1-T10 theorems VERIFIED" /tmp/seal-os.log && echo "PASS: Theorems"
grep -q "Teleported.*O(1)" /tmp/seal-os.log && echo "PASS: Teleport"
grep -q "desktop ready" /tmp/seal-os.log && echo "PASS: Desktop"
```
