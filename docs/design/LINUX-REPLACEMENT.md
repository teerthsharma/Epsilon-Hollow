# Seal OS as a Linux kernel replacement

Status: accepted plan, 2026-09-27. Supersedes the FUTURE_PLAN.md policy that rejected POSIX, Linux, libc and GRUB compatibility.

## Goal

Seal OS boots under any Linux distribution as that distribution's kernel. The distribution's userland runs unmodified, its bootloader and initrd tooling work unchanged, and Linux device drivers are usable. Components Seal OS does not implement natively are brought in as ports of existing open-source kernel code.

## Starting point

Measured at commit `9ebbe2e` by four independent agents, each finding bound to a test that fails on that commit (tests under `tests/linux_parity/`, merged with this plan).

| Area | State at `9ebbe2e` | Evidence |
|---|---|---|
| User processes | No ring-3 instruction has ever executed. `spawn_user` holds `scheduler_lock` across `groups_for_uid` → `/etc/passwd` → `current_uid()`, which takes the same lock; `PerCpu::current_task` is set only inside `schedule()`, whose non-test callers return early while it is null. Every boot logs `context_switches=0`. | `chase_boot.sh usermode-seal`, `cameron_qemu_milestone.sh ring3-{seal,linux,glibc}`, `process/scheduler.rs:379,546,1383,1394,1408,1522` |
| Syscall entry | Stores onto the caller-chosen RSP; no `swapgs`; IA32_FMASK = 0x200 leaves TF, DF and AC set in ring 0; EFER.SCE is never set by the kernel (CI log `efer=0xd00`). | `test_cameron_syscall_entry.py`, `test_foreman_abi_substrate.py`, `process/userspace.rs:132-166,209-219` |
| Syscall ABI | 69 calls, own numbering, three argument registers. 1 of 49 syscalls made by five Ubuntu programs reaches the same call; 22 reach a different one (`access`→rmdir, `getgid`→pkg_remove, `munmap`→mkdir). Linux x86_64 has 386 native syscalls. | `test_chase_syscall_abi.py`, `syscall/table.rs:16-59,680` |
| Process model | One global user-address bump pointer that never reclaims; global `REGIONS` and `FILE_TABLE`; `fork` copies neither; no page cache; `copy_from_user` has no fault fixup; any unresolved page fault, including a user segfault, halts the machine. | `memory/mmap.rs:51,55`, `syscall/table.rs:471,730-734`, `security/smap_smep.rs:89-96`, `drivers/interrupts.rs:497-553` |
| ELF loading | Initial stack is three zero words: no argc, argv, envp or auxv; only `R_X86_64_RELATIVE`; no PT_TLS or `arch_prctl`. | `test_chase_syscall_abi.py`, `process/elf.rs:157-194,716` |
| Boot | UEFI application only. No Linux setup header, no EFI stub, no command line, no initramfs; `seal-mkimage` cannot place userland files in the root partition. | `test_cameron_boot_protocol.py`, `test_cameron_cmdline_initramfs.py`, `test_foreman_image_userland.py` |
| Filesystems | ext2 never reads `s_feature_incompat` and mounts a distro ext4 volume as ext2, then attempts to create `/swap.topo` on it. | `chase_boot.sh ext4`, `cameron_qemu_milestone.sh ext4-root`, `fs/ext2.rs:308` |
| Kernel hardening | Kernel self-probe: 4310 of 4310 scanned kernel pages writable and executable; `kaslr=0` under `-cpu qemu64`. | `chase_boot.sh wx` |
| Security model | `manifold_acl::check_access` runs on every lookup and exec and refuses access Linux permits (T5 denies uid ≥ 27 on root-owned files unless group and other bits are all set, so uid 1000 cannot read a 0644 libc). | `fs/vfs.rs:276,321,355`, `syscall/table.rs:890` |
| Drivers | AHCI, NVMe, virtio-blk (its `init` has no caller), virtio-net, virtio-gpu 2D, e1000, xHCI, HDA. No driver binding model, no ECAM, no MSI, no AML interpreter. | `cameron_qemu_milestone.sh virtio-root`, `drivers/pci.rs`, `drivers/acpi/fadt.rs:121` |
| Linux driver reuse | Binary modules are pinned: 924 of 924 carry one vermagic string and per-symbol CRCs. `e1000.ko` imports 152 symbols, Seal provides 2. 480 driver modules import 7,243 distinct symbols. 869 of 924 declare GPL-only licences. | `test_chase_linux_drivers.py`, `test_cameron_linux_module_abi.py` |

Prior systems that ran unmodified Linux userland on another kernel: WSL1 (about 235 syscalls, later replaced by a real Linux kernel in WSL2), gVisor (290 of 352 implemented or partial), Asterinas (230+), Maestro (135 of 437), FreeBSD Linuxulator, NetBSD compat_linux, Fuchsia Starnix, illumos LX zones, Managarm. Prior systems that reused Linux drivers: FreeBSD LinuxKPI with drm-kmod, Genode dde_linux (drivers as separate user-level components), L4 DDEKit, LKL.

## Decisions

### D1. The Linux x86_64 syscall ABI is the native ABI

Linux numbering, six argument registers (`rdi rsi rdx r10 r8 r9`), negative-errno returns, Linux structure layouts, and the System V initial stack with an auxiliary vector. There is one syscall table. The Seal-numbered POSIX-like calls are renumbered; their only user-mode caller is the embedded `EMERGENCY_SHELL_ELF`.

Seal-specific functionality (manifold, teleport, theorem status, packages, Wi-Fi and Bluetooth settings, chart graft/prune/list, the five FIT and five KV calls) is **not** given private syscall numbers. It is exposed as ioctls on a `/dev/seal` character device and as attributes under `/sys/kernel/seal/`. Reason: distribution sandboxes reject unknown syscall numbers. `SystemCallArchitectures=native` (set by journald, udevd, logind, networkd, resolved) treats any number with bit 30 (`__X32_SYSCALL_BIT`) as foreign, and allowlist filters (Docker's default profile, `@system-service`, bubblewrap) reject every number they do not list. Device nodes and sysfs are governed by ordinary file permissions, which those sandboxes already model.

Every syscall not implemented returns `-ENOSYS`. No handler returns success without performing the operation.

### D2. Seal OS boots and installs like a Linux kernel

- The kernel image carries the x86 Linux boot-protocol setup header and a PE/COFF EFI stub, so GRUB (`linux`/`initrd`), systemd-boot and unified kernel images load it.
- The command line is read from EFI LoadOptions and the setup header (`root=`, `rootfstype=`, `init=`, `console=`, `ro`, `rw`, `quiet`).
- The initramfs is loaded through the EFI `LoadFile2` `LINUX_EFI_INITRD_MEDIA` protocol and the setup-header ramdisk fields, unpacked (cpio newc; gzip and zstd) into a tmpfs root, and `/init` is executed; without an initramfs, `root=` is mounted and `/sbin/init` executed.
- `uname -r` reports `<LTS>-seal`, where `<LTS>` is the Linux long-term release whose userspace ABI the kernel targets. A `seal-kernel` distribution package installs the image through `kernel-install`/`installkernel` and provides `/lib/modules/<LTS>-seal/` whose `modules.builtin` lists every driver Seal provides, so `depmod`, `modprobe`, dracut, mkinitcpio and initramfs-tools work unchanged and produce an initrd that needs no `.ko` files.
- Secure Boot: there is no project signing key. Users sign the image locally and enroll their own key (MOK or `sbctl`); nothing central can leak.

### D3. Linux drivers run unmodified inside isolated driver servers

Linux driver source runs inside LKL (the Linux kernel built as a library), one process per driver server, in user mode. The server reaches its device through a user-mode driver interface (BAR mapping, IRQ delivered as an event, DMA buffers) confined by the IOMMU (VT-d through ACPI DMAR, AMD-Vi through IVRS), and exposes the device to the Seal kernel over virtio-style rings. The Seal kernel registers the device in its own device model, which is the single canonical source for `/sys/devices` and uevents.

- Licensing: the kernel source remains MIT. A driver server is a separate GPL-2.0 program built from a pinned upstream LKL tree and configuration; its source offer is that tree and configuration, published by the project.
- Maintenance: moving to the next LTS rebuilds the servers; a CVE fix is a server rebuild and restart, not a kernel rebuild.
- Security: a faulting or compromised driver is confined to its process and IOMMU domain.

Rejected: loading binary `.ko` modules (build-pinned: vermagic and symbol CRCs on 924 of 924); an in-kernel Linux API shim (it must reproduce the inline functions, macros and structure layouts of `include/linux`, which cannot be done clean-room by authors who have read that code, tracks an interface Linux documents as unstable, and would link GPL-only code into the kernel); relicensing the kernel GPL-2.0 (removes the licence obstacle but not the per-release API churn).

Native spec drivers (AHCI, NVMe, virtio-blk, virtio-net, virtio-gpu, e1000, xHCI) stay in the kernel; each gets a QEMU device-model boot gate.

### D4. Filesystems: refuse first, then implement, then prove crash consistency against Linux

1. ext2 refuses unknown INCOMPAT features and mounts read-only on unknown RO_COMPAT features. This lands before any change that could touch a foreign disk.
2. ext4 read (extents, 64bit, flex_bg, metadata_csum), then write with jbd2 journalling.
3. Crash-consistency gate: QEMU is killed at N injected points inside journal transactions; afterwards a stock Linux kernel must replay the journal, `e2fsck -fn` must report clean, and the volume must mount.
4. tmpfs, full devtmpfs and devpts, per-process procfs, sysfs over the canonical device model, cgroup2.

### D5. The geometric subsystems keep running, under Linux semantics

- Linux discretionary access control and capabilities are authoritative. `manifold_acl` becomes an additional restriction layer in the style of a Linux security module: audit-only by default (it records a certificate of every access it would have refused), enforcing only under an explicit policy. It never grants what Linux semantics deny.
- The sphere-based task picker chooses only among tasks Linux semantics make eligible: it honours scheduling class (SCHED_FIFO, SCHED_RR, SCHED_OTHER), nice, and cgroup2 `cpu.weight`.
- stratum, foliation, ManifoldFS and theorem status are reached through `/dev/seal`, `/sys/kernel/seal/`, and ManifoldFS as a mountable filesystem type.
- A host crate wraps `ml_engine/stratum.rs` unmodified so stratum runs on stock Linux without a kernel change.
- Lock order between `governor_epsilon()` and `scheduler_lock` is fixed before user processes run; the ACL may not take the scheduler lock.
- The contribution rule that requires the boot banner `All T1-T10 theorems VERIFIED; T1-T5 ACTIVE in runtime paths` is replaced by: every theorem line reports certified, refused (with its reason) or not checked, computed from the running kernel's state.

### D6. Progress is measured, and security is tested negatively

- Parity: Linux Test Project `syscalls` pass count, with unimplemented calls returning `-ENOSYS` (LTP reports those as not supported, not passed), plus the distribution ladder below.
- Security conformance: a negative suite in which each test fails when a check is not enforced: setuid, setgroups, capset, `PR_SET_NO_NEW_PRIVS`, seccomp filters on Linux numbers, file permissions for uid 1000, and `copy_from_user` on an unmapped address returning `-EFAULT` without halting.
- Every milestone gate executes in QEMU. Source-inspection tests are pre-checks, never gates.
- Value: each geometric subsystem reports a measured difference against stock Linux on a named workload. Reported with every release; not a blocking gate.

### D7. Anything Seal OS cannot build natively is ported

- `ports/` holds each port: upstream source pinned by hash, licence, patch count, and its QEMU gate. `PORTING.md` explains how an outside contributor adds one. Both land in M0, before the work they are meant to carry.
- First ports: ACPICA for ACPI AML (dual-licensed; used under its BSD-style option, as FreeBSD, NetBSD, Haiku and Fuchsia do) at M7; the LKL driver server at M7; a NetBSD rump server (BSD-licensed drivers and file systems) after the first LKL driver passes. A licence gate rejects files under the 4-clause BSD licence.
- A generic foreign-component contract is extracted only after two port hosts exist, from what they actually needed.
- A native implementation replaces a port only when it passes the same gate.

## Milestones

Each gate is a test that fails today.

| # | Scope | Gate |
|---|---|---|
| M0 | Substrate and safety: ring 3 executes; syscall entry does `swapgs`, switches to a per-CPU kernel stack before its first store, FMASK clears TF, DF, AC, IF and NT, EFER.SCE set; user faults kill the process, never the machine; exception fixups on user copies and no global lock held across one; kernel W^X; ext2 feature refusal; privilege check on package removal; governor/scheduler lock order; theorem lines from live state (T4 refused); CI green; `ports/`, `PORTING.md`, `CONTRIBUTING.md` rewritten | `chase_boot.sh usermode-seal`, `cameron_qemu_milestone.sh ring3-seal`, `test_cameron_syscall_entry.py`, `chase_boot.sh wx` (`wx_violations=0`), `chase_boot.sh ext4`, negative test: `write(fd, 0x1000, 1)` returns `-EFAULT` |
| M1 | Process model: per-process address spaces with VMAs, `MAP_FIXED`, file-backed mappings, reclaiming `munmap`, page cache, per-process fd tables with fork/exec/`O_CLOEXEC` semantics, wait queues, SIGSEGV delivery, reservations sized for a JVM heap | boot-executed tests for each; fork child reads parent fds |
| M2 | Linux ABI core: D1 renumbering, six-argument dispatch, auxv, `arch_prctl`, PT_TLS, `execve` with argv and envp, `exit_group`, `openat` family, `fstat`/`newfstatat`/`statx`, full `mmap`/`munmap`/`mprotect`, `brk`, `getdents64`, `futex`, `clone` threads, `rt_sig*`, `TCGETS`; `/dev/seal` | static glibc `hello` prints on serial; static busybox `sh` runs a script |
| M3 | Linux boot: setup header, EFI stub, command line, initramfs, `uname -r`, `seal-kernel` package layout | QEMU + OVMF + GRUB loads Seal OS as `linux` with an Alpine initramfs and reaches a busybox prompt |
| M4 | Dynamic userland without systemd: `ld.so` entry through PT_INTERP, AF_UNIX sockets, `poll`/`epoll`/`eventfd`, pipes and FIFOs, ptys, full signals, per-process `/proc` | Alpine (musl, OpenRC) boots to a login prompt from its own root; LTP `syscalls` count recorded |
| M5 | ext4 read and write with jbd2 | crash-consistency gate of D4 passes |
| M6 | systemd distributions: cgroup2, inotify, signalfd, timerfd, `name_to_handle_at`, netlink (uevent, rtnetlink), device model with uevents, namespaces (mount, pid, net, user), seccomp on Linux numbers; the distribution's own `kernel-install` and initrd generator | Debian, Ubuntu, Fedora and Arch cloud images each install `seal-kernel` with their own tools and boot to login under QEMU |
| M7 | Hardware: ACPICA port, ECAM, MSI/MSI-X, IOMMU, user-mode driver interface, LKL driver server | a Linux driver, unmodified, in a driver server passes its QEMU device model behind a virtual IOMMU |
| M8 | Bare metal | one reference machine boots a distribution on Seal OS to login |

## Reversibility

- Every milestone lands behind its own gate; the existing Seal image keeps booting.
- The distribution keeps its Linux kernel as the default boot entry until M6 passes on that distribution; Seal OS is an additional entry.
- No write to a foreign filesystem is enabled before the D4 crash-consistency gate passes.
- D1 is the one change that is hard to reverse. Its blast radius at the time of the switch is one embedded 208-byte ELF plus in-kernel callers.

## Decisions reserved for the owner

- Whether GPL-2.0 driver-server binaries are shipped in project images or built by users from the published source. Default: built by users.
- The reference machine for M8. Default: none chosen until M7 passes.
