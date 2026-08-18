# FUTURE_PLAN.md — Epsilon-Hollow Roadmap

## Master Task Registry

**Code-verified pass (2026-08-18):** 46/126 boxes ticked. Every `[x]` carries an inline `path:line` citation to the implementing code; every unticked item is either unbuilt or partial with a stated gap (annotated inline where a partial implementation exists).

This document is the single source of truth for all remaining work. Every task has a checkbox. Agents tick `[ ] -> [x]` as work completes. Each major area links to its design document, which contains the granular task breakdown.

Seal OS target is bare-metal Rust with Seal ABI, SealShell, and Aether-Lang. POSIX, Unix, Linux, libc, and GRUB compatibility goals are rejected unless explicitly labeled as legacy host interop.

---

## Phase 1: Boot to Shell (MVP)

> **Goal:** The kernel boots, initializes all subsystems, and drops into a working shell. All tests pass. This is the minimum viable Seal OS.

### 1.1 Boot Sequence
**Design doc:** [`docs/design/BOOT-SEQUENCE.md`](docs/design/BOOT-SEQUENCE.md)

- [x] Stage 1: EFI pre-exit (memory map, framebuffer, ExitBootServices) — `kernel/seal-os/src/boot/uefi_entry.rs:103`
- [ ] Stage 2: Long mode setup (GDT, IDT stub, E820 parsing) — partial: GDT built, but memory map comes from UEFI GetMemoryMap, no BIOS E820 parsing (`kernel/seal-os/src/boot/uefi_entry.rs:127`)
- [x] Stage 3: Virtual memory bootstrap (identity map, higher half, page tables) — `kernel/seal-os/src/memory/virt.rs:71`
- [x] Stage 3: Bitmap allocator init (mark usable, free count correct) — `kernel/seal-os/src/memory/phys.rs:389`
- [x] Stage 3: Heap init (slab allocator online, Box::new succeeds) — `kernel/seal-os/src/memory/heap.rs:20`
- [x] Stage 4: Subsystem init order verified (12 inits complete in sequence) — `kernel/seal-os/src/lib.rs:266`
- [ ] Stage 5: Init process enters SealShell or Aether-Lang app host — partial: `/bin/init` is absent from the disk image, so the kernel falls back to the in-kernel desktop/SealShell instead of spawning a userspace init (`kernel/seal-os/src/lib.rs:591`)
- [x] UEFI disk image generation — `kernel/seal-mkimage/src/main.rs:28`
- [x] QEMU runner scripts automated — `kernel/seal-os/run-qemu.sh:1`

### 1.2 Interrupt & Exception Handling
**Design doc:** [`docs/design/INTERRUPT-IDT.md`](docs/design/INTERRUPT-IDT.md)

- [ ] IDT with all 256 entries populated — partial: only ~13 vectors are registered (exceptions, IRQs, IPIs), the rest stay default (`kernel/seal-os/src/drivers/interrupts.rs:104`)
- [ ] Exception handlers for vectors 0–20 — partial: only 7 of 21 exception vectors have handlers (breakpoint, double-fault, invalid-opcode, segment-not-present, stack-segment-fault, GP-fault, page-fault) (`kernel/seal-os/src/drivers/interrupts.rs:106`)
- [ ] Page fault handler (COW, lazy alloc, stack growth, fixup tables) — partial: COW and demand-paging/swap-in are real, no stack-growth or fixup-table handling found (`kernel/seal-os/src/drivers/interrupts.rs:497`)
- [x] IO-APIC + Local APIC IRQ routing — `kernel/seal-os/src/drivers/interrupts.rs:252`
- [ ] Interrupt Stack Table (IST1–IST4 allocated) — partial: only IST index 0 (double fault) is set up, not four stacks (`kernel/seal-os/src/memory/gdt.rs:131`)
- [ ] `IrqGuard` RAII + spinlock integration
- [ ] Softirq framework + tasklets
- [x] Local APIC timer fires at 1 kHz — `kernel/seal-os/src/drivers/apic.rs:208`

### 1.3 Syscall Dispatcher
**Design doc:** [`docs/design/SYSCALL-DISPATCHER.md`](docs/design/SYSCALL-DISPATCHER.md)

- [x] `syscall`/`sysret` ABI wired (STAR, LSTAR, SFMASK) — `kernel/seal-os/src/process/userspace.rs:209`
- [ ] Assembly entry/exit with `swapgs` mitigation — partial: `syscall_entry` exists but no `swapgs` instruction is in the entry path (`kernel/seal-os/src/process/userspace.rs:130`)
- [ ] `SyscallFrame` + dispatch table (512 entries) — partial: `SyscallFrame` + `dispatch()` are real, but it is a `match` over ~69 syscalls, not a fixed 512-entry table (`kernel/seal-os/src/syscall/table.rs:680`)
- [x] Tier 1 Seal ABI calls: read, write, open, close, exit, brk, mmap, fork, exec, wait, getpid, chdir, getcwd — `kernel/seal-os/src/syscall/table.rs:709`
- [x] Tier 2 Seal ABI calls: stat, lseek, ioctl, pipe, dup, mkdir, rmdir, unlink, rename, manifold_query, theorem_status, teleport — `kernel/seal-os/src/syscall/table.rs:1059`
- [ ] Tier 3 Seal ABI calls: signals, tasks, nanosleep, watchdog, package, WiFi/Bluetooth settings — partial: signals/nanosleep/watchdog/WiFi dispatch real, no package-management syscall (`kernel/seal-os/src/syscall/table.rs:1394`)
- [ ] VDSO: clock_gettime, gettimeofday, getcpu, time
- [ ] MAC/audit integration on every syscall — partial: `audit_log` is only called for open/sudo/execve/setuid, not every syscall (`kernel/seal-os/src/syscall/table.rs:1680`)

### 1.4 VFS & Device Filesystem
**Design doc:** [`docs/design/VFS-DEVTMPFS.md`](docs/design/VFS-DEVTMPFS.md)

- [ ] `Inode`, `Dentry`, `VfsMount` structs — partial: `Inode` exists, no `Dentry` or `VfsMount` struct found (`kernel/seal-os/src/fs/ext2.rs:93`)
- [x] `FileSystem` trait with Seal-native ops — `kernel/seal-os/src/fs/vfs.rs:78`
- [x] Path resolution (absolute, relative, symlink following, mount crossing) — `kernel/seal-os/src/fs/vfs.rs:290`
- [ ] devtmpfs populated at boot (/dev/null, zero, full, random, urandom, tty, console, pts) — partial: only null/zero/random/console implemented, no full/urandom/tty/pts (`kernel/seal-os/src/fs/devtmpfs.rs:77`)
- [ ] `CharDevice` trait + implementations
- [ ] tmpfs for /tmp, /run, /var/tmp
- [x] procfs minimal (/proc/cpuinfo, meminfo, uptime, version, self, sys) — `kernel/seal-os/src/fs/mod.rs:78`
- [ ] initrd / initramfs CPIO loader
- [ ] mount/umount syscalls

### 1.5 Shell & Userspace
**Design doc:** [`docs/design/SHELL-USERLAND.md`](docs/design/SHELL-USERLAND.md)

- [ ] Init system mounts essential FS, spawns getty — partial: essential FS mounting happens at boot, but there is no getty and no true userspace init process (`kernel/seal-os/src/fs/mod.rs:78`)
- [x] SealShell native verbs (look, create, write, move, search, open, tasks, seal) — `kernel/seal-os/src/apps/shell.rs:1105`
- [ ] P0 utilities mapped to Seal verbs and Aether-Lang modules
- [ ] Seal runtime bindings for allocation, files, windows, tasks, and theorem status
- [ ] ELF loader (PT_LOAD, BSS, stack, auxv) — partial: PT_LOAD segment loading and stack setup are real, no auxv vector is constructed (`kernel/seal-os/src/process/elf.rs:12`)
- [ ] crt0.o startup code

### 1.6 Testing Harness
**Design doc:** [`docs/design/TEST-HARNESS.md`](docs/design/TEST-HARNESS.md)

- [x] In-kernel `test_main()` with `TEST_PASS`/`TEST_FAIL` markers — `kernel/seal-os/src/testing/harness.rs:62`
- [x] QEMU integration test runner (spawn, send, read, timeout) — `kernel/seal-os/tests/qemu_runner.rs:42`
- [ ] Host unit tests for all testable modules — partial: `kernel/seal-os` is excluded from `cargo test --workspace` (`Cargo.toml:21`), so its 552 `#[cfg(test)]` cases only run via the QEMU test-mode harness, never as host tests
- [ ] Property tests (proptest) for DirHash, InodeSlab, VoronoiCap, allocator — partial: only DirHash has a proptest, InodeSlab/VoronoiCap/allocator do not (`kernel/seal-os/tests/prop_dirhash.rs:1`)
- [x] Benchmark regression tests with baseline comparison — `.github/workflows/ci.yml:95`
- [x] CI pipeline: fmt → clippy → host tests → build → QEMU smoke → QEMU full → benchmarks → audit — `.github/workflows/ci.yml:1`
- [ ] Code coverage tracking (> 80% for memory/fs/syscall)

---

## Phase 2: I/O & Persistence

> **Goal:** Real block device I/O, filesystem persistence, and buffer caching. The system can read/write files that survive reboot.

### 2.1 AHCI SATA Driver v2.0
**Design doc:** [`docs/design/AHCI-DRIVER.md`](docs/design/AHCI-DRIVER.md)

- [ ] Multi-port foundation (`AhciPort` + `AhciHba` structs) — partial: `init()` only ever inits one port, `AhciHba.ports` stays empty (`kernel/seal-os/src/drivers/block/ahci.rs:870`)
- [x] Per-port DMA buffers — `kernel/seal-os/src/drivers/block/ahci.rs:788`
- [ ] NCQ: 32-slot command queue, tagged commands, queue depth negotiation — partial: `ncq_supported` is a hardcoded false, NCQ path is dead code, no depth negotiation (`kernel/seal-os/src/drivers/block/ahci.rs:830`)
- [ ] Interrupt-driven I/O (MSI/legacy IRQ, completion queue)
- [ ] Error recovery (timeout detection, port reset, retry with exponential backoff) — partial: timeout + port-reset + 3 retries real, but zero backoff delay (`kernel/seal-os/src/drivers/block/ahci.rs:391`)
- [ ] Aether-Link integration (real LBA telemetry into 6D feature extractor)

### 2.2 Virtio-blk Driver
**Design doc:** [`docs/design/VIRTIO-BLK.md`](docs/design/VIRTIO-BLK.md)

- [x] PCI discovery (vendor 0x1AF4, device 0x1001/0x1045) — `kernel/seal-os/src/drivers/block/virtio_blk.rs:218`
- [x] Feature negotiation + DRIVER_OK — `kernel/seal-os/src/drivers/block/virtio_blk.rs:239`
- [x] Split virtqueue implementation (desc/avail/used rings) — `kernel/seal-os/src/drivers/block/virtio_blk.rs:16`
- [ ] Virtio-blk protocol (read/write/flush) — partial: `flush()` is a no-op stub (`kernel/seal-os/src/drivers/block/virtio_blk.rs:375`)
- [ ] `BlockDevice` trait abstraction — partial: local trait, never registered via `register_block_device`, not unified with the block layer (`kernel/seal-os/src/drivers/block/virtio_blk.rs:7`)
- [ ] Multi-queue support (v2)
- [ ] Interrupt-driven completions (v2)
- [ ] DMA allocator for contiguous physical buffers — partial: plain `alloc_zeroed`, no contiguous-physical allocator (`kernel/seal-os/src/drivers/block/virtio_blk.rs:259`)

### 2.3 Buffer Cache
**Design doc:** [`docs/design/BUFFER-CACHE.md`](docs/design/BUFFER-CACHE.md)

- [ ] `Buffer` struct with metadata + 4 KiB DMA data — partial: payload is a heap `Vec` sized by `block_size`, not DMA, not fixed 4KiB (`kernel/seal-os/src/fs/buffer_cache.rs:13`)
- [ ] Hash table (1024 buckets) for fast lookup
- [x] LRU eviction with clean/dirty handling — `kernel/seal-os/src/fs/buffer_cache.rs:136`
- [x] Read path (cache hit/miss) — `kernel/seal-os/src/fs/buffer_cache.rs:72`
- [ ] Write path (mark dirty, periodic flush) — partial: mark-dirty real, but no periodic/timer flush anywhere (`kernel/seal-os/src/fs/buffer_cache.rs:89`)
- [ ] Read-ahead (sequential detection, 128 KiB window)
- [ ] Write-back vs write-through modes
- [ ] mmap file backing (MAP_SHARED)

### 2.4 Ext2 Driver
**Design doc:** [`docs/design/EXT2-DRIVER.md`](docs/design/EXT2-DRIVER.md)

- [x] Superblock parsing (magic 0xEF53, dynamic revision) — `kernel/seal-os/src/fs/ext2.rs:308`
- [x] Block group descriptor table — `kernel/seal-os/src/fs/ext2.rs:350`
- [x] Inode read/write (direct, indirect, double-indirect blocks) — `kernel/seal-os/src/fs/ext2.rs:684`
- [x] Directory entry iteration, create, remove — `kernel/seal-os/src/fs/ext2.rs:1179`
- [ ] File read/write/truncate — partial: no truncate exists; only whole-file unlink frees blocks (`kernel/seal-os/src/fs/ext2.rs:1510`)
- [x] Block allocation (bitmap scan, group selection) — `kernel/seal-os/src/fs/ext2.rs:943`
- [x] Inode allocation — `kernel/seal-os/src/fs/ext2.rs:1021`
- [ ] Mount integration (read-only + read-write) — partial: no read-only mount mode/flag exists (`kernel/seal-os/src/fs/mod.rs:44`)
- [ ] Legacy ext2 image interop verified without adopting Linux ABI

---

## Phase 3: Networking

> **Goal:** TCP/IP stack, socket API, and NIC drivers. The system can connect to the internet.

### 3.1 NIC Drivers
**Design doc:** [`docs/design/NETWORK-STACK.md`](docs/design/NETWORK-STACK.md)

- [ ] Virtio-net: RX/TX virtqueues, MAC, link status, checksum offload (v2) — partial: RX ring never drained (no receive/poll_rx), no checksum offload, probe-only per `drivers/net/mod.rs:62` (`kernel/seal-os/src/drivers/net/virtio_net.rs:284`)
- [ ] e1000: MMIO, RX/TX rings, EEPROM MAC, interrupts — partial: MMIO/rings/send/recv real and wired in, but interrupts masked off at REG_IMC with no handler; MAC comes from RAL0/RAH0, not an EEPROM read (`kernel/seal-os/src/drivers/net/e1000.rs:163`)

### 3.2 Network Layer
- [x] Ethernet frame parsing/transmission — `kernel/seal-os/src/net/mod.rs:131`
- [ ] ARP cache (request, reply, timeout, eviction) — partial: expired entries filtered on lookup but never removed from the Vec (`kernel/seal-os/src/net/arp.rs:44`)
- [x] IPv4 header parsing/construction + checksum — `kernel/seal-os/src/net/ipv4.rs:23`
- [x] ICMP echo request/reply (ping) — `kernel/seal-os/src/net/icmp.rs:19`
- [ ] Routing table + longest-prefix match

### 3.3 Transport Layer
- [x] UDP: socket, bind, sendto, recvfrom, connect — `kernel/seal-os/src/net/udp.rs:19`
- [x] TCP: full state machine, 3-way handshake, teardown — `kernel/seal-os/src/net/tcp.rs:10`
- [ ] TCP: send/recv with ring buffers — partial: growable Vec, not a fixed-capacity ring (`kernel/seal-os/src/net/tcp.rs:257`)
- [ ] TCP: slow start, congestion avoidance, fast retransmit/recovery
- [ ] TCP: RTT estimation (RFC 6298), RTO backoff — partial: backoff real and tested; no SRTT/RTTVAR sampling, RTO is a fixed 1000-tick constant (`kernel/seal-os/src/net/tcp.rs:456`)
- [x] TCP: TIME_WAIT aging (60s) — `kernel/seal-os/src/net/tcp.rs:147`

### 3.4 Socket API & Services
- [ ] BSD socket syscalls (socket, bind, listen, accept, connect, send, recv, shutdown, setsockopt, getsockopt)
- [ ] Loopback interface (`lo`, 127.0.0.1) — partial: only the single host address 127.0.0.1, not 127.0.0.0/8, no named `lo` interface (`kernel/seal-os/src/net/ipv4.rs:181`)
- [ ] DHCP client (discover → offer → request → ack → renew → rebind) — partial: discover/offer/request/ack real; no lease timer, no Renewing/Rebinding states, `poll()` is inert once Bound (`kernel/seal-os/src/net/dhcp.rs:171`)
- [x] HTTP client capability (curl-equivalent) — `kernel/seal-os/src/drivers/net/http.rs:77`

---

## Phase 4: SMP & Concurrency

> **Goal:** Multi-core support with proper synchronization and load balancing.

**Design doc:** [`docs/design/smp_apic.md`](docs/design/smp_apic.md)

- [x] ACPI MADT parsing for APIC IDs — `kernel/seal-os/src/drivers/acpi/madt.rs:80`
- [x] Per-CPU data via `gsbase` (`PerCpu` struct) — `kernel/seal-os/src/cpu/mod.rs:27`
- [x] AP bootstrap: INIT-SIPI-SIPI → real mode → long mode — `kernel/seal-os/src/boot/ap_trampoline.rs:46`
- [x] AP sets up gsbase, IDT, enters scheduler — `kernel/seal-os/src/cpu/smp.rs:121`
- [ ] I/O APIC routing (timer → all CPUs, device IRQs → round-robin) — partial: `redirect_irq` real but only used to pin IRQ1/IRQ12 to the BSP; no round-robin (`kernel/seal-os/src/drivers/interrupts.rs:252`)
- [x] Per-CPU scheduler runqueues (`ManifoldScheduler` per core) — `kernel/seal-os/src/process/scheduler.rs:222`
- [ ] Load balancing (steal task every 100ms if >2× imbalance) — partial: `steal_task` is real but idle-triggered inside `schedule()`; no 100ms timer, no 2x imbalance check (`kernel/seal-os/src/process/scheduler.rs:469`)
- [x] Preemptive reschedule IPI (`apic_send_ipi(target, RESCHEDULE_VECTOR)`) — `kernel/seal-os/src/process/scheduler.rs:1766`
- [ ] Spinlocks → ticket locks or MCS locks — partial: `TicketLock` exists and is tested but has zero callers outside `sync/`; ~80 sites still use `spin::Mutex` (`kernel/seal-os/src/sync/ticket_lock.rs:1`)
- [ ] RCU for read-mostly data (VFS mount table, device list)
- [ ] Seqlocks for jiffies/ticks — partial: `SeqLock` exists and is tested but `ticks` is a plain AtomicU64, seqlock unused (`kernel/seal-os/src/drivers/interrupts.rs:360`)
- [x] TLB shootdown IPI on `unmap_page()` — `kernel/seal-os/src/sync/tlb.rs:35`

---

## Phase 5: Performance Optimizations

> **Goal:** Every subsystem is benchmarked and optimized. No regressions.

**Design doc:** [`docs/design/PERFORMANCE.md`](docs/design/PERFORMANCE.md)

### Memory
- [ ] Per-CPU slab caches (< 5% slowdown at 4 CPUs)
- [ ] Slab coloring (-20% L1 misses)
- [ ] Huge pages (2 MiB / 1 GiB) — 50% fewer TLB misses
- [ ] Page reclaim / swap — survive 2×RAM pressure

### Scheduling
- [ ] CFS red-black tree (fairness ratio < 1.1)
- [ ] Load balancing (all CPUs > 90% under load)
- [ ] Preemption (< 1 µs wake-to-schedule)

### I/O
- [ ] AHCI NCQ (2× throughput on rotational)
- [ ] Block I/O scheduler (30% improvement via merging)
- [ ] Read-ahead (5× sequential read improvement)

### Network
- [ ] Zero-copy networking (90% line rate)
- [ ] TSO/LRO offload (v2)
- [ ] IRQ coalescing (< 10k IRQ/s at line rate)

### Locking
- [ ] MCS locks (max wait < 2× mean)
- [ ] RCU for VFS (lock-free reads)
- [ ] Seqlocks for ticks (2 loads + compare)

### Build
- [ ] LTO + PGO enabled
- [ ] Kernel < 2 MiB compressed
- [ ] Boot time < 500 ms to shell

---

## Phase 6: The Probabilistic Topology *(The Grand Goal)*

User space is not a flat address space. It is a **topology being observed by the machine** — and the machine does not observe it directly. It observes it through projection.

The ultimate arc of Epsilon-Hollow is to make the computer itself **probabilistic** — not by bolting sampling layers onto a deterministic kernel, but by making probability emerge naturally from the projected topology of user-space state. The theorems T1–T5 are not optimization heuristics. They are the scaffolding of a new kind of computation where the kernel does not *decide* — it *collapses*.

**What this means concretely:**

- **User space as observed manifold**: Every process, every file descriptor, every page table entry is a point in a high-dimensional state manifold. The kernel does not traverse this space linearly. It projects it — through the same JL-projected S² encoding that powers ManifoldFS — into a topology the computer can "see."

- **Probability through projection, not sampling**: Traditional probabilistic OS work adds Bayesian networks or MCMC on top of deterministic schedulers. That is backwards. In the projected topology, probability is not an extra layer — it is the *geometry of uncertainty* that arises naturally when you project a high-dimensional state space onto a curved manifold and ask "what is the measure of this region?" The Voronoi cells become confidence regions. The spectral contraction operator becomes a belief-update rule. The governor becomes a posterior-adaptive regulator.

- **The computer observes, therefore it is probabilistic**: When the kernel maps a user-space operation into its projected topology, it is not computing a deterministic path. It is computing a **distribution over paths** — and then collapsing that distribution via the same topological surgery that makes teleportation O(1). The collapse is the syscall. The distribution before collapse is the *possibility space*.

- **From T1–T5 to T∞**: T1 (Voronoi) gives us spatial indexing on the observation manifold. T2 (Spectral Contraction) gives us belief propagation. T4 (Governor) gives us posterior adaptation. The missing piece — the one that makes the machine probabilistic — is the **measure on the projected space**. When a user process requests a resource, the kernel does not check a boolean permission. It computes the **measure of that request in the projected topology of the user's observed state** — and if the measure exceeds a curvature threshold, the request is granted. The kernel is not enforcing policy. It is measuring topology.

**The end state:**

A kernel where every decision — page allocation, scheduling quantum, I/O prefetch, file teleport — is the collapse of a probability distribution that lives on the projected topology of user-space state. The machine does not "use" probability. The machine *is* probability, because probability is the only thing that geometry can be when it is observed through projection.

This is not a feature to implement. This is the shape the kernel grows into once T1–T10 are fully wired and the topology engine becomes the primary sense organ of the OS.

---

## Appendix: Design Document Index

| Document | Phase | Status |
|---|---|---|
| [`MANIFOLDFS-O1-DESIGN.md`](docs/MANIFOLDFS-O1-DESIGN.md) | 1 | [x] Complete (docs created) |
| [`AHCI-DRIVER.md`](docs/design/AHCI-DRIVER.md) | 2 | [x] Complete (docs created) |
| [`BOOT-SEQUENCE.md`](docs/design/BOOT-SEQUENCE.md) | 1 | [x] Complete (docs created) |
| [`INTERRUPT-IDT.md`](docs/design/INTERRUPT-IDT.md) | 1 | [x] Complete (docs created) |
| [`SYSCALL-DISPATCHER.md`](docs/design/SYSCALL-DISPATCHER.md) | 1 | [x] Complete (docs created) |
| [`VFS-DEVTMPFS.md`](docs/design/VFS-DEVTMPFS.md) | 1 | [x] Complete (docs created) |
| [`SHELL-USERLAND.md`](docs/design/SHELL-USERLAND.md) | 1 | [x] Complete (docs created) |
| [`TEST-HARNESS.md`](docs/design/TEST-HARNESS.md) | 1 | [x] Complete (docs created) |
| [`VIRTIO-BLK.md`](docs/design/VIRTIO-BLK.md) | 2 | [x] Complete (docs created) |
| [`BUFFER-CACHE.md`](docs/design/BUFFER-CACHE.md) | 2 | [x] Complete (docs created) |
| [`EXT2-DRIVER.md`](docs/design/EXT2-DRIVER.md) | 2 | [x] Complete (docs created) |
| [`NETWORK-STACK.md`](docs/design/NETWORK-STACK.md) | 3 | [x] Complete (docs created) |
| [`smp_apic.md`](docs/design/smp_apic.md) | 4 | [x] Complete (docs created) |
| [`PERFORMANCE.md`](docs/design/PERFORMANCE.md) | 3–5 | [x] Complete (docs created) |

---

*Master roadmap produced by Phase X Planning*
*2026-05-19*
