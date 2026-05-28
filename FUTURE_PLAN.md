# FUTURE_PLAN.md — Epsilon-Hollow Roadmap

## Master Task Registry

This document is the single source of truth for all remaining work. Every task has a checkbox. Agents tick `[ ] -> [x]` as work completes. Each major area links to its design document, which contains the granular task breakdown.

Seal OS target is bare-metal Rust with Seal ABI, SealShell, and Aether-Lang. POSIX, Unix, Linux, libc, and GRUB compatibility goals are rejected unless explicitly labeled as legacy host interop.

---

## Phase 1: Boot to Shell (MVP)

> **Goal:** The kernel boots, initializes all subsystems, and drops into a working shell. All tests pass. This is the minimum viable Seal OS.

### 1.1 Boot Sequence
**Design doc:** [`docs/design/BOOT-SEQUENCE.md`](docs/design/BOOT-SEQUENCE.md)

- [ ] Stage 1: EFI pre-exit (memory map, framebuffer, ExitBootServices)
- [ ] Stage 2: Long mode setup (GDT, IDT stub, E820 parsing)
- [ ] Stage 3: Virtual memory bootstrap (identity map, higher half, page tables)
- [ ] Stage 3: Bitmap allocator init (mark usable, free count correct)
- [ ] Stage 3: Heap init (slab allocator online, Box::new succeeds)
- [ ] Stage 4: Subsystem init order verified (12 inits complete in sequence)
- [ ] Stage 5: Init process enters SealShell or Aether-Lang app host
- [ ] UEFI disk image generation
- [ ] QEMU runner scripts automated

### 1.2 Interrupt & Exception Handling
**Design doc:** [`docs/design/INTERRUPT-IDT.md`](docs/design/INTERRUPT-IDT.md)

- [ ] IDT with all 256 entries populated
- [ ] Exception handlers for vectors 0–20
- [ ] Page fault handler (COW, lazy alloc, stack growth, fixup tables)
- [ ] IO-APIC + Local APIC IRQ routing
- [ ] Interrupt Stack Table (IST1–IST4 allocated)
- [ ] `IrqGuard` RAII + spinlock integration
- [ ] Softirq framework + tasklets
- [ ] Local APIC timer fires at 1 kHz

### 1.3 Syscall Dispatcher
**Design doc:** [`docs/design/SYSCALL-DISPATCHER.md`](docs/design/SYSCALL-DISPATCHER.md)

- [ ] `syscall`/`sysret` ABI wired (STAR, LSTAR, SFMASK)
- [ ] Assembly entry/exit with `swapgs` mitigation
- [ ] `SyscallFrame` + dispatch table (512 entries)
- [ ] Tier 1 Seal ABI calls: read, write, open, close, exit, brk, mmap, fork, exec, wait, getpid, chdir, getcwd
- [ ] Tier 2 Seal ABI calls: stat, lseek, ioctl, pipe, dup, mkdir, rmdir, unlink, rename, manifold_query, theorem_status, teleport
- [ ] Tier 3 Seal ABI calls: signals, tasks, nanosleep, watchdog, package, WiFi/Bluetooth settings
- [ ] VDSO: clock_gettime, gettimeofday, getcpu, time
- [ ] MAC/audit integration on every syscall

### 1.4 VFS & Device Filesystem
**Design doc:** [`docs/design/VFS-DEVTMPFS.md`](docs/design/VFS-DEVTMPFS.md)

- [ ] `Inode`, `Dentry`, `VfsMount` structs
- [ ] `FileSystem` trait with Seal-native ops
- [ ] Path resolution (absolute, relative, symlink following, mount crossing)
- [ ] devtmpfs populated at boot (/dev/null, zero, full, random, urandom, tty, console, pts)
- [ ] `CharDevice` trait + implementations
- [ ] tmpfs for /tmp, /run, /var/tmp
- [ ] procfs minimal (/proc/cpuinfo, meminfo, uptime, version, self, sys)
- [ ] initrd / initramfs CPIO loader
- [ ] mount/umount syscalls

### 1.5 Shell & Userspace
**Design doc:** [`docs/design/SHELL-USERLAND.md`](docs/design/SHELL-USERLAND.md)

- [ ] Init system mounts essential FS, spawns getty
- [ ] SealShell native verbs (look, create, write, move, search, open, tasks, seal)
- [ ] P0 utilities mapped to Seal verbs and Aether-Lang modules
- [ ] Seal runtime bindings for allocation, files, windows, tasks, and theorem status
- [ ] ELF loader (PT_LOAD, BSS, stack, auxv)
- [ ] crt0.o startup code

### 1.6 Testing Harness
**Design doc:** [`docs/design/TEST-HARNESS.md`](docs/design/TEST-HARNESS.md)

- [ ] In-kernel `test_main()` with `TEST_PASS`/`TEST_FAIL` markers
- [ ] QEMU integration test runner (spawn, send, read, timeout)
- [ ] Host unit tests for all testable modules
- [ ] Property tests (proptest) for DirHash, InodeSlab, VoronoiCap, allocator
- [ ] Benchmark regression tests with baseline comparison
- [ ] CI pipeline: fmt → clippy → host tests → build → QEMU smoke → QEMU full → benchmarks → audit
- [ ] Code coverage tracking (> 80% for memory/fs/syscall)

---

## Phase 2: I/O & Persistence

> **Goal:** Real block device I/O, filesystem persistence, and buffer caching. The system can read/write files that survive reboot.

### 2.1 AHCI SATA Driver v2.0
**Design doc:** [`docs/design/AHCI-DRIVER.md`](docs/design/AHCI-DRIVER.md)

- [ ] Multi-port foundation (`AhciPort` + `AhciHba` structs)
- [ ] Per-port DMA buffers
- [ ] NCQ: 32-slot command queue, tagged commands, queue depth negotiation
- [ ] Interrupt-driven I/O (MSI/legacy IRQ, completion queue)
- [ ] Error recovery (timeout detection, port reset, retry with exponential backoff)
- [ ] Aether-Link integration (real LBA telemetry into 6D feature extractor)

### 2.2 Virtio-blk Driver
**Design doc:** [`docs/design/VIRTIO-BLK.md`](docs/design/VIRTIO-BLK.md)

- [ ] PCI discovery (vendor 0x1AF4, device 0x1001/0x1045)
- [ ] Feature negotiation + DRIVER_OK
- [ ] Split virtqueue implementation (desc/avail/used rings)
- [ ] Virtio-blk protocol (read/write/flush)
- [ ] `BlockDevice` trait abstraction
- [ ] Multi-queue support (v2)
- [ ] Interrupt-driven completions (v2)
- [ ] DMA allocator for contiguous physical buffers

### 2.3 Buffer Cache
**Design doc:** [`docs/design/BUFFER-CACHE.md`](docs/design/BUFFER-CACHE.md)

- [ ] `Buffer` struct with metadata + 4 KiB DMA data
- [ ] Hash table (1024 buckets) for fast lookup
- [ ] LRU eviction with clean/dirty handling
- [ ] Read path (cache hit/miss)
- [ ] Write path (mark dirty, periodic flush)
- [ ] Read-ahead (sequential detection, 128 KiB window)
- [ ] Write-back vs write-through modes
- [ ] mmap file backing (MAP_SHARED)

### 2.4 Ext2 Driver
**Design doc:** [`docs/design/EXT2-DRIVER.md`](docs/design/EXT2-DRIVER.md)

- [ ] Superblock parsing (magic 0xEF53, dynamic revision)
- [ ] Block group descriptor table
- [ ] Inode read/write (direct, indirect, double-indirect blocks)
- [ ] Directory entry iteration, create, remove
- [ ] File read/write/truncate
- [ ] Block allocation (bitmap scan, group selection)
- [ ] Inode allocation
- [ ] Mount integration (read-only + read-write)
- [ ] Legacy ext2 image interop verified without adopting Linux ABI

---

## Phase 3: Networking

> **Goal:** TCP/IP stack, socket API, and NIC drivers. The system can connect to the internet.

### 3.1 NIC Drivers
**Design doc:** [`docs/design/NETWORK-STACK.md`](docs/design/NETWORK-STACK.md)

- [ ] Virtio-net: RX/TX virtqueues, MAC, link status, checksum offload (v2)
- [ ] e1000: MMIO, RX/TX rings, EEPROM MAC, interrupts

### 3.2 Network Layer
- [ ] Ethernet frame parsing/transmission
- [ ] ARP cache (request, reply, timeout, eviction)
- [ ] IPv4 header parsing/construction + checksum
- [ ] ICMP echo request/reply (ping)
- [ ] Routing table + longest-prefix match

### 3.3 Transport Layer
- [ ] UDP: socket, bind, sendto, recvfrom, connect
- [ ] TCP: full state machine, 3-way handshake, teardown
- [ ] TCP: send/recv with ring buffers
- [ ] TCP: slow start, congestion avoidance, fast retransmit/recovery
- [ ] TCP: RTT estimation (RFC 6298), RTO backoff
- [ ] TCP: TIME_WAIT aging (60s)

### 3.4 Socket API & Services
- [ ] BSD socket syscalls (socket, bind, listen, accept, connect, send, recv, shutdown, setsockopt, getsockopt)
- [ ] Loopback interface (`lo`, 127.0.0.1)
- [ ] DHCP client (discover → offer → request → ack → renew → rebind)
- [ ] HTTP client capability (curl-equivalent)

---

## Phase 4: SMP & Concurrency

> **Goal:** Multi-core support with proper synchronization and load balancing.

**Design doc:** [`docs/design/smp_apic.md`](docs/design/smp_apic.md)

- [ ] ACPI MADT parsing for APIC IDs
- [ ] Per-CPU data via `gsbase` (`PerCpu` struct)
- [ ] AP bootstrap: INIT-SIPI-SIPI → real mode → long mode
- [ ] AP sets up gsbase, IDT, enters scheduler
- [ ] I/O APIC routing (timer → all CPUs, device IRQs → round-robin)
- [ ] Per-CPU scheduler runqueues (`ManifoldScheduler` per core)
- [ ] Load balancing (steal task every 100ms if >2× imbalance)
- [ ] Preemptive reschedule IPI (`apic_send_ipi(target, RESCHEDULE_VECTOR)`)
- [ ] Spinlocks → ticket locks or MCS locks
- [ ] RCU for read-mostly data (VFS mount table, device list)
- [ ] Seqlocks for jiffies/ticks
- [ ] TLB shootdown IPI on `unmap_page()`

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
