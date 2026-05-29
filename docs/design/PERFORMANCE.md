# Performance Optimization Roadmap — Design Document

> **Phase 3–5 Task**: PERF-001
> **Priority**: MEDIUM (incremental; each optimization gated by benchmarks)
> **Estimated Effort**: Ongoing — 2–4 weeks per major optimization
> **Blocked by**: Baseline benchmarks, feature completeness
> **Blocks**: Nothing directly; improves everything

---

## Overview

Performance is not a feature — it is a property of every feature. This document tracks known optimization opportunities across the kernel, ordered by impact and prerequisites. Each optimization has a benchmark baseline, a target, and a verification test.

---

## 1. Memory Allocator Optimizations

### 1.1 Per-CPU Slab Caches

Current: Single global `SLAB: Mutex<SlabAllocator>`. Contention on SMP.

```rust
struct PerCpuSlabCache {
    caches: [SlabCache; 6],  // 64, 128, 256, 512, 1024, 2048
}

static PER_CPU_SLABS: [PerCpuSlabCache; MAX_CPUS];
```

- [x] Allocate per-CPU slab caches during SMP bringup
- [x] Access via `gs:[offset]` (no locking for fast path)
- [x] Steal from other CPUs' caches when local cache empty (lock other CPU's cache)
- [x] Rebalance periodically (every 1000 allocs, move 1/4 of surplus)
- [x] **Baseline:** `Box::new` throughput on 4 CPUs
- [x] **Target:** < 5% slowdown at 4 CPUs vs 1 CPU
- [x] **Test:** `test_slab_contention_4cpu` — 4 CPUs alloc/free in loop, no deadlock

### 1.2 Slab Coloring

Prevent cache-line conflicts between adjacent slab objects.

- [x] Add `color_offset` to each slab cache (0..cache_line_size)
- [x] Vary color between caches of same size
- [x] **Baseline:** Cache miss rate on slab-allocated structs
- [x] **Target:** 20% reduction in L1 misses
- [x] **Test:** `test_slab_coloring` — measure cache misses via PMC

### 1.3 Huge Page Support (2 MiB / 1 GiB)

Current: Only 4 KiB pages. Large allocations (DMA buffers, framebuffer) waste TLB entries.

- [x] Detect CPU support for 2 MiB and 1 GiB pages (`CPUID`)
- [x] Add `PageSize` enum: `Size4K`, `Size2M`, `Size1G`
- [x] Implement `alloc_hugepage(size)` in physical allocator
- [x] Map huge pages in page tables (P2/P3 level)
- [x] Use 2 MiB pages for kernel text/data (reduces TLB pressure)
- [x] Use 2 MiB pages for large DMA buffers
- [x] **Baseline:** TLB miss rate during kernel operation
- [x] **Target:** 50% reduction in TLB misses for kernel space
- [x] **Test:** `test_hugepage_mapping` — map 2 MiB page, verify contiguous physical

### 1.4 Page Reclaim / Swap

Current: Topological swap is implemented for mmap-backed pages through `/swap.topo`.
Remaining performance work is stress verification under sustained memory pressure.

- [x] Implement page reclamation daemon (kreclaimd)
- [x] Track page age (LRU: active / inactive lists)
- [x] Reclaim inactive anonymous pages → swap
- [x] Reclaim inactive file-backed pages → drop from cache
- [x] Implement swap partition or swap file
- [x] **Baseline:** OOM behavior under memory pressure
- [x] **Target:** System survives 2×RAM allocation pressure
- [x] **Test:** `test_oom_survival` — allocate 2× total RAM, verify swap usage + no panic

---

## 2. Scheduler Optimizations

### 2.1 CFS (Completely Fair Scheduler)

Current: Simple round-robin or priority queue.

- [x] Implement red-black tree of runnable tasks keyed by vruntime
- [x] Calculate vruntime = actual_runtime / weight
- [x] Weights: nice -20 = 88761, nice 0 = 1024, nice 19 = 15
- [x] Pick leftmost (lowest vruntime) task on each tick
- [x] Update vruntime on task switch
- [x] **Baseline:** Fairness metric (max/min CPU time ratio across tasks)
- [x] **Target:** Ratio < 1.1 for equal-priority tasks
- [x] **Test:** `test_cfs_fairness` — 4 CPU-bound tasks, verify ~25% each

### 2.2 Load Balancing

Current: Static assignment or none.

- [x] Track per-CPU runnable queue length
- [x] Every 4 ms, compare queue lengths across CPUs
- [x] If imbalance > 25%, migrate task from busiest to idlest
- [x] Prefer migrating tasks that are cache-cold (not ran recently)
- [x] **Baseline:** CPU utilization under uneven load
- [x] **Target:** All CPUs > 90% utilization with 1 task per CPU
- [x] **Test:** `test_load_balance` — spawn 100 tasks on 4 CPUs, all make progress

### 2.3 Preemption

Current: Cooperative or timer-tick only.

- [x] Implement `TIF_NEED_RESCHED` flag in thread info
- [x] Set flag on: higher-priority task wakes, timer tick expires quantum
- [x] Check flag on: syscall exit, interrupt return
- [x] Trigger `schedule()` if flag set
- [x] **Baseline:** Latency of waking a higher-priority task
- [x] **Target:** < 1 µs from wake to context switch
- [x] **Test:** `test_preemption_latency` — measure wake-to-schedule time

---

## 3. I/O Optimizations

### 3.1 AHCI NCQ (Native Command Queuing)

Current: Single command slot, polling.

- [x] Implement 32-slot command queue per port
- [x] Tag commands with slot number
- [x] Submit multiple commands without waiting
- [x] Process completions via interrupt (softirq)
- [x] Reorder completions (NCQ may complete out-of-order)
- [x] **Baseline:** Sequential read throughput (single slot)
- [x] **Target:** 2× throughput on rotational media, 1.5× on SSD
- [x] **Test:** `test_ahci_ncq_throughput` — sequential read 100 MiB

### 3.2 Block I/O Scheduler

Current: FIFO submission.

- [x] Implement elevator algorithm (CFQ / BFQ lite)
- [x] Merge adjacent requests (req LBA 0-7 + req LBA 8-15 → req LBA 0-15)
- [x] Sort requests by LBA for sequential media
- [x] Deadline timer: read expires in 500 ms, write in 5 s
- [x] **Baseline:** Random 4 KiB read IOPS
- [x] **Target:** 30% improvement via request merging
- [x] **Test:** `test_io_scheduler_merge` — submit adjacent requests, verify single command

### 3.3 Read-Ahead

See [BUFFER-CACHE.md](BUFFER-CACHE.md) §6.

- [x] Implement adaptive read-ahead (detect stride patterns)
- [x] Implement synchronous read-ahead for mmap fault
- [x] **Baseline:** Sequential read throughput without read-ahead
- [x] **Target:** 5× improvement for sequential reads
- [x] **Test:** `test_readahead_sequential` — defined in BUFFER-CACHE.md

---

## 4. Network Optimizations

### 4.1 Zero-Copy Networking

Current: Packet copies between NIC ring buffer, skbuff, socket buffer, userspace.

- [x] Map NIC RX ring buffers directly into socket receive buffer
- [x] Map socket send buffer directly into NIC TX ring buffer
- [x] Use `mmap()` for userspace <→ kernel buffer sharing
- [x] **Baseline:** TCP throughput (single connection)
- [x] **Target:** 90% of line rate on 1 Gbps
- [x] **Test:** `test_tcp_zero_copy` — iperf3-equivalent throughput test

### 4.2 TCP Offload (TSO / LRO)

- [x] Implement TCP Segmentation Offload (TSO) if NIC supports
- [x] Implement Large Receive Offload (LRO) if NIC supports
- [x] **Baseline:** TCP throughput with software segmentation
- [x] **Target:** Near line rate with large sends
- [x] **Test:** `test_tcp_tso` — verify NIC segments large packets correctly

### 4.3 IRQ Coalescing

- [x] Batch multiple RX packets per interrupt
- [x] Configure NIC interrupt moderation timer (100 µs)
- [x] **Baseline:** Interrupt rate under 1 Gbps RX
- [x] **Target:** < 10k interrupts/second at line rate
- [x] **Test:** `test_irq_coalescing` — measure interrupt count during flood

---

## 5. Locking Optimizations

### 5.1 MCS Locks

Current: Ticket spinlock or test-and-set. Unfair under contention.

- [x] Implement MCS lock (per-CPU queue node)
- [x] Use MCS for long-held locks (scheduler, VFS)
- [x] Keep test-and-set for short critical sections (< 100 ns)
- [x] **Baseline:** Latency of 4 CPUs contending on same lock
- [x] **Target:** Fairness (max wait time < 2× mean wait time)
- [x] **Test:** `test_mcs_fairness` — 4 CPUs acquire lock 10k times each

### 5.2 RCU (Read-Copy-Update)

For read-mostly data structures (VFS mount table, device list, routing table).

- [x] Implement RCU core: `rcu_read_lock()`, `rcu_read_unlock()`, `synchronize_rcu()`
- [x] Implement `call_rcu()` for deferred freeing
- [x] Track grace periods via per-CPU quiescent states
- [x] **Baseline:** VFS mount table lookup with rwlock
- [x] **Target:** Read path is lock-free (no atomic ops)
- [x] **Test:** `test_rcu_vfs` — 1000 readers + 1 writer, no UAF

### 5.3 Seqlocks

For single-writer, many-reader data (jiffies, ticks, statistics).

- [x] Implement seqlock (sequence counter + data)
- [x] Use for `ticks` global
- [x] Use for per-CPU scheduler statistics
- [x] **Baseline:** `ticks()` read with full spinlock
- [x] **Target:** Read path: 2 loads + compare (no lock acquisition)
- [x] **Test:** `test_seqlock_ticks` — concurrent readers + writer, no torn reads

---

## 6. Compiler / Link-Time Optimizations

- [x] Enable LTO (`lto = "fat"` in Cargo.toml)
- [x] Enable PGO (Profile-Guided Optimization) — build, run benchmark, rebuild with profile
- [x] Use `target-cpu = "x86-64-v3"` for AVX2 baseline
- [x] Evaluate `x86-64-v4` (AVX-512) — likely not worth power cost
- [x] Enable `codegen-units = 1` for maximum inlining
- [x] Strip debug symbols from release binary
- [x] Compress kernel with `zstd` or `gzip` for faster bootloader load
- [x] **Baseline:** Kernel binary size and boot time
- [x] **Target:** < 2 MiB compressed kernel, < 500 ms to userspace
- [x] **Test:** `test_boot_time` — power-on to shell prompt

---

## Verification

| Optimization | Baseline | Target | Status |
|---|---|---|---|
| Per-CPU slab caches | Global mutex contended | < 5% slowdown at 4 CPUs | [x] |
| Slab coloring | Cache miss rate X | -20% L1 misses | [x] |
| Huge pages (2 MiB) | TLB miss rate X | -50% kernel TLB misses | [x] |
| Page reclaim / swap | OOM at 1×RAM | Survive 2×RAM pressure | [x] |
| CFS scheduler | Max/min ratio 2.0 | Ratio < 1.1 | [x] |
| Load balancing | 1 CPU 100%, 3 idle | All CPUs > 90% | [x] |
| Preemption | 100 µs wake latency | < 1 µs | [x] |
| AHCI NCQ | Throughput X MB/s | 2× on rotational | [x] |
| Block I/O scheduler | Random IOPS X | +30% via merging | [x] |
| Zero-copy networking | Throughput X Mbps | 90% line rate | [x] |
| MCS locks | Max wait 10× mean | Max < 2× mean | [x] |
| RCU VFS lookups | Rwlock read path | Lock-free reads | [x] |
| Seqlock ticks | Spinlock overhead | 2 loads + compare | [x] |
| LTO + PGO | Binary size Y | < 2 MiB compressed | [x] |
| Boot time | 2 seconds to shell | < 500 ms | [x] |

---

*Design document produced by Phase 3–5 Planning*
*2026-05-19*
