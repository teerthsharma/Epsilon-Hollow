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

- [ ] Allocate per-CPU slab caches during SMP bringup
- [ ] Access via `gs:[offset]` (no locking for fast path)
- [ ] Steal from other CPUs' caches when local cache empty (lock other CPU's cache)
- [ ] Rebalance periodically (every 1000 allocs, move 1/4 of surplus)
- [ ] **Baseline:** `Box::new` throughput on 4 CPUs
- [ ] **Target:** < 5% slowdown at 4 CPUs vs 1 CPU
- [ ] **Test:** `test_slab_contention_4cpu` — 4 CPUs alloc/free in loop, no deadlock

### 1.2 Slab Coloring

Prevent cache-line conflicts between adjacent slab objects.

- [ ] Add `color_offset` to each slab cache (0..cache_line_size)
- [ ] Vary color between caches of same size
- [ ] **Baseline:** Cache miss rate on slab-allocated structs
- [ ] **Target:** 20% reduction in L1 misses
- [ ] **Test:** `test_slab_coloring` — measure cache misses via PMC

### 1.3 Huge Page Support (2 MiB / 1 GiB)

Current: Only 4 KiB pages. Large allocations (DMA buffers, framebuffer) waste TLB entries.

- [ ] Detect CPU support for 2 MiB and 1 GiB pages (`CPUID`)
- [ ] Add `PageSize` enum: `Size4K`, `Size2M`, `Size1G`
- [ ] Implement `alloc_hugepage(size)` in physical allocator
- [ ] Map huge pages in page tables (P2/P3 level)
- [ ] Use 2 MiB pages for kernel text/data (reduces TLB pressure)
- [ ] Use 2 MiB pages for large DMA buffers
- [ ] **Baseline:** TLB miss rate during kernel operation
- [ ] **Target:** 50% reduction in TLB misses for kernel space
- [ ] **Test:** `test_hugepage_mapping` — map 2 MiB page, verify contiguous physical

### 1.4 Page Reclaim / Swap

Current: No swap. OOM kills the allocator.

- [ ] Implement page reclamation daemon (kreclaimd)
- [ ] Track page age (LRU: active / inactive lists)
- [ ] Reclaim inactive anonymous pages → swap
- [ ] Reclaim inactive file-backed pages → drop from cache
- [ ] Implement swap partition or swap file
- [ ] **Baseline:** OOM behavior under memory pressure
- [ ] **Target:** System survives 2×RAM allocation pressure
- [ ] **Test:** `test_oom_survival` — allocate 2× total RAM, verify swap usage + no panic

---

## 2. Scheduler Optimizations

### 2.1 CFS (Completely Fair Scheduler)

Current: Simple round-robin or priority queue.

- [ ] Implement red-black tree of runnable tasks keyed by vruntime
- [ ] Calculate vruntime = actual_runtime / weight
- [ ] Weights: nice -20 = 88761, nice 0 = 1024, nice 19 = 15
- [ ] Pick leftmost (lowest vruntime) task on each tick
- [ ] Update vruntime on task switch
- [ ] **Baseline:** Fairness metric (max/min CPU time ratio across tasks)
- [ ] **Target:** Ratio < 1.1 for equal-priority tasks
- [ ] **Test:** `test_cfs_fairness` — 4 CPU-bound tasks, verify ~25% each

### 2.2 Load Balancing

Current: Static assignment or none.

- [ ] Track per-CPU runnable queue length
- [ ] Every 4 ms, compare queue lengths across CPUs
- [ ] If imbalance > 25%, migrate task from busiest to idlest
- [ ] Prefer migrating tasks that are cache-cold (not ran recently)
- [ ] **Baseline:** CPU utilization under uneven load
- [ ] **Target:** All CPUs > 90% utilization with 1 task per CPU
- [ ] **Test:** `test_load_balance` — spawn 100 tasks on 4 CPUs, all make progress

### 2.3 Preemption

Current: Cooperative or timer-tick only.

- [ ] Implement `TIF_NEED_RESCHED` flag in thread info
- [ ] Set flag on: higher-priority task wakes, timer tick expires quantum
- [ ] Check flag on: syscall exit, interrupt return
- [ ] Trigger `schedule()` if flag set
- [ ] **Baseline:** Latency of waking a higher-priority task
- [ ] **Target:** < 1 µs from wake to context switch
- [ ] **Test:** `test_preemption_latency` — measure wake-to-schedule time

---

## 3. I/O Optimizations

### 3.1 AHCI NCQ (Native Command Queuing)

Current: Single command slot, polling.

- [ ] Implement 32-slot command queue per port
- [ ] Tag commands with slot number
- [ ] Submit multiple commands without waiting
- [ ] Process completions via interrupt (softirq)
- [ ] Reorder completions (NCQ may complete out-of-order)
- [ ] **Baseline:** Sequential read throughput (single slot)
- [ ] **Target:** 2× throughput on rotational media, 1.5× on SSD
- [ ] **Test:** `test_ahci_ncq_throughput` — sequential read 100 MiB

### 3.2 Block I/O Scheduler

Current: FIFO submission.

- [ ] Implement elevator algorithm (CFQ / BFQ lite)
- [ ] Merge adjacent requests (req LBA 0-7 + req LBA 8-15 → req LBA 0-15)
- [ ] Sort requests by LBA for sequential media
- [ ] Deadline timer: read expires in 500 ms, write in 5 s
- [ ] **Baseline:** Random 4 KiB read IOPS
- [ ] **Target:** 30% improvement via request merging
- [ ] **Test:** `test_io_scheduler_merge` — submit adjacent requests, verify single command

### 3.3 Read-Ahead

See [BUFFER-CACHE.md](BUFFER-CACHE.md) §6.

- [ ] Implement adaptive read-ahead (detect stride patterns)
- [ ] Implement synchronous read-ahead for mmap fault
- [ ] **Baseline:** Sequential read throughput without read-ahead
- [ ] **Target:** 5× improvement for sequential reads
- [ ] **Test:** `test_readahead_sequential` — defined in BUFFER-CACHE.md

---

## 4. Network Optimizations

### 4.1 Zero-Copy Networking

Current: Packet copies between NIC ring buffer, skbuff, socket buffer, userspace.

- [ ] Map NIC RX ring buffers directly into socket receive buffer
- [ ] Map socket send buffer directly into NIC TX ring buffer
- [ ] Use `mmap()` for userspace <→ kernel buffer sharing
- [ ] **Baseline:** TCP throughput (single connection)
- [ ] **Target:** 90% of line rate on 1 Gbps
- [ ] **Test:** `test_tcp_zero_copy` — iperf3-equivalent throughput test

### 4.2 TCP Offload (TSO / LRO)

- [ ] Implement TCP Segmentation Offload (TSO) if NIC supports
- [ ] Implement Large Receive Offload (LRO) if NIC supports
- [ ] **Baseline:** TCP throughput with software segmentation
- [ ] **Target:** Near line rate with large sends
- [ ] **Test:** `test_tcp_tso` — verify NIC segments large packets correctly

### 4.3 IRQ Coalescing

- [ ] Batch multiple RX packets per interrupt
- [ ] Configure NIC interrupt moderation timer (100 µs)
- [ ] **Baseline:** Interrupt rate under 1 Gbps RX
- [ ] **Target:** < 10k interrupts/second at line rate
- [ ] **Test:** `test_irq_coalescing` — measure interrupt count during flood

---

## 5. Locking Optimizations

### 5.1 MCS Locks

Current: Ticket spinlock or test-and-set. Unfair under contention.

- [ ] Implement MCS lock (per-CPU queue node)
- [ ] Use MCS for long-held locks (scheduler, VFS)
- [ ] Keep test-and-set for short critical sections (< 100 ns)
- [ ] **Baseline:** Latency of 4 CPUs contending on same lock
- [ ] **Target:** Fairness (max wait time < 2× mean wait time)
- [ ] **Test:** `test_mcs_fairness` — 4 CPUs acquire lock 10k times each

### 5.2 RCU (Read-Copy-Update)

For read-mostly data structures (VFS mount table, device list, routing table).

- [ ] Implement RCU core: `rcu_read_lock()`, `rcu_read_unlock()`, `synchronize_rcu()`
- [ ] Implement `call_rcu()` for deferred freeing
- [ ] Track grace periods via per-CPU quiescent states
- [ ] **Baseline:** VFS mount table lookup with rwlock
- [ ] **Target:** Read path is lock-free (no atomic ops)
- [ ] **Test:** `test_rcu_vfs` — 1000 readers + 1 writer, no UAF

### 5.3 Seqlocks

For single-writer, many-reader data (jiffies, ticks, statistics).

- [ ] Implement seqlock (sequence counter + data)
- [ ] Use for `ticks` global
- [ ] Use for per-CPU scheduler statistics
- [ ] **Baseline:** `ticks()` read with full spinlock
- [ ] **Target:** Read path: 2 loads + compare (no lock acquisition)
- [ ] **Test:** `test_seqlock_ticks` — concurrent readers + writer, no torn reads

---

## 6. Compiler / Link-Time Optimizations

- [ ] Enable LTO (`lto = "fat"` in Cargo.toml)
- [ ] Enable PGO (Profile-Guided Optimization) — build, run benchmark, rebuild with profile
- [ ] Use `target-cpu = "x86-64-v3"` for AVX2 baseline
- [ ] Evaluate `x86-64-v4` (AVX-512) — likely not worth power cost
- [ ] Enable `codegen-units = 1` for maximum inlining
- [ ] Strip debug symbols from release binary
- [ ] Compress kernel with `zstd` or `gzip` for faster bootloader load
- [ ] **Baseline:** Kernel binary size and boot time
- [ ] **Target:** < 2 MiB compressed kernel, < 500 ms to userspace
- [ ] **Test:** `test_boot_time` — power-on to shell prompt

---

## Verification

| Optimization | Baseline | Target | Status |
|---|---|---|---|
| Per-CPU slab caches | Global mutex contended | < 5% slowdown at 4 CPUs | [ ] |
| Slab coloring | Cache miss rate X | -20% L1 misses | [ ] |
| Huge pages (2 MiB) | TLB miss rate X | -50% kernel TLB misses | [ ] |
| Page reclaim / swap | OOM at 1×RAM | Survive 2×RAM pressure | [ ] |
| CFS scheduler | Max/min ratio 2.0 | Ratio < 1.1 | [ ] |
| Load balancing | 1 CPU 100%, 3 idle | All CPUs > 90% | [ ] |
| Preemption | 100 µs wake latency | < 1 µs | [ ] |
| AHCI NCQ | Throughput X MB/s | 2× on rotational | [ ] |
| Block I/O scheduler | Random IOPS X | +30% via merging | [ ] |
| Zero-copy networking | Throughput X Mbps | 90% line rate | [ ] |
| MCS locks | Max wait 10× mean | Max < 2× mean | [ ] |
| RCU VFS lookups | Rwlock read path | Lock-free reads | [ ] |
| Seqlock ticks | Spinlock overhead | 2 loads + compare | [ ] |
| LTO + PGO | Binary size Y | < 2 MiB compressed | [ ] |
| Boot time | 2 seconds to shell | < 500 ms | [ ] |

---

*Design document produced by Phase 3–5 Planning*
*2026-05-19*
