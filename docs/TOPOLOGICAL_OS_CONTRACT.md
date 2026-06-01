# Seal OS Topological Contract

Seal OS is a topological OS only when topology is a runtime invariant, not a theme.
This document defines the minimum contract for version 0.4.5 and the later
"perfect" target.

## Hard Rules

1. The kernel is bare-metal Rust, assembly at CPU edges, and Aether-Lang for
   native OS language/app logic. No Unix ABI, no Linux ABI, no POSIX contract,
   no libc target.
2. Boot must fail closed if any T1-T10 theorem gate fails.
3. T1-T5 must drive live kernel decisions, not only print boot lines.
4. T6-T10 must stay boot-gated for the HFT/ML world-model path until they are
   wired into hot runtime governors.
5. O(1) claims must name the bounded constant. If a path scans RAM, directory
   size, task count, or file bytes, the claim must say so.
6. "Surpasses Ubuntu" is allowed only per benchmark row in
   `docs/BENCHMARK_PLAN.md`, never as a blanket claim without measurements.

## Theorem-To-Subsystem Matrix

| Theorem | Required runtime ownership | Current 0.4.5 status |
|---|---|---|
| T1/TSS | Voronoi partitioning for frames, tasks, file/inode lookup, packets | Runtime active in TopoRAM, scheduler, ManifoldFS, topological networking |
| T2/SCM | Spectral contraction for prefetch and next-runnable prediction | Runtime active in TopoRAM, scheduler, ManifoldFS |
| T3/GMC | Entropy/Betti fragmentation control and merge decisions | Runtime active in TopoRAM, ManifoldFS, topological power |
| T4/AGCR | Adaptive governor for timeslice, cache pressure, render pacing | Runtime active in scheduler, ManifoldFS, compositor, swap pressure |
| T5/HCS | Hyperbolic separation for lifetime, path depth, hierarchy | Runtime active in TopoRAM, virtual memory, ManifoldFS, power topology |
| T6/RGCS | Gradient coherence for ML/HFT sync | Boot-gated only |
| T7/PHKP | Persistent homology KV partition latency | Boot-gated only |
| T8/TEB | Energy floor for erasure and data movement | Boot-gated only |
| T9/CMA | Cross-manifold alignment error | Boot-gated only |
| T10/WPHB | Predictive horizon bound | Boot-gated only |

## Required Boot Evidence

The VM serial log must contain:

```text
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
[Aether-Lang] runtime proof: parser=ok interpreter=ok app_host=ok script=aether_boot_probe result=seal-topology-ok
```

`kernel/seal-mkimage --check-theorem-log` is the Rust audit gate for theorem
lines. `kernel/seal-mkimage --check-aether-runtime` is the Rust audit gate for
the Aether parser/interpreter/app-host boot proof.

## O(1) Claim Discipline

Allowed today:

- Slab allocation: O(1) per fixed size class.
- Single-frame physical allocation: O(1) bounded topological free-index lookup
  across eight cells; the bitmap remains the source of truth.
- Bitmap mutation is allocator-private; runtime topology code may use bounded
  read-only bitmap samples but cannot bypass the topological free index.
- Multi-page contiguous physical allocation: O(1) search with respect to
  installed RAM via 128 bounded topological candidate probes, with
  `MAX_CONTIGUOUS_RUN_PAGES = 64` bounding per-call marking and summary refresh.
  Larger transfers must use chunked/scatter-gather I/O instead of one
  unbounded contiguous request.
- TopoRAM public allocation and free-side topology repair share that 64-page
  run cap, so frame metadata repair is bounded by the same constant as physical
  allocation.
- Scheduler selection: O(1) over eight Voronoi cells and fixed priority buckets.
  The `[BENCH] scheduler-select-next` boot marker proves live requeue selection
  with zero context switches, max 9 bitmap tests, and max 256 bucket scan.
- Same-ManifoldFS teleport: O(1) metadata/connectivity update. The
  `[BENCH] manifold-teleport` boot marker proves same-inode movement and bounded
  metadata ops across a small directory-load sweep, with
  `persistence_bytes_per_move=0` forbidding file-byte rewrites.
- TCP packet demux: same-port listener and accepted sockets must resolve through
  the flow-specific match before listener fallback. The `[BENCH] tcp-packet-demux`
  boot marker proves accepted-socket payload delivery for that fixture.
- TopoRAM hint allocation: `[BENCH] toporam-alloc` must show target Voronoi-cell
  hits with zero target-cell fallback, zero zone fallback, monotonic cycles, and
  no frame leak. Prefetch and fragmentation telemetry use bounded sampling, not
  full RAM scans, in runtime hot paths.
- Voronoi locate: O(K) with fixed K, therefore bounded constant for the compiled
  index size.

Not yet closed:

- Scatter-gather drivers are still needed so devices that do not require
  physical contiguity can avoid contiguous pressure entirely.
- Persistent payload-first ManifoldFS extents are not yet the only disk layout.
- T6-T10 are not yet hot-path runtime governors.
- No-von-Neumann-bottleneck is an architectural direction, not a proven universal
  property. Benchmarks must show data movement reductions per workload.

## Perfect Topological OS Closure

Seal OS can claim "perfect topological OS for HFT/ML data movement" only when all
of these are true:

1. VM boot proof passes QEMU and VirtualBox on clean machines.
2. Rust audit gates pass: theorem log, Seal ABI, unsafe inventory, generated-file
   hygiene, and language surface.
3. Every T1-T10 theorem has a Rust gate, named runtime callsite or boot-gated
   reason, and proof-strength status in `docs/THEOREMS.md`.
4. O(1) allocator/data-move claims are backed by code paths and benchmark rows.
5. Ubuntu comparison rows in `docs/BENCHMARK_PLAN.md` show measured wins for the
   named HFT/ML workloads under the same host constraints.
