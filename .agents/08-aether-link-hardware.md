# 08 — Aether-Link Hardware: Wire Prefetch Algorithm to Real I/O

## Goal

Connect the Aether-Link adaptive prefetch engine (currently a pure algorithm) to actual block device I/O, so the 6D telemetry extraction and sigmoid fetch probability drive real NVMe/DMA prefetch decisions — achieving HFT-grade I/O latency where the only bottleneck is hardware.

## Current State

`kernel/aether/aether-link/src/lib.rs` (~363 lines):
- **Working**: `TelemetryExtractor` — 6D feature vector per I/O cycle (Δ offset, velocity, variance, Goertzel spectral power, history depth, context entropy)
- **Working**: `DecisionEngine` — arctan angle mapping, adaptive threshold, sigmoid fetch probability
- **Working**: Presets — HFT (conservative: low threshold, high confidence required), Gaming (aggressive: prefetch broadly)
- **Working**: ~18ns/cycle computational overhead
- **Not connected**: No actual I/O — `fetch()` computes probability but never issues a DMA read or NVMe command

`kernel/aether/aether-link/src/fast_math.rs` (~124 lines):
- Padé approximant `atan`, standard `exp`, `sigmoid`, Quake III `fast_inv_sqrt` (unused)

## Implementation Steps

1. **Define the I/O backend trait**
   ```rust
   pub trait IoBackend {
       fn read_block(&self, lba: u64, buf: &mut [u8]) -> Result<(), IoError>;
       fn prefetch_block(&self, lba: u64) -> Result<(), IoError>; // non-blocking hint
       fn block_size(&self) -> usize;
       fn queue_depth(&self) -> usize;
   }
   ```

2. **Implement VirtIO block backend** (for QEMU testing)
   - Reuse/share `VirtioBlock` from plan 03 (ManifoldFS)
   - Implement `prefetch_block` as a low-priority read into a prefetch cache
   - Track prefetch hit/miss ratio

3. **Implement NVMe backend** (for real hardware)
   - NVMe submission/completion queue pair
   - PCIe BAR mapping for NVMe controller registers
   - Use NVMe dataset management hint for prefetch (non-destructive)
   - DMA ring buffer for zero-copy block reads

4. **Wire `DecisionEngine` to `IoBackend`**
   - On each I/O request: extract 6D telemetry → compute fetch probability
   - If probability > threshold: issue `prefetch_block` for predicted next LBAs
   - Prediction model: linear extrapolation from velocity + spectral analysis for periodic patterns
   - Prefetch cache: ring buffer of pre-fetched blocks, evicts LRU

5. **Implement prefetch cache**
   - Fixed-size cache (configurable, default 256 blocks = 1MB at 4K blocks)
   - On read: check cache first → cache hit = zero-latency read
   - On prefetch decision: async read into cache slot
   - Track: hit rate, miss rate, wasted prefetch rate

6. **Add HFT timing instrumentation**
   - Use TSC (Time Stamp Counter) for nanosecond-precision timing
   - Measure: I/O request → completion latency, prefetch hit → serve latency, cache miss penalty
   - Expose via `pub fn io_stats() -> IoStats`

7. **Aether-Link integration in ManifoldFS**
   - ManifoldFS file reads go through Aether-Link
   - Teleport operations: Aether-Link predicts which blocks will be needed after surgery
   - Sequential scan: Aether-Link detects linear pattern and prefetches ahead

8. **Benchmark harness**
   - Sequential read: 1000 blocks, measure average latency with/without Aether-Link
   - Random read: 1000 random LBAs, measure cache hit rate
   - Teleport burst: 100 teleports, measure I/O amplification
   - Compare HFT preset vs Gaming preset performance profiles

## Dependencies

- **03-manifold-fs** (block device abstraction)
- **01-kernel-safety** (no deadlocks in I/O path)

## Acceptance Criteria

- [ ] Aether-Link issues real prefetch commands to block device
- [ ] Prefetch cache serves hits with measurably lower latency than cache misses
- [ ] HFT preset: ≥80% prefetch accuracy on sequential workloads
- [ ] Gaming preset: ≥60% prefetch accuracy on mixed workloads
- [ ] No regression in ManifoldFS correctness
- [ ] Benchmark shows measurable latency reduction vs naive I/O
- [ ] `cargo test -p aether-link` passes
- [ ] Works under QEMU with virtio-blk

## Files to Modify

- `kernel/aether/aether-link/src/lib.rs` (add IoBackend integration)
- `kernel/seal-os/src/fs/manifold_fs.rs` (wire reads through Aether-Link)

## Files to Create

- `kernel/aether/aether-link/src/backend.rs` (IoBackend trait)
- `kernel/aether/aether-link/src/cache.rs` (prefetch cache)
- `kernel/aether/aether-link/src/nvme.rs` (NVMe backend — future)
- `kernel/aether/aether-link/src/bench.rs` (benchmark harness)
