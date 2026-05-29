# ManifoldVRAM Fast Path

Seal OS is aiming at GPU-native data movement for HFT and ML. VRAM is useful
only when it keeps bytes out of CPU copy loops; it does not make a byte copy
O(1) by itself.

## Rule

File teleport has an O(1) metadata-rewiring core because Seal OS changes topology metadata:

- inode parent/connectivity
- ManifoldFS Voronoi cell membership
- capability handle
- optional GPU buffer descriptor

The payload is not copied during teleport. If bytes are copied into VRAM, that
operation is O(N), even if the constant is much faster than system RAM.

## Design

ManifoldVRAM is an optional hot-data arena:

1. Disk or RAM keeps persistent truth.
2. TopoRAM allocates CPU-visible physical frames through the topological free
   index.
3. VRAM stores hot ManifoldPayload embeddings, tensor batches, HFT ring buffers,
   and GPU-side Voronoi buckets.
4. Teleport mutates descriptors in O(1); materialization into VRAM is lazy.
5. Hardware that supports peer-to-peer DMA can move NVMe or NIC buffers directly
   into VRAM, bypassing CPU copy loops.

## VM Contract

Oracle/VirtualBox may not expose real GPU passthrough. Seal OS must therefore
boot and pass theorem gates without GPU acceleration. The GPU path is an
optional fast path:

- Virtio GPU or framebuffer path: correctness fallback.
- Native GPU driver path: ManifoldVRAM cache and topology kernels.
- Peer DMA path: benchmark-only unless hardware proves support.

## Benchmarks Required

VRAM claims need separate rows in `docs/BENCHMARK_PLAN.md`:

- CPU ManifoldFS teleport vs VRAM descriptor teleport.
- NVMe-to-RAM copy vs NVMe-to-VRAM peer DMA.
- Tensor batch load through CPU heap vs ManifoldVRAM hot cache.
- HFT packet ring CPU copy vs NIC-to-VRAM peer path where hardware supports it.
