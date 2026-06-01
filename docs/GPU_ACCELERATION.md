# GPU Acceleration Architecture for Seal OS

## Overview

Seal OS offloads topological computations to discrete GPUs via vendor firmware interfaces. This document describes the architecture, driver model, and verification strategy.

## Design Philosophy

- **Firmware-first**: We interact with GPU firmware (VBIOS/ATOM BIOS for AMD, GSP for NVIDIA, GuC for Intel) to discover capabilities and load microcode, not to rely on closed-source userspace drivers.
- **Unified abstraction**: All GPU vendors expose a common `TopologyAccelerator` trait. Kernel subsystems (scheduler, ManifoldFS, TopoRAM) call this trait without knowing the vendor.
- **CPU fallback**: Every GPU-accelerated operation has a deterministic CPU path. If the GPU driver fails to load or the firmware is missing, the system degrades gracefully to software topology.
- **Security**: Firmware blobs are scanned, checksum-verified, and loaded into isolated GPU memory. The kernel never trusts GPU-generated topology results without a CPU spot-check on the first dispatch.

## Hardware Targets

| Vendor | Target Family | Firmware Interface | Status |
|--------|---------------|-------------------|--------|
| AMD | GCN 1.0–RDNA 3 | ATOM BIOS + PM4 compute | **Infrastructure ready** — PM4 ring, MMIO, firmware scan, and CPU fallback are real. GCN shader binaries are honest stubs (correct ABI, placeholder logic). |
| NVIDIA | Turing–Ada | GSP-RM firmware | Stub + blueprint |
| Intel | Gen 12+ Xe | GuC + HuC firmware | Stub + blueprint |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Kernel Subsystems                       │
│  TopoRAM │ ManifoldFS │ Scheduler │ aether-core │ Tensor Viz │
└──────────────────────┬──────────────────────────────────────┘
                       │ TopologyAccelerator trait
┌──────────────────────▼──────────────────────────────────────┐
│              drivers/gpu/topology_accel.rs                   │
│         (command batching, async fences, fallback)           │
└──────────────────────┬──────────────────────────────────────┘
                       │ vendor dispatch
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
┌──────────────┐ ┌──────────┐ ┌────────────┐
│  AMD GCN     │ │ NVIDIA   │ │  Intel     │
│  Backend     │ │ GSP Stub │ │  GuC Stub  │
│  (pm4/sdma)  │ │ (blueprint)│ │ (blueprint)│
└──────┬───────┘ └────┬─────┘ └─────┬──────┘
       │              │             │
       ▼              ▼             ▼
┌──────────────┐ ┌──────────┐ ┌────────────┐
│ ATOM BIOS    │ │ GSP blob │ │  GuC blob  │
│ parser       │ │ loader   │ │  loader    │
│ firmware_scan│ │ (signed) │ │  (signed)  │
└──────────────┘ └──────────┘ └────────────┘
```

## Firmware Scanner (`drivers/gpu/firmware_scanner.rs`)

### Responsibilities

1. **PCI enumeration** — Walk PCI bus for class 0x03 (Display Controller) and subclass 0x00 (VGA) / 0x02 (3D Controller).
2. **Vendor detection** — Read PCI vendor ID (offset 0x00):  
   - `0x1002` → AMD  
   - `0x10DE` → NVIDIA  
   - `0x8086` → Intel
3. **BAR discovery** — Map PCI BAR0 (MMIO registers), BAR1 (VRAM aperture), BAR5 (VBIOS ROM for AMD).
4. **VBIOS parsing (AMD)** — Parse ATOM BIOS image from BAR5:
   - Validate `0xAA55` ROM signature
   - Walk ATOM directory to find `Data_Table` and `Command_Table`
   - Extract `ASIC_Init` parameters, engine clock, memory clock, VRAM size
   - Identify chip family (`CHIP_FAMILY_*` enum) to select correct ISA encoding
5. **Capability reporting** — Populate `GpuFirmwareCaps`:
   - `has_compute_queue` — PM4 compute ring available
   - `has_sdma` — SDMA engine for async memcpy
   - `vram_size_mb` — Detected VRAM
   - `topology_kernels_supported` — Bitmask of which topology ops have GPU kernels
6. **Firmware blob loading** — For AMD, no signed blob is required (open PM4 interface). For NVIDIA/Intel, stubs document where to place vendor-signed firmware files.

## AMD GCN Backend (`drivers/gpu/amd_gcn.rs`)

### Register Model

AMD GPUs expose a **Memory-Mapped IO (MMIO)** aperture via PCI BAR0. Key register blocks:

| Block | Offset | Purpose |
|-------|--------|---------|
| `GRBM` | 0x8000 | Graphics ring buffer manager |
| `CP_ME` | 0x86C0 | Command processor micro-engine |
| `SRBM` | 0x0E00 | System ring buffer manager |
| `SDMA0` | 0x5000 | SDMA engine 0 registers |
| `IH` | 0x3E00 | Interrupt handler |

### Ring Buffer Model

The Command Processor (CP) consumes packets from a **ring buffer** in GPU-visible memory:

1. Kernel allocates a contiguous physical page (4 KiB aligned) for the ring.
2. Kernel writes `W_PTR` (write pointer) register; CP reads `R_PTR` (read pointer).
3. Driver writes PM4 packets into the ring, then updates `W_PTR`.
4. CP executes packets, updates `R_PTR`, optionally raises an interrupt.

### PM4 Compute Packet Format

PM4 (Packet Type 4) is the native command format for GCN. A compute dispatch requires:

1. `PM4_SET_SH_REG` — Write compute shader registers (user data, shader address)
2. `PM4_DISPATCH_DIRECT` — Launch thread grid (X, Y, Z dimensions)
3. `PM4_EVENT_WRITE_EOP` — End-of-pipeline fence for completion

### Topology GPU Kernels (Embedded ISA)

Kernel binaries are embedded in `drivers/gpu/shader_binaries.rs` as `&[u32]` slices. Each kernel follows the AMD GCN compute ABI (kernel descriptor, SGPR layout, `flat_store` encoding) but contains **honest placeholder logic** — they execute on the GPU without crashing but do not yet perform the full topological computation. This is documented explicitly in source comments.

Replacing stubs with real compiled ISA is a matter of running the OpenCL C source (provided in comments) through `clang -target amdgcn-amd-amdhsa` and pasting the resulting binary words into the arrays.

Available kernels:

| Kernel | Status | Topology Op | Input | Output |
|--------|--------|-------------|-------|--------|
| `voronoi_assign` | Stub (correct ABI) | T1 — assign points to Voronoi cells | `PointCloud*`, `Centroids*` | `CellIds*` |
| `jl_project` | Stub (correct ABI) | JL projection 128→3 | `f64[128]*`, `seed` | `f64[3]*` |
| `spectral_step` | Stub (correct ABI) | T2 — one contraction step | `f64[D]*`, `alpha` | `f64[D]*` |
| `s2_distance` | Stub (correct ABI) | S² great-circle distance matrix | `PointsA*`, `PointsB*` | `f64[N][M]*` |

### Memory Model

GPU-visible memory is allocated through a **GART (Graphics Address Remapping Table)** or **P2P PCIe BAR mapping**. Seal OS allocates physically contiguous pages and maps them into the GPU's PCIe aperture so the GPU can DMA directly to/from kernel buffers.

```
CPU side:        Kernel heap page ( TopoRAM metadata, ManifoldPayload )
                 │
                 ▼ DMA (SDMA)
GPU side:        PCIe BAR1 aperture → GPU cache → Compute unit SRAM
```

## NVIDIA GSP Backend (Stub + Blueprint)

NVIDIA's GSP (GPU System Processor) firmware is signed and required for modern GPUs (Turing+).

### What is stubbed

- `nvidia_gsp.rs` contains the full struct definitions, PCI BAR maps, and GSP message queue layout.
- The actual GSP firmware binary (`gsp.bin`) is **not included** (vendor-signed, redistribution prohibited).

### How to complete

1. Extract `gsp.bin` from the NVIDIA proprietary driver using `nvidia-firmware` tools.
2. Place the blob at `firmware/nvidia/gsp-<version>.bin`.
3. The driver will load it into GPU WPR (Write-Protected Region) memory and boot the GSP-RM.
4. Once GSP-RM is running, use the RM (Resource Manager) API to allocate a compute queue and submit CUDA-like kernels.

## Intel GuC Backend (Stub + Blueprint)

Intel's GuC (Graphics micro-Controller) firmware is also signed.

### What is stubbed

- `intel_guc.rs` contains MMIO register maps, GuC communication ABI, and HuC authentication flow.
- The GuC blob (`guc_<version>.bin`) is **not included**.

### How to complete

1. Obtain `guc_*.*.bin` and `huc_*.*.bin` from `linux-firmware` (redistributable under Intel's license).
2. Place blobs at `firmware/intel/guc.bin` and `firmware/intel/huc.bin`.
3. The driver uploads GuC via the `DMA_CTRL` + `DMA_ADDR_{LO,HI}` MMIO sequence.
4. After GuC is authenticated, use `H2G` (Host-to-GuC) messages to create a compute context.

## Topology Accelerator API

```rust
pub trait TopologyAccelerator {
    /// Probe firmware and initialise hardware.
    unsafe fn init(&mut self, pci_dev: &PciDevice) -> Result<(), GpuError>;

    /// Allocate GPU-visible memory.
    fn alloc_gpu_mem(&mut self, size: usize) -> Result<GpuBuffer, GpuError>;

    /// Copy host → GPU.
    fn upload(&mut self, dst: &GpuBuffer, src: &[u8]) -> Result<(), GpuError>;

    /// Copy GPU → host.
    fn download(&mut self, dst: &mut [u8], src: &GpuBuffer) -> Result<(), GpuError>;

    /// Launch Voronoi cell assignment kernel (T1).
    fn dispatch_voronoi(
        &mut self,
        points: &GpuBuffer,
        centroids: &GpuBuffer,
        cell_ids: &GpuBuffer,
        n_points: usize,
        n_cells: usize,
    ) -> Result<GpuFence, GpuError>;

    /// Launch JL projection kernel (128-dim sparse → 3-dim).
    fn dispatch_jl_project(
        &mut self,
        input: &GpuBuffer,
        output: &GpuBuffer,
        n_vectors: usize,
        seed: u32,
    ) -> Result<GpuFence, GpuError>;

    /// Launch one spectral contraction step (T2).
    fn dispatch_spectral_step(
        &mut self,
        state: &GpuBuffer,
        alpha: f64,
        dim: usize,
    ) -> Result<GpuFence, GpuError>;

    /// Block until fence signals completion.
    fn wait_fence(&mut self, fence: GpuFence, timeout_ms: u32) -> Result<(), GpuError>;

    /// Returns true if this accelerator is ready.
    fn is_ready(&self) -> bool;

    /// Human-readable name.
    fn name(&self) -> &'static str;
}
```

## Integration with Kernel Subsystems

### TopoRAM (`memory/topo_ram.rs`)

When initialising TopoRAM, the kernel checks `TopologyAccelerator::is_ready()`. If ready:
- `alloc_frames()` uses GPU-accelerated Voronoi cell assignment for bulk allocations.
- Spectral prefetch hints are computed on GPU before the CPU prefetch path runs.

### ManifoldFS (`fs/manifold_fs.rs`)

- `store()` — JL projection and L2 normalization for the 64-block payload can be offloaded.
- `find()` — content-addressable lookup uses GPU S² distance matrix computation for batch queries.

### Scheduler (`process/scheduler.rs`)

- `select_next_task()` — T2 prediction (spectral contraction) runs a GPU kernel every N ticks to refresh the prediction state, while the CPU hot path uses cached results.

### aether-core Bridge

`drivers/gpu/topo_accel.rs` implements a bridge crate that mirrors `aether-core` APIs but dispatches to GPU when batch size exceeds a threshold (e.g., > 64 points).

## Verification Strategy

### In-Kernel Benchmark (`apps/gpu_benchmark.rs`)

Run from SealShell:

```
gpu_benchmark
```

The benchmark:

1. Detects the active accelerator (AMD GPU or CPU fallback).
2. Allocates test buffers (spherical point clouds, JL vectors, spectral state).
3. Uploads data, runs warmup, then times 32 iterations via `rdtsc`.
4. Downloads results and verifies correctness:
   - **Voronoi**: every cell ID is in `[0, n_cells)`.
   - **JL projection**: output values are finite.
   - **Spectral step**: output matches expected damping `state * (1 - alpha)`.
5. Reports average cycles per iteration.

When running on the **CPU fallback**, the benchmark measures real software topology execution time.
When running on **AMD hardware**, it measures PM4 ring submission + GPU execution + fence polling time.

### Firmware Scanner Test

Boot with kernel command line `fwscan=verbose` to print:
- Every PCI GPU device found
- Vendor, device ID, BAR sizes
- VBIOS version (AMD)
- Architecture family (Polaris, Vega, RDNA, etc.)
- Firmware blob presence / missing / checksum
- Fallback decision and reason

## QEMU Testing

QEMU does not expose real GPU MMIO in a way that supports compute. For CI and development, use:

```bash
# GPU passthrough (requires real AMD GPU + IOMMU)
./run_qemu_gpu.sh --passthrough 0000:0a:00.0

# Software fallback (no GPU detected, CPU topology still works)
./run_qemu_gpu.sh
```

The `run_qemu_gpu.sh` script probes `/sys/bus/pci` for AMD GPUs and automatically enables passthrough if found; otherwise it boots with software fallback.

## Build Instructions

### Prerequisites

- Rust nightly (same as base kernel)
- `llvm-objcopy` (for embedding ISA binaries)
- `opencl-headers` (optional, for recompiling shaders)

### Compile GPU kernels (optional)

```bash
cd kernel/seal-os/src/drivers/gpu/shaders/
# Edit .cl files, then:
make shaders  # Uses clang with GCN target
```

### Build kernel with GPU support

```bash
cd kernel/seal-os
cargo +nightly build --release
```

The GPU driver subsystem is built by default. There is no separate feature flag — the firmware scanner probes at boot and falls back to software if no supported GPU is found.

## Performance Targets

| Operation | CPU (software) | GPU (AMD RX 580) | Speedup |
|-----------|---------------|------------------|---------|
| Voronoi assign 1024 pts | ~12 µs | ~0.8 µs | **15×** |
| JL project 1024 vectors | ~45 µs | ~2.1 µs | **21×** |
| Spectral step 1024-dim | ~18 µs | ~1.2 µs | **15×** |
| S² distance 1024×1024 | ~210 µs | ~4.5 µs | **47×** |

These are design targets based on theoretical throughput (GCN compute units @ 1.2 GHz). **Measured results** require:
1. Real AMD GCN/RDNA hardware (not available in QEMU).
2. Production shader binaries replacing the current stubs.

The CPU fallback path is fully functional and provides a verified baseline for comparison once GPU kernels are compiled.

---

*Last updated: 2026-06-01*
