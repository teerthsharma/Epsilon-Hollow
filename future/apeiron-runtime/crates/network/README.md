<div align="center">

<img src="https://img.shields.io/badge/AETHER--Link-v0.1.0-blue?style=for-the-badge&logo=rust&logoColor=white" alt="AETHER-Link"/>

# AETHER-Link

**Adaptive I/O Prefetch Kernel**

*Sub-15ns decision latency • 65M ops/sec • Zero allocations*

[![License](https://img.shields.io/badge/License-MIT%2FApache--2.0-green?style=flat-square)](LICENSE-MIT)
[![Rust](https://img.shields.io/badge/Rust-1.70+-orange?style=flat-square&logo=rust)](https://www.rust-lang.org/)
[![Build](https://img.shields.io/github/actions/workflow/status/teerthsharma/aether-link/ci.yml?style=flat-square&label=CI)](https://github.com/teerthsharma/aether-link/actions)

[Paper](https://doi.org/10.13140/RG.2.2.22443.91687) · [Quick Start](#quick-start) · [Benchmarks](#benchmarks) · [Docs](docs/)

</div>

---

## What is AETHER-Link?

A high-performance I/O prefetch kernel that predicts whether to bypass the OS page cache **before** the request is issued. Designed for latency-critical systems where every nanosecond counts.

| Target | Use Case |
|--------|----------|
| **HFT Systems** | Market data prefetching with deterministic timing |
| **DirectStorage** | GPU asset streaming for games |
| **WSL2** | NVMe bypass for Linux workloads |
| **LLM Inference** | Model weight loading acceleration |

---

## Performance

Benchmarked on local hardware:

```
┌────────────────────────────────────────────────────┐
│  AETHER-Link Kernel Performance                    │
├─────────────────────────┬──────────────────────────┤
│  Full Decision Cycle    │  14.6 ns                 │
│  Feature Extraction     │  0.99 ns                 │
│  Adaptive Gate          │  3.68 ns                 │
│  Throughput             │  65.3 million ops/sec    │
└─────────────────────────┴──────────────────────────┘
```

For reference: DDR4 memory latency is ~14-18ns. AETHER decisions are **faster than RAM**.

---

## Quick Start

```rust
use aether_link::AetherLinkKernel;

// Initialize with HFT-optimized preset
let mut kernel = AetherLinkKernel::new_hft();

// In your I/O loop
let lba_history = get_recent_lbas();  // Recent LBA addresses

if kernel.process_io_cycle(&lba_history) {
    direct_storage_read(request);  // Bypass OS → GPU
} else {
    standard_read(request);        // Standard path
}
```

### Presets

```rust
AetherLinkKernel::new_hft()    // Conservative (minimize false positives)
AetherLinkKernel::new_gaming() // Aggressive (maximize prefetch rate)
```

---

## How It Works

```
I/O Stream → Extract Features → Adaptive Decision → Prefetch/Skip
              (0.99 ns)          (14.6 ns total)
```

1. **Feature Extraction** – 6D telemetry from LBA history (spatial, temporal, spectral)
2. **Adaptive Threshold** – Self-tuning decision boundary, no training required
3. **Probabilistic Gate** – Sigmoid probability vs adaptive epsilon

The kernel learns from every I/O cycle with O(1) complexity and zero heap allocations.

> 📄 **Theory**: See [docs/QUANTUM_THEORY.md](docs/QUANTUM_THEORY.md) for mathematical foundations

---

## Build

```bash
git clone https://github.com/teerthsharma/aether-link
cd aether-link
cargo build --release
cargo test
cargo bench
```

---

## Documentation

| Doc | Description |
|-----|-------------|
| [Architecture](docs/ARCHITECTURE.md) | System design & math |
| [Benchmarks](docs/BENCHMARKS.md) | Performance analysis |
| [Theory](docs/QUANTUM_THEORY.md) | Algorithm foundations |
| [NVIDIA](docs/NVIDIA_INTEGRATION.md) | DPU/GPU integration |

---

## Roadmap

- [x] Core adaptive kernel
- [x] Presets (HFT, Gaming)
- [x] Criterion benchmarks
- [ ] DirectStorage integration
- [ ] WSL2 kernel module
- [ ] CUDA batch kernel

---

## Citation

```bibtex
@article{sharma2024aether,
  title   = {AETHER - Adaptive Event-driven Threshold Hybrid Entangled Rendering},
  author  = {Sharma, Teerth},
  journal = {ResearchGate},
  year    = {2024},
  doi     = {10.13140/RG.2.2.22443.91687}
}
```

---

## License

[MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE)

