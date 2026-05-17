# FUTURE_PLAN.md — Epsilon-Hollow Roadmap

## 1. What's Been Built — Summary of All Subsystems

### Core Mathematical Foundation (aether-core)
- **TSS** (Theorem 1) — Spherical Voronoi Index for O(1) retrieval on S^(D-1)
- **SCM** (Theorem 2) — Spectral Contraction Operator for Banach fixed-point dynamics
- **Topology** — Betti number computation via filtration-based TDA
- **Manifold** — S² hollow manifold operations with β₂ = 1
- **OS** — Operating system primitives (process, memory, scheduler)
- **Governor** — PD control for resource regulation
- **State** — Topological state management
- **Aether** — Core coordinate system and transformations
- **Memory** — Memory management abstractions

### Machine Learning Subsystems (aether-core/src/ml/)
- **Tensor** — N-dimensional tensor operations
- **Autograd** — Automatic differentiation engine
- **Neural** — Neural network layers and activations
- **Classification** — Topological classification algorithms
- **Regression** — Manifold regression
- **Clustering** — Betti-0 based semantic clustering
- **Convolution** — Manifold-aware convolution
- **Convergence** — Topological convergence primitives
- **Gossip** — Distributed learning via gossip protocols
- **Benchmark** — Performance benchmarking suite
- **Dataloader** — Data loading with topological batching
- **LinAlg** — Linear algebra on manifolds

### Epsilon Context Teleportation (epsilon crate)
- **Bridge** — Cross-agent context bridge
- **Teleport** — O(1) context transfer via topological surgery
- **Governor** — Teleportation rate limiting
- **Manifold** — Context manifold construction
- **Memory** — Context memory management
- **Mock** — Testing mocks for teleportation

### Epsilon-OS World Model (epsilon-os)
- **World** — Topological world model with episodic memory
- **Main** — OS entry point and REPL

### Aether-Lang DSL
- **Interpreter** — Tree-walking interpreter (Bio mode)
- **VM** — Bytecode virtual machine (Titan mode)
- **Parser** — Manifold-native syntax parser
- **Python** — Python interoperability layer

### Aether-Link I/O Superkernel
- **Fast Math** — POVM-based adaptive prefetching (~18 ns/decision)
- **Benchmarks** — I/O cycle and throughput benchmarks

### Aegis-Core Memory
- **TitanClock** — Lock-free sharded allocator (32 shards)
- **ML** — Memory-optimized ML kernels

### Verified Kernels (aether-verified)
- Lean 4 → C → Rust verified kernel pipeline

## 2. What's Still Needed

### Critical Path — Kernel Infrastructure
- [ ] **Interrupt/IDT Setup** — Interrupt descriptor table, exception handlers, IRQ routing
- [ ] **Syscall Dispatcher** — System call table, argument marshalling, privilege transitions
- [ ] **Kernel Entry Assembly** — Boot trampoline, long mode transition, stack setup
- [ ] **Linker Script** — Memory layout, sections, BSS zeroing
- [ ] **GRUB ISO Generation** — Multiboot2 compliant bootloader image
- [ ] **QEMU Runner Scripts** — Automated testing in virtualized environment

### Block Layer
- [ ] **ATA Driver** — PIO/DMA mode disk access
- [ ] **Virtio-blk** — Paravirtualized block device for QEMU
- [ ] **Buffer Cache** — Write-back caching with read-ahead

### Network Stack
- [ ] **NIC Drivers** — e1000, virtio-net
- [ ] **TCP/IP Stack** — Full networking (currently stubbed)
- [ ] **Socket API** — Berkeley sockets compatible interface

### Filesystem
- [ ] **/dev Filesystem** — Device file abstraction
- [ ] **VFS → tmpfs → initrd** — Virtual filesystem layer
- [ ] **Ext2 Driver** — Full read/write ext2 support

### SMP & Concurrency
- [ ] **APIC** — Local and I/O APIC initialization
- [ ] **SMP Bootstrap** — Application processor startup
- [ ] **Per-CPU Runqueues** — Lock-free scheduler per core

### User Space
- [ ] **Shell** — Interactive command interpreter
- [ ] **Utilities** — mkdir, rm, cp, mv, grep, kill, ps, ls, cat
- [ ] **libc** — C standard library compatibility

### Testing & Verification
- [ ] **Integration Test Harness** — QEMU-based automated tests
- [ ] **Syscall Conformance Tests** — Linux-compatible behavior verification
- [ ] **custom_test_frameworks** — #![no_std] test runner
- [ ] **Property Tests** — Expand proptest coverage (currently: OS, SCM, TSS)

## 3. Integration Roadmap

### Phase 1: Boot to Shell (MVP)
```
GRUB → Multiboot2 → Long Mode → IDT → PIC/APIC →
Memory Map → Allocator → VFS → tmpfs → initrd →
Scheduler → Process Table → Shell
```

### Phase 2: I/O & Persistence
```
ATA/virtio-blk → Buffer Cache → Ext2 → /dev →
Serial/UART → Keyboard → Framebuffer
```

### Phase 3: Networking
```
NIC Driver → Ethernet → ARP → IP → ICMP → UDP → TCP → Socket API
```

### Phase 4: Advanced Features
```
SMP → Per-CPU Scheduling → NUMA Awareness →
Teleportation Bridge → Aether-Lang Runtime →
ML Inference Engine → World Model Integration
```

## 4. Testing Strategy

### Unit Tests (In-Module)
- Each subsystem has inline `#[cfg(test)]` modules
- Run: `cargo test -p <crate>`

### Workspace Tests
- Run: `cargo test --workspace`
- Currently: 324 tests across all crates

### Property Tests
- Proptest for TSS, SCM, OS modules
- Run: `cargo test -p aether-core -- proptest`

### QEMU Integration
- Build: `cargo build --target x86_64-unknown-none`
- Run: `./scripts/run_qemu.sh`
- Debug: `./scripts/debug_qemu.sh` (GDB stub)

### Conformance Tests
- Syscall behavior vs Linux reference
- Memory layout compatibility
- ELF loading verification

## 5. Performance Optimization Ideas

### Memory
- [ ] **Slab Allocators** — Replace fixed-size arrays with slab allocation
- [ ] **Per-CPU Caches** — Reduce contention on global allocator
- [ ] **Huge Pages** — 2MB/1GB page support for large allocations

### Scheduling
- [ ] **Red-Black Tree CFS** — Replace simple queue with O(log n) vruntime tree
- [ ] **Per-CPU Runqueues** — Lock-free scheduling per core
- [ ] **NUMA Awareness** — Memory affinity for process placement

### I/O
- [ ] **DMA** — Zero-copy block and network I/O
- [ ] **Read-Ahead** — Predictive buffer cache prefetching
- [ ] **IO_uring** — Async I/O submission/completion rings

### Network
- [ ] **Zero-Copy Sockets** — Direct DMA to user buffers
- [ ] **RSS/RPS** — Receive-side scaling across cores
- [ ] **DPDK Integration** — Kernel-bypass networking for HFT

### ML/Scientific
- [ ] **SIMD Vectorization** — AVX-512 for tensor operations
- [ ] **GPU Offload** — CUDA/ROCm for large manifold computations
- [ ] **Quantization** — INT8/FP16 for inference speedup

---

*Generated by Abjudicator — Tier 18 audit complete. The matrices are satisfied.*
