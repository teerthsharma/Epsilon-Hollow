# Changelog

All notable changes to LAAMBA GOVERNOR.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [1.0.0] — 2026-05-26

### Added — V1 Ship

- **Desktop App** — Tauri 2.0 + React 18 + TypeScript + Vite + Tailwind CSS
- **6 Panels** — Sample Bay, Engine Rack, Pipeline Mixer, Topology Scope, Parameter Roll, Experiment Timeline, Console
- **14 Engines** — TOML manifest auto-discovery for Faraday, Hamilton, OmniTopos, Lambda-Topo, TopoFlow, Topo-ASM, Epsilon-Hollow, Aether-Lang, TopoBridge-Q, NeuralWhisper, Hollow-ASM, PGTable-ASM, Vec-SIMD, Phi-Mem
- **4 Pipeline Templates** — full-theory-chain, battle-royale, topology-auto-pick, fusion
- **Governor Brain** — `TopologicalGovernor` with 2-layer MLP PolicyNet (5 vitals → 7 manifolds)
- **Battle Royale** — `GovernorOrchestrator` runs 5 topologies over 5 rounds with TransferBus and Comparator
- **Vitals Extractor** — 13 topological features: intrinsic_dim, spectral_gap, curvature_proxy, small_world_coeff, etc.
- **CLI Bridge** — `cli/governor_cli.py` with `vitals`, `analyze`, `battle`, `rank` commands
- **Rust Backend** — 14 IPC commands, engine registry, pipeline DAG + topological sort executor, experiment history
- **3D Visualization** — React Three Fiber manifold viewer with orbit controls
- **Canvas Charts** — Persistence diagrams, Betti curves, convergence plots
- **Drag & Drop** — CSV import via file dialog and HTML5 drag-drop
- **Experiment History** — Every run saved with result recall
- **Dark Theme** — FL Studio-inspired `#0d0d0d` background with `#00E5FF` cyan accent

### Verified Build Artifacts

- Frontend bundle: `dist/assets/index-BRM2w27p.js` (341KB) + CSS (18KB)
- Rust check: `cargo check` — 0 errors
- Preview server: `npm run preview` → `localhost:4173`
- Tauri dev: `localhost:1420`
- Screenshot proof: `screenshot.png` — 1600×900, all panels render

---

## [1.1.0] — Planned (V2 Instruments)

### Added

- Live Embedding Viewer — real-time 3D animation of embedding optimization
- Curvature Heatmap — per-point sectional curvature `K(p) = (2π - Σθᵢ) / A(p)`
- Persistence Barcode — interactive barcode diagram with filtration slider
- Spectral Analyzer — graph Laplacian eigenvalues, spectral gap λ₁, Fiedler vector
- Transfer Learning Bus — hot-swap embeddings between topologies mid-run
- Topology Fusion Console — weighted blend of multiple manifold embeddings

---

## [1.2.0] — Planned (V3 Formula Playgrounds)

### Added

- Riemannian Metric Playground — edit gᵢⱼ, see geodesics update live
- Homology Calculator — compute H₀, H₁, H₂ with coefficient fields
- Betti Number Explorer — epsilon slider, watch βₖ change in real-time
- Euler Characteristic Tracker — χ(ε) over filtration
- Wasserstein Distance — compare persistence diagrams between datasets
- Bottleneck Distance — stability analysis
- Vietoris-Rips Complex Builder — step-by-step simplicial complex visualization
- Mapper Algorithm — interactive Mapper with lens function selection
- Morse Theory Viewer — critical points and gradient flow
- Hodge Decomposition — harmonic / exact / coexact forms

---

## [1.3.0] — Planned (V4 Physics Engines)

### Added

- Faraday Tensor Field — EM field visualization on manifolds
- Hamilton N-Body — symplectic integration on curved spaces
- OmniTopos Phase Sequences — cosmological phase transitions
- Gauge Theory Playground — SU(2)/SU(3) connection visualization

---

## [1.4.0] — Planned (V5 Quantum Bridge)

### Added

- TopoBridge-Q — Qiskit integration, quantum persistent homology
- Quantum SWAP Test — fidelity measurement circuits
- Topological Quantum Codes — surface codes on genus-g surfaces
- Hybrid Classical-Quantum — automatic fallback to simulator

---

## [1.5.0] — Planned (V6 Systems)

### Added

- Epsilon-Hollow OS — Seal OS microkernel demo with topological scheduling
- Aether-Lang — Topology programming language runner
- Topo-ASM SIMD Backend — AVX-512 replaces Python ripser
- Phi-Mem Allocator — topological fragmentation detection
- Vec-SIMD — Poincaré disk ops in packed SIMD

---

*Built by seal. Topology is the shape of truth.*
