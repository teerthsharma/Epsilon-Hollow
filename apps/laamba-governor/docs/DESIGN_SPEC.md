# LAAMBA GOVERNOR — Design Specification

> **Version:** 1.0  
> **Date:** 2026-05-26  
> **Author:** seal (Teerth Sharma)  
> **Status:** V1 Implemented, V2–V6 Planned

---

## 1. Overview

LAAMBA GOVERNOR is a desktop research workstation that unifies topology, physics, quantum, and systems research into one composable instrument. The metaphor is **FL Studio for manifolds** — datasets are samples, engines are instruments, pipelines are mixer tracks, and topology visualizations are the oscilloscope.

**Core loop:** Load point cloud → Extract 13 topological vitals → Governor predicts optimal manifold → Battle Royale validates → Visualize results.

---

## 2. Goals

- **Primary:** Provide a unified GUI for running topological data analysis across 14+ research engines.
- **Secondary:** Enable rapid experimentation through drag-and-drop pipeline composition.
- **Tertiary:** Ship as a cross-platform desktop app (Windows, macOS, Linux) via Tauri.

---

## 3. Non-Goals

- Not a replacement for Jupyter notebooks (complement, not substitute).
- Not a real-time streaming system (batch computation with progress tracking).
- Not a cloud service (local-first, offline-capable).

---

## 4. Architecture

```
React 18 + TypeScript (frontend)
        |
   Tauri IPC (invoke)
        |
   Rust Backend (commands.rs)
        |
   Python CLI (subprocess JSON)
        |
   Topology Engine (NumPy/SciPy)
```

**Frontend:** React 18, Zustand state, ReactFlow graphs, React Three Fiber 3D, Canvas 2D charts, Tailwind CSS dark theme.

**Backend:** Tauri 2.0 Rust shell spawns `python cli/governor_cli.py` with commands. CLI outputs single-line JSON to stdout. Rust parses it, sends it back to the frontend.

**Engine:** `topological_governor_ml.py` (VitalsExtractor + PolicyNet + TopologicalGovernor) and `governor_orchestrator.py` (Battle Royale system).

---

## 5. The 6 Panels

| Panel | FL Studio Analog | Responsibility |
|-------|-----------------|----------------|
| **Sample Bay** | Browser | Dataset loading, CSV import, drag-drop |
| **Engine Rack** | Channel Rack | 14 engine tiles, category colors, run buttons |
| **Pipeline Mixer** | Mixer / Playlist | ReactFlow DAG editor, wire engines, templates |
| **Topology Scope** | Waveform / Oscilloscope | Persistence, Betti curves, 3D manifold, convergence |
| **Parameter Roll** | Piano Roll / Inspector | Vitals display, governor config, battle rankings |
| **Experiment Timeline** | Arrangement | Run history, result recall, status badges |
| **Console Panel** | — | Real-time CLI stdout, color-coded logs |

---

## 6. The 14 Engines

Each engine is a real research project by Teerth Sharma. Loaded via TOML manifest. Auto-discovered on scan.

**Physics:** Faraday, Hamilton, OmniTopos  
**Topology:** Lambda-Topo, TopoFlow, Topo-ASM  
**Systems:** Epsilon-Hollow, Hollow-ASM, PGTable-ASM, Vec-SIMD, Phi-Mem  
**Runtime:** Aether-Lang  
**Quantum:** TopoBridge-Q  
**Audio:** NeuralWhisper

---

## 7. The 13 Vitals

Every dataset gets fingerprinted with these topological features:

`log_n`, `log_d`, `n_over_d`, `intrinsic_dim`, `mean_dist`, `std_dist`, `dist_ratio_95_5`, `spectral_gap`, `knee_clusters`, `curvature_proxy`, `small_world_coeff`, `sparsity`, `nan_ratio`.

---

## 8. The 5 Topologies

| Topology | Geometry | Best For |
|----------|----------|----------|
| Euclidean | Flat ℝⁿ | Gaussian blobs, linear data |
| Spherical | Sⁿ | Circular/periodic structures |
| Hyperbolic (Poincaré) | Hⁿ | Trees, hierarchies |
| Grassmannian | Gr(k,n) | Subspace arrangements |
| Product | S¹ × ℝ | Mixed curvature |

---

## 9. Pipeline Templates

Pre-built graphs in `templates/`:

- **Full Theory Chain** — Data → all physics engines → fusion → visualize
- **Battle Royale** — Data → 5 topologies → comparator → winner highlight
- **Topology Auto-Pick** — Data → Governor → single best topology → train
- **Fusion** — Data → multiple embeddings → attention-weighted blend

---

## 10. Data Flow

1. User selects dataset in Sample Bay
2. `scan_datasets` reads `data/index.json`
3. `dataset_preview` calls CLI `vitals` command, extracts 13 features
4. User clicks PLAY/BATTLE in Toolbar or Pipeline Mixer
5. `run_battle` spawns CLI process, 5 topologies compete over 5 rounds
6. Results flow into Zustand store
7. Topology Scope renders persistence diagrams, Betti curves, 3D manifold, convergence
8. Parameter Roll shows governor config, battle ranking, vitals
9. Experiment Timeline records the run for recall

---

## 11. Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | React 18, TypeScript, Vite, Tailwind CSS |
| State | Zustand |
| Graphs | ReactFlow |
| 3D | React Three Fiber, Three.js, Drei |
| Charts | Canvas 2D API |
| Backend | Tauri 2.0 (Rust) |
| Compute | Python 3.11 (NumPy, SciPy) |
| IPC | Tauri invoke → subprocess JSON stdout |

---

## 12. Plugin API

LAAMBA uses a **manifest-driven plugin system**. No recompilation needed to add engines.

1. Create `engines/myengine.toml`
2. Create compute script (Python or compiled binary)
3. Auto-discovered on app restart

See `PLUGIN_API.md` for full specification.

---

## 13. Implementation Plan (7 Phases)

### Phase 1 — Skeleton (Days 1–3)
- Tauri 2.0 scaffold, React 18 + Vite + Tailwind
- Panel layout with react-resizable-panels
- Zustand store with all types

### Phase 2 — Backend Bridge (Days 4–6)
- Python CLI bridge (`governor_cli.py`)
- Rust IPC commands (`commands.rs`)
- Dataset scanning and vitals extraction

### Phase 3 — Core Engines (Days 7–10)
- `TopologicalGovernor` (PolicyNet MLP)
- `GovernorOrchestrator` (Battle Royale)
- `VitalsExtractor` (13 features)

### Phase 4 — Visualization (Days 11–14)
- Topology Scope: Persistence, Betti, 3D Manifold, Convergence
- Parameter Roll: vitals + rankings
- Experiment Timeline: history + recall

### Phase 5 — Pipeline System (Days 15–18)
- ReactFlow DAG editor
- Pipeline templates
- Topological sort executor

### Phase 6 — Engine Integration (Days 19–22)
- TOML manifest parser
- Engine registry
- Engine manager (subprocess spawning)

### Phase 7 — Polish (Days 23–25)
- Dark theme refinement
- Console panel
- Build verification, README, docs

---

## 14. Success Criteria

- [x] App launches and loads real datasets
- [x] 14 engines load from TOML manifests
- [x] Vitals extraction works (13 features)
- [x] Analysis works (Governor predicts manifold)
- [x] Battle Royale works (5 topologies compete)
- [x] Ranking works (manifold probabilities)
- [x] All visualization panels render real data
- [x] Drag-drop CSV import
- [x] Experiment history with result recall
- [x] `cargo check` passes with 0 errors
- [x] `npm run preview` produces working build

---

*Built by seal. Topology is the shape of truth.*
