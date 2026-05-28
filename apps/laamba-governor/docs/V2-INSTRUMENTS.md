# V2 — Instruments (Live Topology Tools)

> For the next AI/developer working on this: this document specifies exactly what to build.
> The V1 app is a Tauri 2.0 (Rust backend) + React 18 + TypeScript frontend.
> State lives in `src/store.ts` (Zustand). CLI bridge is `cli/governor_cli.py`.
> All visualization currently uses Canvas 2D + React Three Fiber.

## 1. Live Embedding Viewer

**What**: Real-time 3D animation showing how points move during embedding optimization. Currently `TopologyScope.tsx` manifold mode generates static synthetic points. This should show actual optimization trajectory.

**Implementation plan**:
- Add new CLI command `python cli/governor_cli.py embed <path> <topology>` that runs embedding optimization and outputs JSON frames:
  ```json
  {"frame": 0, "points": [[x,y,z], ...], "loss": 0.45, "topology": "hyperbolic_poincare"}
  {"frame": 1, "points": [[x,y,z], ...], "loss": 0.38, "topology": "hyperbolic_poincare"}
  ```
- In `governor_orchestrator.py`, the `TopologyRunner._embed()` method already computes embeddings. Modify it to yield intermediate states.
- On the frontend, create `src/panels/EmbeddingViewer.tsx` using React Three Fiber:
  - Receive frames via Tauri streaming (or poll)
  - Animate point positions with `useFrame()` hook
  - Color points by cluster assignment
  - Show loss curve overlay (bottom-left HUD)
  - Orbit controls for user camera
- Add to `TopologyScope.tsx` as a 5th mode tab: "EMBEDDING"

**Key files to modify**:
- `cli/governor_cli.py` — add `cmd_embed()` 
- `governor_orchestrator.py` — `TopologyRunner._embed()` yield intermediates
- `src-tauri/src/ipc/commands.rs` — add `run_embed` command
- `src/panels/TopologyScope.tsx` — add embedding tab
- New: `src/panels/EmbeddingViewer.tsx`

**Dependencies**: Already installed — `@react-three/fiber`, `three`, `@react-three/drei`

---

## 2. Curvature Heatmap

**What**: Color-mapped visualization of estimated sectional curvature over the point cloud surface. Red = positive curvature (sphere-like), Blue = negative curvature (saddle-like), White = flat.

**Implementation plan**:
- Add CLI command `python cli/governor_cli.py curvature <path>` outputting per-point curvature estimates:
  ```json
  {"points": [[x,y,z], ...], "curvatures": [0.1, -0.3, ...], "min": -0.5, "max": 0.8}
  ```
- Curvature estimation algorithm:
  1. Build k-NN graph (k=15)
  2. For each point, fit local PCA to neighborhood
  3. Estimate curvature via angle deficit in Voronoi dual
  4. Formula: `K(p) = (2*pi - sum(angles)) / area(voronoi_cell(p))`
- Frontend: 3D scatter with per-vertex coloring via `THREE.BufferAttribute` on color channel
- Color map: `curvature → HSL(240*(1-t), 100%, 50%)` where t normalized to [0,1]

**Key formulas**:
```
Sectional curvature at point p:
K(p) = (2π - Σ θ_i) / A(p)

where θ_i are angles of triangles incident to p
and A(p) is the area of the Voronoi cell of p

Gauss-Bonnet theorem (verification):
Σ K(p_i) * A(p_i) = 2π * χ(M)
```

**Key files**:
- New: `cli/curvature_estimator.py`
- `cli/governor_cli.py` — add `cmd_curvature()`
- `src-tauri/src/ipc/commands.rs` — add `run_curvature`
- New: `src/panels/CurvatureHeatmap.tsx`

---

## 3. Persistence Barcode (Interactive)

**What**: A barcode diagram (horizontal bars from birth to death) with an interactive filtration slider. Dragging the slider shows which topological features are alive at that epsilon.

**Implementation plan**:
- Currently `TopologyScope.tsx` PersistenceDiagram generates synthetic persistence points from vitals. Replace with real persistent homology computation.
- Add CLI command `python cli/governor_cli.py persistence <path>` using ripser or gudhi:
  ```json
  {
    "H0": [[birth, death], ...],
    "H1": [[birth, death], ...],
    "H2": [[birth, death], ...]
  }
  ```
- Dependencies needed: `pip install ripser` or `pip install gudhi`
- Frontend: Canvas 2D barcode renderer with:
  - Horizontal bars for each feature (color-coded by dimension)
  - Vertical epsilon slider
  - Highlight bars that cross the slider position
  - Click a bar to highlight corresponding points in 3D viewer

**Ripser integration** (Python):
```python
from ripser import ripser
result = ripser(data, maxdim=2)
diagrams = result['dgms']  # List of (birth, death) arrays per dimension
```

**Key files**:
- `cli/governor_cli.py` — add `cmd_persistence()`
- `src/panels/TopologyScope.tsx` — replace synthetic with real
- New: `src/panels/PersistenceBarcode.tsx`

---

## 4. Spectral Analyzer

**What**: Eigenvalue spectrum of the graph Laplacian. Shows spectral gap, number of connected components (zero eigenvalues), and the Fiedler vector.

**Implementation plan**:
- CLI command `python cli/governor_cli.py spectrum <path> --k=50`:
  ```json
  {
    "eigenvalues": [0.0, 0.001, 0.05, ...],
    "spectral_gap": 0.05,
    "fiedler_vector": [0.1, -0.3, ...],
    "n_components": 1
  }
  ```
- Algorithm:
  1. Build k-NN graph
  2. Compute graph Laplacian: `L = D - A`
  3. Find smallest k eigenvalues via `scipy.sparse.linalg.eigsh`
  4. Spectral gap = λ₁ (smallest nonzero eigenvalue)
- Frontend: Plotly or Canvas line chart of sorted eigenvalues, with spectral gap annotated

**Key formulas**:
```
Graph Laplacian: L = D - A
  D = degree matrix (diagonal)
  A = adjacency matrix

Normalized Laplacian: L_norm = D^{-1/2} L D^{-1/2}

Spectral gap: λ₁ = min nonzero eigenvalue of L
  - Large λ₁ → well-connected (Euclidean-like)
  - Small λ₁ → tree-like structure (hyperbolic)

Cheeger inequality: λ₁/2 ≤ h(G) ≤ √(2λ₁)
  where h(G) is the Cheeger constant
```

---

## 5. Transfer Learning Bus

**What**: Hot-swap embeddings between topologies mid-run. Start with Euclidean, transfer to Hyperbolic, compare.

**Implementation plan**:
- The `governor_orchestrator.py` already has `TransferBus` class with `transfer()` method
- Add CLI command `python cli/governor_cli.py transfer <path> --from=euclidean --to=hyperbolic`
- Frontend: Visual bus connecting topology nodes in Pipeline Mixer, with animated particle flow showing transfer

**Already exists in code** (`governor_orchestrator.py`):
```python
class TransferBus:
    def transfer(self, embedding, from_topo, to_topo, data):
        # Projects embedding from one geometry to another
```

---

## 6. Topology Fusion Console

**What**: Weighted blend of multiple manifold embeddings. The `TopologyFusion` class already exists.

**Already exists** (`governor_orchestrator.py`):
```python
class TopologyFusion:
    def fuse(self, embeddings, weights):
        # Weighted average of normalized embeddings
```

**Implementation**: Expose via CLI, add UI panel showing fusion weights as sliders, output blended embedding to 3D viewer.
