# LAAMBA GOVERNOR — Design Specification

> **The Digital Analysis Workstation for Topological Research**
> Author: Teerth Sharma | Date: 2026-05-26

---

## 1. THESIS

Teerth Sharma has built 14+ engines across topology, physics, quantum computing,
systems programming, and ML. Each engine proves one piece of a larger theory:
**the universe is a topological object.**

- **Faraday** proves E and H fields converge to a single Banach fixed point (the God Tensor)
- **Hamilton** extends that convergence to N-body quantum states
- **OmniTopos** traces the cosmological phase sequence: VACUUM -> H0 -> H0H1 -> H0H1H2 -> GOD_FIXED
- **TopoBridge-Q** proves homological information transfer on real IBM quantum hardware
- **Lambda-Topo** stores topological signatures as memory
- **Epsilon-Hollow** runs topology as an operating system
- **topo-asm** makes persistent homology fast enough for real-time (43.8x over ripser)
- **Aether-Lang** gives topology its own programming language

These are not 14 separate projects. They are one theory, fractured across 14 repositories.

**The Governor is the instrument that makes the theory run end-to-end.**

Like FL Studio turned isolated synthesizers into compositions, the Governor turns
isolated topological proofs into a single composable research apparatus for
investigating the topological structure of reality.

---

## 2. EXISTING CODEBASE (to absorb, not replace)

The repo already contains significant groundwork:

| File | What it does | Absorb into |
|------|-------------|-------------|
| `topological_governor_ml.py` | VitalsExtractor, ManifoldConfig, policy network | Tauri backend: engine/vitals.rs |
| `governor_orchestrator.py` | Battle Royale (runners, transfer bus, comparator, fusion) | Tauri backend: pipeline/executor.rs |
| `topological_governor_full.py` | 40 use cases as callable functions | Sample presets / example pipelines |
| `governor-studio/main.py` | FastAPI + canvas node graph prototype | Reference for API design |
| `governor-studio/static/index.html` | Canvas-based UI prototype | Reference for visual style |
| `GOVERNOR_USECASES.md` | 40 use cases documented | Built-in use case templates |

The Python engines remain callable via CLI subprocess. The Tauri backend wraps them.

---

## 3. TECH STACK

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| Backend | **Tauri 2.0 (Rust)** | Native performance, Rust backend matches existing Rust engines |
| Frontend | **React 18 + TypeScript + Vite** | 9.4M JS + 6M TS in Teerth's repos; largest sci-viz ecosystem |
| Node Graph | **ReactFlow** | Industry standard for visual wiring |
| 3D Visualization | **React Three Fiber** (Three.js) | Declarative 3D for manifold point clouds |
| 2D Scientific Plots | **Plotly.js** | Persistence diagrams, convergence curves, Betti plots |
| State Management | **Zustand** | Lightweight, fast, minimal boilerplate |
| Styling | **Tailwind CSS + custom dark theme** | FL Studio aesthetic with neon accents |
| Panel Docking | **rc-dock** or custom | Drag-to-rearrange, collapse, float panels |

---

## 4. THE SIX PANELS

The Governor has six core panels, each mapping to an FL Studio concept:

### 4.1 SAMPLE BAY (FL Studio: Browser Panel)

**Purpose:** Import, browse, preview, and clean datasets before feeding them to engines.

Musicians import samples, loops, and stems. Researchers import datasets.

**Data types ("samples"):**

| Type | Extensions | Preview |
|------|-----------|---------|
| Point clouds | .npy, .csv, .ply, .xyz | 3D scatter plot + quick Betti numbers |
| Meshes/Geometries | .obj, .stl, .step | 3D wireframe render |
| Tensors/Matrices | .npy, .pt, .safetensors | Shape + heatmap |
| Tabular data | .csv, .parquet, .json | Stats table + distribution plot |
| Simulation outputs | .hdf5, .npz | Shape + metadata |
| Previous run results | auto-indexed | Full topology scope snapshot |

**Import sources:**

| Source | Method |
|--------|--------|
| Local filesystem | Drag-and-drop or folder watch |
| URL fetch | Paste URL, auto-download |
| arXiv supplementary | Search by paper ID, pull data |
| Kaggle API | Search and pull (requires API key) |
| Previous experiment results | Auto-indexed from Experiment Timeline |

**Transform pipeline (pre-processing before engines):**

| Transform | Description |
|-----------|-------------|
| Clean | Remove NaN, outliers, duplicates |
| Project | JL random projection to lower dimensions |
| Normalize | L2 normalize to S^2 (sphere) |
| Subsample | Random or farthest-point sampling |
| Augment | Add noise via Epsilon-CLI stochastic resonance |
| Split | Train/test/val partitions |
| PCA | Dimensionality reduction via eigendecomposition |

**Interactions:**
- Drag dataset onto Engine Rack tile -> auto-fills engine input
- Drag dataset into Pipeline Mixer -> becomes source node
- Hover any file -> sidebar shows shape, quick Betti numbers, 3D thumbnail, basic stats
- Right-click -> open in external tool, copy path, delete

### 4.2 ENGINE RACK (FL Studio: Channel Rack)

**Purpose:** Grid of all engines with status indicators and quick controls.

**Engine manifest format (engine.toml):**

```toml
[engine]
name = "Faraday"
category = "Physics"
icon = "zap"
color = "#FF6B35"
description = "God Tensor via Banach fixed-point E<->H coupling"

[engine.entry]
type = "cli"
command = "python -m faraday.cli"
working_dir = "~/repos/faraday"

[engine.inputs]
geometry = { type = "mesh", desc = "Cavity geometry" }
resolution = { type = "int", default = 64, range = [8, 512] }
epochs = { type = "int", default = 50000 }

[engine.outputs]
god_tensor = { type = "tensor", shape = [16] }
betti_curve = { type = "timeseries" }
banach_loss = { type = "scalar" }

[engine.metrics]
convergence = { source = "stdout", pattern = "loss: ([0-9.e-]+)" }
betti_0 = { source = "stdout", pattern = "B0=([0-9.]+)" }
```

**Engine categories:**

| Category | Color | Engines |
|----------|-------|---------|
| Physics | #FF6B35 (red/orange) | Faraday, Hamilton, OmniTopos |
| Topology | #AA00FF (purple) | Lambda-Topo, TopoFlow, topo-asm |
| Systems | #00B0FF (blue) | Epsilon-Hollow, hollow-asm, pgtable-asm, vec-simd |
| Runtime | #00E676 (green) | Aether-Lang, Epsilon-CLI |
| Quantum | #FFD600 (gold) | TopoBridge-Q |
| Audio/ML | #FF4081 (pink) | NeuralWhisper |
| Memory | #B0BEC5 (silver) | phi-mem |

**Engine tile UI:**

Each tile shows:
- Engine name + category color bar
- Power toggle (ON/OFF) + status LED (idle/running/done/error)
- Progress bar (epoch/total or percentage)
- 2-3 live metric values parsed from stdout
- Action buttons: Run, Config (opens Parameter Roll), Scope (opens Topology Scope)

**Interactions:**
- Click tile -> opens Parameter Roll for that engine
- Drag tile into Pipeline Mixer -> creates engine node
- Right-click -> clone config, reset, view full logs, open repo in VS Code
- Double-click status LED -> open console filtered to this engine

### 4.3 PIPELINE MIXER (FL Studio: Mixer + Routing)

**Purpose:** Visual node graph for composing data flow between engines.

Built on ReactFlow.

**Node types:**

| Node | Shape | Color | Purpose |
|------|-------|-------|---------|
| DataSource | Rounded rect | Orange | Import from Sample Bay |
| Engine | Colored rect | Category color | Run an engine |
| Transform | Diamond | White | Data transform (clean, project, normalize) |
| Splitter | Triangle | Gray | Fork one output to multiple inputs |
| Merger | Inverted triangle | Gray | Combine multiple outputs |
| Comparator | Hexagon | Yellow | Compare results, pick winner (Battle Royale) |
| Fusion | Pentagon | Purple | Cross-topology attention fusion |
| Checkpoint | Disk icon | Gray | Save intermediate state to disk |
| Viewer | Eye icon | Red | Tap into wire to visualize data |
| Export | Arrow-out | Gray | Export results to file |

**Wire types (color-coded by data type):**

| Data Type | Wire Color | Wire Style |
|-----------|-----------|------------|
| Tensor/Matrix | Red (#FF4444) | Thick solid |
| Scalar/Metric | Green (#00E676) | Thin solid |
| Timeseries | Blue (#00B0FF) | Dashed |
| Point Cloud | Purple (#AA00FF) | Dotted |
| File/Path | Gray (#888888) | Thin solid |

**Type checking:** Wires only connect when output type matches input type.
Mismatched connections show a red X tooltip explaining the mismatch.

**Pipeline execution:**
1. Topological sort of the node DAG
2. Execute nodes in dependency order
3. Stream metrics to Topology Scope as each node runs
4. On completion, results available at Viewer/Export nodes

**Pre-built pipeline templates (from the 40 use cases):**
- Faraday -> Hamilton -> OmniTopos (full theory chain)
- Data -> VitalsExtractor -> Governor -> Best Topology (auto-pick)
- Data -> 5 Topologies -> Comparator (Battle Royale)
- Data -> Topologies -> Fusion (cross-topology attention)

### 4.4 EXPERIMENT TIMELINE (FL Studio: Playlist)

**Purpose:** Sequence experiments over time, track history, enable reproducibility.

**Visual layout:**
- Horizontal timeline with time on X-axis
- Each engine run is a colored block on its own lane
- Blocks can have dependency arrows (Faraday -> Hamilton means "wait")
- Blocks are draggable to reorder/reschedule
- Scroll left to see past experiments

**Features:**
- **Run history**: Every experiment auto-saved with full config + results
- **Snapshots**: Bookmark any point in time, restore full state later
- **Compare mode**: Select 2+ experiments, see side-by-side results in Topology Scope
- **Replay**: Click any past experiment block to restore its results
- **Queue**: Drag blocks to future positions to schedule runs
- **Dependencies**: Draw arrows between blocks to create sequential pipelines

**Data model:**

```typescript
interface Experiment {
  id: string;
  name: string;
  pipeline: PipelineGraph;     // full node graph snapshot
  params: Record<string, any>; // all engine parameters
  results: ExperimentResults;  // outputs, metrics, timeseries
  timestamp: number;
  duration_ms: number;
  status: 'queued' | 'running' | 'completed' | 'failed';
}
```

### 4.5 PARAMETER ROLL (FL Studio: Piano Roll)

**Purpose:** Deep editing for a single engine's configuration.

Opens when clicking an Engine Rack tile or double-clicking an engine node in Pipeline Mixer.

**Control types:**

| Control | For | Example |
|---------|-----|---------|
| Slider | Numeric params | Resolution: 8-512, Epochs: 1-100000 |
| Curve editor | Scheduled params | Epsilon-CLI noise gain schedule over time |
| Dropdown | Categorical | Geometry type, solver method, manifold choice |
| Toggle | Boolean | auto_transfer, verbose logging |
| Code editor | Scripts | Aether-Lang .aether files (Monaco editor) |
| Matrix editor | 2D params | Coupling matrices, filter banks |

**Features:**
- **Presets**: Save/load parameter sets ("micro-42", "medium-42", "production")
- **Diff view**: Compare two presets side-by-side
- **Randomize**: Randomize params within valid ranges for exploration
- **Lock**: Lock specific params so they survive preset changes
- **History**: Undo/redo parameter changes

### 4.6 TOPOLOGY SCOPE (FL Studio: Oscilloscope / Spectrum Analyzer)

**Purpose:** Live and post-hoc visualization of topological data.

**Six visualization modes:**

| Mode | What it shows | Library | When to use |
|------|---------------|---------|-------------|
| Persistence Diagram | Birth-death scatter, colored by dimension | Plotly.js | After ripser/topo-asm runs |
| Betti Curves | Beta_0, Beta_1, Beta_2 over filtration | Plotly.js | Watching topology emerge |
| Manifold View | 3D point cloud on S^2 or in R^3, rotatable | React Three Fiber | Inspecting embeddings |
| Convergence Plot | Banach loss / metric over epochs, log scale | Plotly.js | During Faraday/Hamilton runs |
| Barcode View | Horizontal bars (birth->death per feature) | Custom SVG | Persistence summary |
| Phase Portrait | OmniTopos phase state progression | Custom SVG/Canvas | During OmniTopos runs |

**Live mode:** When an engine is running, the Scope streams its metrics in real-time
via Tauri events. Convergence curves update live.

**Snap mode:** After a run completes, the Scope shows the final state. Click any
block in the Experiment Timeline to snap to that moment's data.

**Multi-scope:** Can open multiple Scope panels to watch different metrics simultaneously.

---

## 5. RUST BACKEND (TAURI)

### 5.1 Module Structure

```
src-tauri/src/
  main.rs                    -- Tauri app entry, window management
  lib.rs                     -- Module declarations
  engine/
    mod.rs                   -- Module exports
    registry.rs              -- Discover engine.toml files, build engine catalog
    manager.rs               -- Spawn/kill/monitor engine subprocesses
    parser.rs                -- Parse stdout lines for metrics via regex
    manifest.rs              -- engine.toml schema and deserialization
  pipeline/
    mod.rs
    graph.rs                 -- Pipeline data structures (nodes, edges, types)
    executor.rs              -- Topological sort + sequential DAG execution
    types.rs                 -- Data type definitions (tensor, scalar, timeseries, etc.)
  timeline/
    mod.rs
    history.rs               -- Experiment storage (SQLite or JSON files)
    snapshot.rs              -- State snapshots for reproducibility
  sample/
    mod.rs
    browser.rs               -- File system scanning, dataset discovery
    preview.rs               -- Quick stats, shape info for datasets
    transform.rs             -- Clean, project, normalize, subsample operations
    import.rs                -- URL fetch, file import handlers
  ipc/
    mod.rs
    commands.rs              -- All Tauri commands exposed to frontend
    events.rs                -- Event types for metric streaming
  state.rs                   -- Global app state
```

### 5.2 Key Tauri Commands

```rust
// Engine lifecycle
#[tauri::command] fn scan_engines(paths: Vec<String>) -> Vec<EngineManifest>;
#[tauri::command] fn engine_start(id: String, params: Value) -> Result<String, String>;
#[tauri::command] fn engine_stop(id: String) -> Result<(), String>;
#[tauri::command] fn engine_status(id: String) -> EngineStatus;

// Pipeline
#[tauri::command] fn pipeline_validate(graph: PipelineGraph) -> ValidationResult;
#[tauri::command] fn pipeline_run(graph: PipelineGraph) -> Result<String, String>;
#[tauri::command] fn pipeline_stop(run_id: String) -> Result<(), String>;

// Sample Bay
#[tauri::command] fn scan_datasets(paths: Vec<String>) -> Vec<DatasetInfo>;
#[tauri::command] fn dataset_preview(path: String) -> DatasetPreview;
#[tauri::command] fn dataset_transform(path: String, ops: Vec<TransformOp>) -> Result<String, String>;
#[tauri::command] fn import_url(url: String, dest: String) -> Result<String, String>;

// Timeline
#[tauri::command] fn experiments_list() -> Vec<ExperimentSummary>;
#[tauri::command] fn experiment_get(id: String) -> Experiment;
#[tauri::command] fn experiment_save(exp: Experiment) -> Result<String, String>;
#[tauri::command] fn experiment_compare(ids: Vec<String>) -> ComparisonResult;

// Metrics streaming (via Tauri events, not commands)
// Backend emits: "metric-update", "engine-status-change", "pipeline-progress"
```

### 5.3 Data Flow

```
Engine subprocess stdout
  -> Tauri reads line-by-line (tokio::process::Command)
  -> parser.rs extracts numbers via regex patterns from engine.toml
  -> Tauri event emitted: { engine_id, metric_name, value, timestamp }
  -> Frontend React receives via listen()
  -> Zustand store updates
  -> Plotly/Three.js re-renders
```

### 5.4 Python Interop

The existing Python code (topological_governor_ml.py, governor_orchestrator.py,
topological_governor_full.py) is called via subprocess:

```rust
// Example: running the TopologicalGovernor
let output = Command::new("python")
    .args(&["-m", "topological_governor_ml"])
    .arg("--input").arg(data_path)
    .arg("--output").arg(result_path)
    .stdout(Stdio::piped())
    .spawn()?;
```

For the built-in vitals extraction and topology prediction, a subset of the
Python logic is ported to Rust for zero-overhead hot-path performance.

---

## 6. FRONTEND STRUCTURE

```
src/
  App.tsx                    -- Root component, panel layout
  main.tsx                   -- Entry point
  layouts/
    DockLayout.tsx           -- Panel docking system (drag, collapse, float)
    PanelFrame.tsx           -- Shared panel chrome (title bar, minimize, close)
  panels/
    SampleBay.tsx            -- Dataset browser + import + transforms
    EngineRack.tsx           -- Engine grid with tiles
    PipelineMixer.tsx        -- ReactFlow node graph
    ExperimentTimeline.tsx   -- Horizontal timeline with blocks
    ParameterRoll.tsx        -- Deep engine config editor
    TopologyScope.tsx        -- Visualization panel (6 modes)
    ConsolePanel.tsx         -- Raw logs and command entry
  components/
    engine/
      EngineTile.tsx         -- Single engine tile in rack
      EngineStatusLED.tsx    -- Colored status indicator
      MetricBadge.tsx        -- Live metric display
    pipeline/
      NodeGraph.tsx          -- ReactFlow wrapper with custom nodes
      EngineNode.tsx         -- Custom ReactFlow node for engines
      DataSourceNode.tsx     -- Custom node for data sources
      TransformNode.tsx      -- Custom node for transforms
      ComparatorNode.tsx     -- Custom node for Battle Royale
      WireRenderer.tsx       -- Custom edge renderer (color-coded)
    scope/
      PersistenceDiagram.tsx -- Plotly scatter (birth vs death)
      BettiCurves.tsx        -- Plotly line chart (beta_0,1,2 vs filtration)
      ManifoldViewer.tsx     -- React Three Fiber 3D point cloud
      ConvergencePlot.tsx    -- Plotly log-scale line chart
      BarcodeView.tsx        -- Custom SVG horizontal bars
      PhasePortrait.tsx      -- Custom SVG/Canvas phase state
    timeline/
      TimelineBlock.tsx      -- Single experiment block
      DependencyArrow.tsx    -- Arrow between dependent blocks
    sample/
      DatasetCard.tsx        -- File entry in Sample Bay
      PreviewPanel.tsx       -- Stats + thumbnail on hover
      TransformChain.tsx     -- Transform pipeline builder
    params/
      SliderControl.tsx      -- Numeric parameter slider
      CurveEditor.tsx        -- Scheduled parameter curve
      PresetSelector.tsx     -- Preset save/load
  hooks/
    useEngine.ts             -- Engine state + lifecycle hook
    useMetricStream.ts       -- Subscribe to live metric events from Tauri
    usePipeline.ts           -- Pipeline state + execution hook
    useTimeline.ts           -- Experiment history hook
    useDataset.ts            -- Dataset browser hook
  stores/
    engineStore.ts           -- Zustand: engine registry, status, metrics
    pipelineStore.ts         -- Zustand: node graph, connections, execution state
    timelineStore.ts         -- Zustand: experiment history, current run
    sampleStore.ts           -- Zustand: dataset catalog, transforms
    scopeStore.ts            -- Zustand: visualization mode, data bindings
    uiStore.ts               -- Zustand: panel layout, theme, preferences
  styles/
    governor.css             -- FL Studio dark theme, neon accents
    variables.css            -- CSS custom properties
  types/
    engine.ts                -- Engine manifest, status, metric types
    pipeline.ts              -- Node, edge, graph types
    experiment.ts            -- Experiment, result types
    dataset.ts               -- Dataset info, preview, transform types
    scope.ts                 -- Visualization mode, data binding types
```

---

## 7. ENGINE MANIFESTS (14 engines)

Each engine gets an `engines/<name>.toml` file. Here are the 14:

### Physics Engines
- `faraday.toml` -- God Tensor via Banach fixed-point
- `hamilton.toml` -- N-body EM coupling via tensor product
- `omnitopos.toml` -- Cosmological phase sequence

### Topology Engines
- `lambda-topo.toml` -- TDA memory with ripser + FAISS
- `topoflow.toml` -- 3D persistence landscape visualization
- `topo-asm.toml` -- AVX-512 persistent homology

### Systems Engines
- `epsilon-hollow.toml` -- Seal OS microkernel
- `hollow-asm.toml` -- Bare-metal topological scheduling
- `vec-simd.toml` -- AVX-512 SIMD math library
- `pgtable-asm.toml` -- x86_64 page table management

### Runtime Engines
- `aether-lang.toml` -- Topology programming language
- `epsilon-cli.toml` -- Stochastic resonance noise injection

### Quantum Engine
- `topobridge-q.toml` -- IBM Quantum homology transfer

### Audio/ML Engine
- `neuralwhisper.toml` -- 82M-param browser TTS

### Memory Engine
- `phi-mem.toml` -- Barcode-signature content-addressable storage

---

## 8. PRE-BUILT PIPELINE TEMPLATES

Loaded from the 40 use cases in GOVERNOR_USECASES.md and topological_governor_full.py:

| Template | Pipeline | Purpose |
|----------|----------|---------|
| Full Theory Chain | Faraday -> Hamilton -> OmniTopos | Test the complete cosmological thesis |
| Topology Auto-Pick | Data -> VitalsExtractor -> Governor -> Best Topology | Let governor choose the right manifold |
| Battle Royale | Data -> [Euclidean, Spherical, Hyperbolic, Grassmannian, Product] -> Comparator | All topologies fight, best wins |
| Fusion | Data -> [All Topologies] -> CrossAttention Fusion | Blend all topologies via attention |
| Quantum Verify | Data -> Faraday -> TopoBridge-Q -> Recovery Measure | Verify topology transfers on quantum hardware |
| EM Cascade | Data -> topo-asm (fast homology) -> Faraday -> God Tensor | Fast homology then coupling |

---

## 9. VISUAL DESIGN

### Color Palette
- Background: #0D0D0D (near black)
- Panel background: #141414
- Border: #2A2A2A
- Text: #E0E0E0
- Text dim: #888888
- Accent: #00E5FF (cyan, like FL Studio)
- Success: #00E676
- Error: #FF1744
- Warning: #FFD600

### Typography
- Font: 'Segoe UI', system-ui, sans-serif
- Monospace: 'JetBrains Mono', 'Cascadia Code', monospace
- Headings: 11px uppercase, letter-spacing 1px (FL Studio style)
- Body: 12px
- Metrics: 13px monospace

### Layout
- Single window with dockable panels
- Default layout: Sample Bay (left) | Pipeline Mixer (center) | Inspector (right) | Console (bottom)
- Engine Rack and Topology Scope float or dock as tabs
- Panels can be dragged, collapsed, floated, or tabbed together
- Full-screen mode for Pipeline Mixer or Topology Scope

---

## 10. FILE STRUCTURE

```
laamba-silence/
  src-tauri/
    Cargo.toml
    tauri.conf.json
    src/
      main.rs
      lib.rs
      engine/
      pipeline/
      timeline/
      sample/
      ipc/
      state.rs
  src/
    App.tsx
    main.tsx
    layouts/
    panels/
    components/
    hooks/
    stores/
    styles/
    types/
  engines/                    -- Engine manifest TOML files
    faraday.toml
    hamilton.toml
    ... (14 total)
  templates/                  -- Pre-built pipeline templates
    full-theory-chain.json
    battle-royale.json
    ... (6+ templates)
  docs/
    plugin-api.md             -- Plugin API documentation
    engine-manifest.md        -- How to write engine.toml files
  legacy/                     -- Existing Python code (preserved)
    topological_governor_ml.py
    governor_orchestrator.py
    topological_governor_full.py
    governor-studio/
  package.json
  tsconfig.json
  vite.config.ts
  tailwind.config.js
  README.md
```

---

## 11. PLUGIN API (documented for future)

The Governor starts with CLI subprocess wrappers. The plugin API is documented
so future engines can integrate more deeply.

**Plugin interface (documented, not implemented in v1):**

```rust
pub trait GovernorPlugin {
    fn name(&self) -> &str;
    fn category(&self) -> EngineCategory;
    fn inputs(&self) -> Vec<PortSpec>;
    fn outputs(&self) -> Vec<PortSpec>;
    fn run(&self, inputs: HashMap<String, DataPayload>) -> Result<HashMap<String, DataPayload>>;
    fn metrics(&self) -> Vec<MetricSpec>;
    fn on_metric(&self, callback: Box<dyn Fn(MetricUpdate)>);
}
```

Documentation lives at `docs/plugin-api.md` with:
- How to write engine.toml (CLI integration)
- How to implement GovernorPlugin (native Rust integration)
- Data type specifications (tensor, scalar, timeseries, point cloud)
- Example: wrapping a Python script as an engine
- Example: writing a native Rust engine

---

## 12. IMPLEMENTATION PHASES

### Phase 1: Skeleton (days 1-3)
- Tauri project scaffolding with Rust backend + React frontend
- Panel docking layout with 6 empty panels
- Dark theme CSS with FL Studio aesthetic
- Engine manifest schema and parser

### Phase 2: Engine Rack + Console (days 4-6)
- Engine discovery (scan for engine.toml files)
- Engine tiles with status LEDs
- Subprocess spawning and stdout capture
- Metric parsing via regex
- Console panel with filtered log output

### Phase 3: Pipeline Mixer (days 7-10)
- ReactFlow integration with custom node types
- Wire type checking and color coding
- Topological sort DAG executor
- Pre-built pipeline templates

### Phase 4: Topology Scope (days 11-14)
- Plotly.js integration for 2D plots
- React Three Fiber for 3D manifold viewer
- Live metric streaming via Tauri events
- Six visualization modes switchable via tabs

### Phase 5: Sample Bay (days 15-17)
- File browser with dataset discovery
- Dataset preview (shape, stats, quick Betti)
- Drag-and-drop into Engine Rack and Pipeline Mixer
- Transform pipeline (clean, project, normalize)

### Phase 6: Timeline + Parameter Roll (days 18-21)
- Experiment Timeline with colored blocks
- Experiment save/load/compare
- Parameter Roll with sliders, curves, presets
- Dependency arrows between timeline blocks

### Phase 7: Polish + Templates (days 22-25)
- Pre-built pipeline templates from 40 use cases
- Keyboard shortcuts
- Window management refinement
- Plugin API documentation
- README and user guide

---

## 13. NON-GOALS (v1)

- Cloud deployment / multi-user
- Real-time collaborative editing
- GPU compute from within Governor (engines handle their own GPU)
- Mobile support
- Plugin hot-loading (documented API, but manual restart in v1)
- Built-in code editor for engine source (use VS Code)

---

## 14. SUCCESS CRITERIA

1. Can wire Faraday -> Hamilton -> OmniTopos and watch the phase sequence emerge live
2. Can import a dataset, run VitalsExtractor, and see the governor pick a manifold
3. Can run a Battle Royale (5 topologies on same data) and see the winner in the Scope
4. Can save an experiment and restore it later with identical results
5. All 14 engines discoverable and launchable from the Engine Rack
6. Live convergence curves during Faraday runs (Banach loss over epochs)
7. 3D point cloud rendering of manifold embeddings in the Scope
8. Dark theme that feels like FL Studio
