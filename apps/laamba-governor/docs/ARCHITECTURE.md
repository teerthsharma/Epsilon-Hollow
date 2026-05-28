# Architecture

```
+------------------------------------------------------------------+
|                        LAAMBA GOVERNOR                            |
|                     Desktop Application                          |
+------------------------------------------------------------------+
|                                                                  |
|  +------------------+  +------------------+  +-----------------+ |
|  |   Sample Bay     |  |  Pipeline Mixer  |  |  Engine Rack    | |
|  |                  |  |                  |  |                 | |
|  |  Load datasets   |  |  ReactFlow DAG   |  |  14 engines     | |
|  |  CSV point clouds |  |  Battle Royale   |  |  TOML manifests | |
|  |  Real vitals      |  |  5 topologies    |  |  Run analysis   | |
|  +--------+---------+  +--------+---------+  +--------+--------+ |
|           |                      |                      |         |
|           v                      v                      v         |
|  +----------------------------------------------------------+    |
|  |                    Zustand Store                          |    |
|  |  datasets | vitals | analysis | battle | experiments      |    |
|  +---------------------------+-------------------------------+    |
|                              |                                    |
|  +------------------+  +----v-------------+  +-----------------+ |
|  | Topology Scope   |  | Parameter Roll   |  | Experiment      | |
|  |                  |  |                  |  | Timeline        | |
|  |  Persistence     |  |  Governor config |  |                 | |
|  |  Betti curves    |  |  Battle ranking  |  |  Run history    | |
|  |  3D manifold     |  |  Topological     |  |  Result recall  | |
|  |  Convergence     |  |  vitals          |  |  Status badges  | |
|  +------------------+  +------------------+  +-----------------+ |
|                                                                  |
+----------------------------+-------------------------------------+
                             |
                    Tauri IPC (invoke)
                             |
+----------------------------v-------------------------------------+
|                     Rust Backend                                  |
|                                                                  |
|  commands.rs:                                                    |
|    scan_datasets  -> reads data/index.json                       |
|    dataset_preview -> python cli/governor_cli.py vitals <path>   |
|    run_analysis   -> python cli/governor_cli.py analyze <path>   |
|    run_battle     -> python cli/governor_cli.py battle <path>    |
|    run_rank       -> python cli/governor_cli.py rank <path>      |
|    scan_engines   -> reads engines/*.toml                        |
|                                                                  |
+----------------------------+-------------------------------------+
                             |
                    subprocess (stdout JSON)
                             |
+----------------------------v-------------------------------------+
|                   Python Topology Engine                          |
|                                                                  |
|  governor_cli.py          topological_governor_ml.py             |
|    cmd_vitals()             VitalsExtractor (13 features)        |
|    cmd_analyze()            PolicyNet (2-layer MLP)              |
|    cmd_battle()             TopologicalGovernor                  |
|    cmd_rank()               ManifoldConfig                       |
|                                                                  |
|  governor_orchestrator.py                                        |
|    ComputeKernel           TopologyRunner                        |
|    GovernorOrchestrator    TopologyFusion                         |
|    TransferBus             OutcomeComparator                     |
+------------------------------------------------------------------+
```

## Data Flow

1. **User selects dataset** in Sample Bay
2. `scan_datasets` reads `data/index.json`, returns dataset list with absolute paths
3. `dataset_preview` calls CLI `vitals` command, extracts 13 topological features
4. **User clicks PLAY/BATTLE** in Toolbar or Pipeline Mixer
5. `run_battle` spawns CLI process, 5 topologies compete over 5 rounds
6. Results flow into Zustand store
7. Topology Scope renders persistence diagrams, Betti curves, 3D manifold, convergence
8. Parameter Roll shows governor config, battle ranking, vitals
9. Experiment Timeline records the run for recall

## Tech Stack

| Layer      | Technology                        |
|------------|-----------------------------------|
| Frontend   | React 18, TypeScript, Vite        |
| State      | Zustand                           |
| Graphs     | ReactFlow                         |
| 3D         | React Three Fiber, Three.js, Drei |
| Charts     | Canvas 2D API                     |
| Styling    | Tailwind CSS                      |
| Backend    | Tauri 2.0 (Rust)                  |
| Compute    | Python 3.11 (NumPy, SciPy)        |
| IPC        | Tauri invoke → subprocess JSON    |
