# AI Handoff Guide

> Read this if you're an AI continuing work on LAAMBA GOVERNOR.

## Quick Context

**Owner**: Teerth Sharma (@seal), 19, Bangalore. Building a research workstation for topological data analysis.

**Metaphor**: FL Studio for topology. Datasets are samples. Engines are instruments. Pipeline is the mixer. Topology Scope is the oscilloscope.

**Stack**: Tauri 2.0 (Rust) + React 18 + TypeScript + Python CLI

## Current State (V1)

Working:
- App launches, loads real datasets from `data/index.json`
- 14 engines load from `engines/*.toml`
- Vitals extraction works (click dataset → 13 topological features)
- Analysis works (Governor predicts manifold)
- Battle Royale works (5 topologies compete)
- Ranking works (manifold probabilities)
- All visualization panels render real data
- Drag-drop CSV import (needs dialog plugin permission fix — see below)
- Drag datasets onto engines to run analysis
- Experiment history with result recall

Known issues:
- `dialog:default` permission may need app restart (not just hot-reload) to take effect
- Large datasets (>10k points) may timeout the CLI subprocess
- No streaming — long computations block until complete
- The 3D manifold viewer generates synthetic point distributions based on topology type, not actual embeddings

## File Map

```
THE CRITICAL FILES:
src/store.ts                    ← ALL app state (Zustand). Read this first.
src-tauri/src/ipc/commands.rs   ← ALL backend commands. The Rust<->Python bridge.
cli/governor_cli.py             ← ALL CLI commands. Python topology engine.
topological_governor_ml.py      ← The Governor brain (VitalsExtractor + PolicyNet)
governor_orchestrator.py        ← Battle Royale system

FRONTEND PANELS:
src/App.tsx                     ← Layout + dataset auto-scan on mount
src/panels/SampleBay.tsx        ← Dataset list + drag-drop import
src/panels/EngineRack.tsx       ← Engine list + run buttons + drop target
src/panels/PipelineMixer.tsx    ← ReactFlow graph + BATTLE button
src/panels/TopologyScope.tsx    ← 4-mode visualization (Canvas 2D + Three.js)
src/panels/ParameterRoll.tsx    ← Vitals + config display
src/panels/ExperimentTimeline.tsx ← Run history
src/panels/ConsolePanel.tsx     ← Log output
src/components/Toolbar.tsx      ← PLAY/STOP/RANK buttons

BACKEND:
src-tauri/src/lib.rs            ← Tauri app setup + command registration
src-tauri/src/engine/           ← Engine manifest parsing
src-tauri/capabilities/default.json ← Plugin permissions
```

## How Data Flows

```
User clicks dataset → invoke("dataset_preview", {path}) 
  → Rust spawns `python cli/governor_cli.py vitals <path>`
  → Python outputs JSON to stdout
  → Rust parses JSON, returns to frontend
  → Zustand store updates vitalsResult
  → TopologyScope re-renders persistence/betti

User clicks BATTLE → invoke("run_battle", {path})
  → Same pattern, but `battle` command
  → Returns rounds, scores, rankings
  → TopologyScope convergence view shows score history
  → PipelineMixer highlights winner node
  → ExperimentTimeline records the run
```

## How to Add a New Feature

### New CLI command
1. Add function in `cli/governor_cli.py`: `def cmd_foo(path):`
2. Register in `main()`: `elif command == "foo": cmd_foo(path)`
3. Add Tauri command in `commands.rs`: `pub async fn run_foo(path: String) -> Result<Value, String>`
4. Register in `lib.rs` invoke_handler
5. Call from frontend: `await invoke("run_foo", { path: ds.path })`

### New visualization mode
1. Add component in `src/panels/TopologyScope.tsx` (or new file)
2. Add tab to MODES array
3. Read from Zustand store (vitalsResult, analysisResult, battleResult)

### New engine
1. Create `engines/myengine.toml` (see existing ones for format)
2. The EngineRack auto-discovers it on scan

## Branding

- Seal's personal project — keep "by seal" credit visible
- Color: #00E5FF (cyan accent), dark theme (#0d0d0d bg)
- Tone: research instrument, not enterprise software
- The README has the full project description

## What Needs Work Next

See `docs/SCOPE.md` for the full roadmap. Priority order:

1. **Fix dialog permissions** — may just need `npm run tauri dev` restart
2. **Real persistent homology** — install `ripser`, replace synthetic PH diagrams with real computation
3. **Streaming results** — use Tauri events to stream CLI output line-by-line instead of waiting for completion
4. **V2 instruments** — see `docs/V2-INSTRUMENTS.md`
5. **V3 playgrounds** — see `docs/V3-FORMULAS-PLAYGROUNDS.md`
