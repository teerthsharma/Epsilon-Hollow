# LAAMBA GOVERNOR — Plugin API Specification

> **Version:** 1.0  
> **Date:** 2026-05-26

---

## Overview

LAAMBA uses a **manifest-driven plugin system**. Engines are declared via TOML files in the `engines/` directory. No recompilation of the Rust backend or React frontend is required to add a new engine.

---

## The `GovernorPlugin` Pattern

While engines are currently loaded via TOML manifests, the long-term architecture targets a `GovernorPlugin` trait in Rust:

```rust
pub trait GovernorPlugin: Send + Sync {
    fn name(&self) -> &str;
    fn category(&self) -> &str;
    fn inputs(&self) -> Vec<Port>;
    fn outputs(&self) -> Vec<Port>;
    async fn execute(&self, inputs: HashMap<String, Value>) -> Result<Value, String>;
    fn metrics_regex(&self) -> Vec<Regex>;
}
```

For V1, this trait is simulated by the TOML manifest + external process pattern.

---

## Manifest Schema

### Required Fields

```toml
[engine]
name = "MyEngine"
category = "Physics"          # Physics | Topology | Systems | Runtime | Quantum | Audio
description = "What it does"

[engine.entry]
type = "cli"                  # "cli" | "python" | "wasm"
command = "python -m myengine.run"
working_dir = "~/repos/myengine"  # optional
```

### Inputs

```toml
[engine.inputs.data]
type = "csv"                  # csv | mesh | tensor | scalar | int | filepath
description = "Input point cloud"
default = "data/default.csv"  # optional
range = [8, 512]              # optional, for numeric types
```

### Outputs

```toml
[engine.outputs.god_tensor]
type = "tensor"
shape = [16]                  # optional, for tensors

description = "The output"
```

### Metrics

```toml
[engine.metrics.convergence]
source = "stdout"             # stdout | stderr | file
pattern = "loss: ([0-9.e-]+)"
```

The Rust backend spawns the engine process and parses stdout line-by-line using these regex patterns. Matching values are streamed to the frontend as live metrics.

---

## Integration Steps

### Step 1: Create Manifest

Create `engines/myengine.toml` following the schema above.

### Step 2: Create Compute Script

**Python example:**

```python
#!/usr/bin/env python3
import json, sys, numpy as np

def main(data_path, **params):
    data = np.loadtxt(data_path, delimiter=",")
    # ... your computation ...
    result = {
        "god_tensor": data.mean(axis=0).tolist(),
        "banach_loss": 1.755e-16,
    }
    print(json.dumps(result))

if __name__ == "__main__":
    main(sys.argv[1])
```

**Rust/compiled example:**

```toml
[engine.entry]
type = "cli"
command = "./target/release/myengine"
```

### Step 3: Auto-Discovery

Restart LAAMBA GOVERNOR. The Engine Rack scans `engines/` on mount. Your engine appears as a new tile with:
- Color from manifest
- Icon from category mapping
- Run button
- Drag-drop dataset target

---

## Frontend Integration

### Custom Visualization

If your engine produces novel output types, add a visualization mode to `TopologyScope.tsx`:

1. Add mode to `MODES` array:
   ```typescript
   { id: "myengine", label: "MyEngine", icon: Zap }
   ```

2. Add render branch:
   ```tsx
   {mode === "myengine" && <MyEngineViewer />}
   ```

3. Read from Zustand store:
   ```typescript
   const { analysisResult } = useStore();
   const myData = analysisResult?.outputs?.god_tensor;
   ```

### Custom Node Type (Pipeline Mixer)

If your engine needs a special node appearance in the DAG:

1. Define node type in `PipelineMixer.tsx`:
   ```typescript
   const nodeTypes = {
     engine: EngineNode,
     myengine: MyEngineNode,
   };
   ```

2. Use in template JSON:
   ```json
   { "id": "my-node", "type": "myengine", "position": { "x": 100, "y": 100 } }
   ```

---

## CLI Bridge Extension

To expose your engine through the Python CLI bridge (`cli/governor_cli.py`):

```python
def cmd_engine(engine_id: str, path: str, params: dict) -> None:
    # Load engine manifest
    manifest = load_manifest(f"engines/{engine_id}.toml")
    # Build command
    cmd = manifest["entry"]["command"]
    # Spawn and capture
    result = subprocess.run([cmd, path], capture_output=True, text=True)
    emit(json.loads(result.stdout))
```

Then add the Tauri command in `src-tauri/src/ipc/commands.rs`:

```rust
#[tauri::command]
pub async fn run_engine(id: String, path: String, params: Value) -> Result<Value, String> {
    run_cli(&["engine", &id, &path]).await
}
```

Register in `lib.rs`:
```rust
.invoke_handler(tauri::generate_handler![
    // ... existing commands ...
    ipc::commands::run_engine,
])
```

---

## Backend Auto-Preference (V2)

When multiple backends implement the same algorithm (e.g., `topo-asm` vs `ripser`), the Governor should auto-prefer the fastest:

```python
class TopologicalGovernor:
    def __init__(self):
        if topo_asm_available():
            self.ph_backend = "topo-asm"   # AVX-512, 43.8× speedup
        else:
            self.ph_backend = "ripser"     # Python fallback
```

This requires:
1. Backend capability registry
2. Benchmark result caching
3. Fallback chain definition in manifest

---

## Security Considerations

- Engine commands run as the user process (no sandboxing in V1).
- Manifests are loaded from the filesystem — validate schema before execution.
- Working directory paths are resolved relative to the manifest file.
- Never execute `engine.entry.command` directly without tokenization (use `sh -c` or `cmd /C` wrappers).

---

## Future: WASM Plugins

V3 will support WASM-based plugins for near-native performance without compiling Rust:

```toml
[engine.entry]
type = "wasm"
module = "engines/myengine.wasm"
export = "compute"
```

This enables:
- Third-party engine distribution
- Sandbox isolation
- Hot-reload without app restart

---

*Plugin API v1.0 — Built by seal.*
