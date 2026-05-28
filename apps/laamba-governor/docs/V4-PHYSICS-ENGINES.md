# V4 — Physics Engine Integration

> Each engine has a TOML manifest in `engines/`. The Rust backend loads them via `src-tauri/src/engine/registry.rs`.
> Currently engines are display-only. This doc specifies how to make each one compute real results.

## Architecture Pattern

Every engine follows the same pattern:
1. TOML manifest defines name, entry point, inputs, outputs
2. Python/Rust compute backend does the work
3. CLI bridge exposes it: `python cli/governor_cli.py engine <engine_id> <path> [params]`
4. Tauri command calls CLI, returns JSON to frontend
5. Frontend panel visualizes the result

---

## 1. Faraday — God Tensor (EM Field on Manifolds)

**TOML**: `engines/faraday.toml`

**What**: Electromagnetic field computation via the Faraday tensor on curved spacetime. Already exists as a standalone project by seal.

**Math**:
```
Faraday tensor: F_μν = ∂_μ A_ν - ∂_ν A_μ

Maxwell's equations in curved spacetime:
∇_μ F^{μν} = J^ν  (with source)
∇_{[μ} F_{νρ]} = 0  (Bianchi identity)

Stress-energy tensor:
T^{μν} = F^{μα} F^ν_α - (1/4) g^{μν} F_{αβ} F^{αβ}

Energy density:
u = (1/2)(E² + B²)

Poynting vector:
S = E × B
```

**Implementation**:
- Entry point: the existing `faraday/` project files (if present in seal's repos)
- CLI: `python cli/governor_cli.py engine faraday <mesh_file> --resolution=64 --epochs=50000`
- Output: field values on grid, energy density, convergence curve
- Visualization: 3D vector field (E,B arrows) + energy density heatmap

---

## 2. Hamilton — N-Body on Symplectic Manifolds

**TOML**: `engines/hamilton.toml`

**What**: Hamiltonian N-body simulation with gauge symmetry, preserving symplectic structure.

**Math**:
```
Hamilton's equations:
dq/dt = ∂H/∂p
dp/dt = -∂H/∂q

Symplectic form: ω = Σ dp_i ∧ dq_i

Liouville's theorem: d/dt(det J) = 0
  (phase space volume preserved)

Symplectic integrator (Stormer-Verlet):
  p_{n+1/2} = p_n - (Δt/2) ∇V(q_n)
  q_{n+1} = q_n + Δt p_{n+1/2} / m
  p_{n+1} = p_{n+1/2} - (Δt/2) ∇V(q_{n+1})

N-body Hamiltonian:
H = Σ |p_i|²/2m_i + Σ_{i<j} V(|q_i - q_j|)

Gauge coupling (SU(2)):
H = Σ |p_i - g A(q_i)|²/2m + V_gauge
```

**Implementation**:
- Symplectic integrator in Python (NumPy)
- CLI: `python cli/governor_cli.py engine hamilton --n_bodies=4 --dim=3 --gauge=SU2 --steps=10000`
- Output: trajectories, energy conservation error, phase space volume
- Visualization: 3D trajectory lines, energy plot, phase portrait

---

## 3. OmniTopos — Cosmological Phase Sequences

**TOML**: `engines/omnitopos.toml`

**What**: Compute topological phase transitions in cosmological models using persistent homology.

**Math**:
```
Phase transition detection:
1. Sample cosmological field at time steps t_0, ..., t_N
2. Compute persistence diagram D(t_i) at each step
3. Track birth/death of features across time
4. Phase transition = discontinuity in Betti numbers

Topological entropy:
S_top(ε) = Σ_k (-1)^k log β_k(ε)

Persistent entropy:
E(D) = -Σ_i (l_i / L) log(l_i / L)
  where l_i = death_i - birth_i, L = Σ l_i
```

**Implementation**:
- Generate cosmological field snapshots (FRW metric perturbations)
- Compute persistence at each snapshot
- Track topological phase transitions
- Visualization: animated persistence diagram over time

---

## 4. TopoBridge-Q — Quantum Topology

**TOML**: `engines/topobridge-q.toml`

**What**: Bridge between topological data analysis and quantum computing. IBM Quantum integration.

**Math**:
```
Quantum state on Bloch sphere:
|ψ⟩ = cos(θ/2)|0⟩ + e^{iφ} sin(θ/2)|1⟩

Topological quantum codes:
Surface code on genus-g surface encodes 2g logical qubits

Quantum persistent homology:
- Encode boundary operator as quantum circuit
- Use quantum phase estimation for Betti numbers
- Exponential speedup for rank computation
```

**Implementation**:
- Requires: `pip install qiskit qiskit-aer`
- CLI: `python cli/governor_cli.py engine topobridge-q <path> --backend=aer_simulator`
- Can work in simulation mode without IBM Quantum access

---

## 5. NeuralWhisper — Topology-Guided TTS

**TOML**: `engines/neuralwhisper.toml`

**What**: Text-to-speech where the acoustic manifold is chosen by topological analysis of the voice embedding space.

**Concept**: Voice embeddings live on a manifold. The Governor predicts which manifold geometry to use for the voice space, then NeuralWhisper generates speech using that geometry.

**Implementation**: Separate large project — doc only for now.

---

## Engine Integration Template

For any new engine, follow this template:

### 1. Create TOML manifest
```toml
[engine]
name = "MyEngine"
version = "0.1.0"
category = "Topology"
description = "What it does"

[engine.entry]
type = "python"
script = "engines/myengine/run.py"

[engine.inputs.data]
type = "csv"
description = "Input point cloud"

[engine.outputs.result]
type = "json"
description = "Computation result"
```

### 2. Create Python compute script
```python
# engines/myengine/run.py
import json, sys, numpy as np

def main(data_path, **params):
    data = np.loadtxt(data_path, delimiter=",")
    # ... compute ...
    result = {"key": "value"}
    print(json.dumps(result))

if __name__ == "__main__":
    main(sys.argv[1])
```

### 3. Add CLI bridge
In `cli/governor_cli.py`, add:
```python
def cmd_engine(engine_id, path, params):
    # Load engine manifest, run entry script, return JSON
```

### 4. Add Tauri command
In `src-tauri/src/ipc/commands.rs`:
```rust
#[tauri::command]
pub async fn run_engine(id: String, path: String, params: Value) -> Result<Value, String> {
    run_cli(&["engine", &id, &path]).await
}
```

### 5. Create frontend visualization
Custom panel in `src/panels/` or mode in existing panel.
