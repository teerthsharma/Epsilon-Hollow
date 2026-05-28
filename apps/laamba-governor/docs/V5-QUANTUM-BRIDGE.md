# V5 — Quantum Bridge

> Connecting topological data analysis with quantum computing.

## TopoBridge-Q Architecture

```
Classical TDA Pipeline          Quantum Acceleration
─────────────────────          ────────────────────
Point Cloud                    Quantum State Prep
    ↓                              ↓
Distance Matrix    ──────→    Quantum Distance (SWAP test)
    ↓                              ↓
Boundary Operator  ──────→    Quantum Rank (phase estimation)
    ↓                              ↓
Betti Numbers     ←──────     Quantum Betti (exponential speedup)
    ↓                              ↓
Persistence        ──────→    Quantum Persistence
```

## Key Algorithms

### 1. Quantum Persistent Homology (Lloyd et al. 2016)

```
Input: Boundary matrices ∂_k as sparse matrices
Goal: Compute rank(∂_k) to get Betti numbers

Classical: O(n^3) via Smith normal form
Quantum: O(n * polylog(n)) via quantum phase estimation

Steps:
1. Encode ∂_k as block of unitary U = exp(i ∂_k t)
2. Apply quantum phase estimation to U
3. Count near-zero eigenvalues → nullity
4. β_k = nullity(∂_k) - rank(∂_{k+1})
```

### 2. Quantum SWAP Test (Distance)

```
Circuit for fidelity between |ψ⟩ and |φ⟩:
|0⟩ ─── H ─── CSWAP ─── H ─── Measure
|ψ⟩ ─────────── │ ──────────
|φ⟩ ─────────── │ ──────────

P(0) = (1 + |⟨ψ|φ⟩|²) / 2
Distance: d(ψ,φ) = √(1 - |⟨ψ|φ⟩|²)
```

### 3. Topological Quantum Error Correction

```
Surface code on torus (genus 1):
- Physical qubits on edges of lattice
- X-stabilizers on vertices: A_v = Π_{e∈v} X_e  
- Z-stabilizers on faces: B_f = Π_{e∈f} Z_e
- Logical operators = non-trivial cycles (homology generators)
- Code distance = shortest non-trivial cycle length

Encodes 2g logical qubits on genus-g surface
```

## Implementation Plan

### Phase 1: Qiskit Simulation
```python
# pip install qiskit qiskit-aer
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

def quantum_betti(boundary_matrix, dim):
    """Estimate Betti number using quantum phase estimation."""
    n_qubits = int(np.ceil(np.log2(boundary_matrix.shape[0])))
    qc = QuantumCircuit(n_qubits + 1, 1)
    # ... build circuit ...
    simulator = AerSimulator()
    result = simulator.run(transpile(qc, simulator)).result()
    return estimate_rank_from_counts(result.get_counts())
```

### Phase 2: IBM Quantum Hardware
- Use `qiskit-ibm-runtime` for real quantum hardware
- Need IBM Quantum account (free tier: 10 min/month)
- Calibrate for noise, use error mitigation

### Phase 3: Hybrid Classical-Quantum
- Use quantum for expensive subroutines (rank computation)
- Classical for everything else
- Automatic fallback to classical simulator when hardware unavailable

## CLI Interface
```bash
python cli/governor_cli.py quantum <path> --backend=aer_simulator --dim=2
python cli/governor_cli.py quantum <path> --backend=ibm_brisbane --shots=1000
```

## Output Format
```json
{
  "command": "quantum",
  "backend": "aer_simulator",
  "betti_numbers": [1, 2, 0],
  "circuit_depth": 45,
  "n_qubits": 8,
  "shots": 1024,
  "fidelity": 0.97,
  "elapsed_ms": 1200
}
```
