# Epsilon-Hollow

Topological operating system kernel with mathematically verified memory management.

Epsilon-Hollow models agent memory as a Riemannian manifold on S^(D-1). Episodes are
stored on the manifold, clustered via Betti-0 connected components, and retrieved through
O(1) amortized spherical Voronoi tessellation. A spectral contraction operator drives
forward dynamics. Ten theorems govern the system's behavior; each is implemented in Rust,
verified at boot, and backed by Lean 4 proofs.

## Build

Requires Rust 1.85+.

```bash
cd kernel/epsilon/epsilon
cargo build --workspace
cargo test --workspace        # 139 tests
cargo clippy --workspace --all-targets -- -D warnings
```

Run the OS shell:

```bash
cargo run -p epsilon-os
```

With LLM reasoning (Minimax 2.7):

```bash
cargo run -p epsilon-os -- --api-key YOUR_KEY
# or
MINIMAX_API_KEY=key cargo run -p epsilon-os
```

## Shell Commands

```
/step <text>     Ingest observation into topological memory
/query <text>    Retrieve from memory + LLM synthesis
/dream <text>    Counterfactual rollout through latent dynamics
/memory          Memory stats: size, Betti-0, TSS state, cell distribution
/theorems        Run T1-T10 verification suite
/status          Dim, capacity, API key status
/history         Ingestion log
/reset           Clear memory
```

Plain text at the prompt is treated as `/query`.

## Architecture

```
kernel/
  epsilon/epsilon/             Rust workspace (resolver = "2")
    crates/
      aether-core/             Math foundation
        src/
          tss.rs               SphericalVoronoiIndex<K> — O(1) cell lookup on S^2
          scm.rs               SpectralContractionOperator<D> — Banach fixed-point dynamics
          governor.rs          GeometricGovernor — PD adaptive threshold control
          topology.rs          Betti number computation (filtration-based)
          manifold.rs          Point cloud, sparse graph, topological surgery
          memory.rs            Spatial clustering, SIMD-aligned allocation
          state.rs             LatentState<D> — deviation metrics
          ml/                  Autograd, clustering, neural nets, gossip
          os/                  Page tables, CPU context, hardware topology
      epsilon/                 Context Teleportation research (bridge, manifold, teleport)
      epsilon-os/              Operating system shell
        src/
          world.rs             ManifoldMemory, LatentPredictor, World, theorem verification
          main.rs              CLI REPL
  aether/
    aether-verified/           Theorem implementations (T1-T10 formulas)
      src/                     Rust: TSS, SCM, GMC, AGCR, HCS, RGCS, PHKP, TEB, CMA, WPHB
      lean/                    Lean 4 proofs: Betti, Chebyshev, Governor, Pruning
    aether-link/               I/O prefetching kernel (POVM-based, DirectStorage)
    Aether-Lang/               Language compiler (lexer, parser, AST, ML runtime)
  runtime/
    apeiron-runtime-master/    Runtime workspace (DSP, kernel, CLI)
```

## Theorems

| ID | Name | What it governs | Status |
|----|------|-----------------|--------|
| T1 | TSS  | O(1) retrieval via spherical Voronoi tessellation | **Active** — wired into `retrieve()` |
| T2 | SCM  | Spectral contraction toward attractor (Banach) | **Active** — wired into `LatentPredictor::step()` |
| T3 | GMC  | Renyi entropy bound on memory consolidation | Verified |
| T4 | AGCR | Governor convergence (eigenvalue-bounded PD control) | Verified |
| T5 | HCS  | Hyperbolic vs. Euclidean separation ratio | Verified |
| T6 | RGCS | Tangent deviation bound for sync frequency | Verified |
| T7 | PHKP | Betti-guided latency (topological persistence) | Verified |
| T8 | TEB  | Landauer energy bound per bit erasure | Verified |
| T9 | CMA  | Alignment error (Procrustes curvature + SVD) | Verified |
| T10 | WPHB | Predictive horizon (information + stability) | Verified |

**Active** = the theorem's output drives runtime decisions.
**Verified** = formula implemented and passes boot-time verification, not yet wired into control flow.

All 10 pass `cargo test` and boot-time verification.

## How Memory Works

1. **Store**: Text is encoded to an L2-normalized vector via trigram hashing. The vector is
   assigned to a cluster (by cosine similarity threshold) and pushed into a `VecDeque<Episode>`.
   When episode count exceeds 16, the spherical Voronoi index activates: the first 3 dimensions
   are projected to (theta, phi) on S^2, and the episode is assigned to one of 8 cells.

2. **Retrieve**: With TSS active, the query vector is projected to S^2, the nearest Voronoi
   cell is located in O(8) = O(1), and only that cell's episodes are scored by cosine
   similarity. Falls back to full scan if the cell has fewer than k results.

3. **Evict**: FIFO via `VecDeque::pop_front()`. Cell bookkeeping is maintained through an
   absolute index scheme (`base_offset`) so Voronoi assignments survive eviction.

4. **Dream**: Counterfactual rollout. The latent predictor applies the spectral contraction
   operator (T2/SCM) to step the state toward the nearest memory attractor, producing a
   reward trace over the horizon.

## Tests

```
aether-core    70 unit + 7 proptest
epsilon        37 unit
epsilon-os     25 unit (6 TSS-specific)
               ─────────
               139 total, 0 failures
```

## Lean Proofs

```bash
cd kernel/aether/aether-verified/lean
lake build
```

Proves: Betti number properties, Chebyshev concentration bounds, governor convergence,
pruning correctness.

## Python Verification

```bash
python tests/verify_theorems.py
```

Cross-checks Lean/Rust theorem implementations against a Python reference.

## CI

GitHub Actions (`.github/workflows/ci.yml`): fmt, clippy, test, bench-compile.
Runs on every push and PR against `main`.

## License

Custom Attribution License. Copyright (c) 2024 Teerth Sharma.
See [LICENSE](LICENSE).
