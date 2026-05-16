# Epsilon-Hollow

The largest open-source geometrical scientific Rust operating system.

Three projects. One repository. Mathematically verified.

Epsilon-Hollow is built on 10 formally verified theorems (Lean 4 proofs), a custom
topological DSL (Aether-Lang), and a novel data teleportation primitive that transfers
context between agents in O(1) via topological surgery on S² manifolds.

## The Three Projects

### 1. The OS — Scientific Rust Kernel

A bare-metal x86_64 operating system for scientific computing with capabilities
no traditional OS offers:

- **Data Teleportation** (epsilon-core) — O(1) context transfer via topological surgery
  on hollow S² manifolds with Betti number β₂ = 1. Maps semantic geometry from S^(E−1)
  to a verified topological receptacle, breaking the O(N²) token processing bottleneck.
  Operates on O(P) payload points instead.

- **I/O Speed** (aether-link) — POVM-based adaptive prefetching at ~18 ns per decision.
  Profiles for HFT (conservative), gaming (aggressive), and WSL2 acceleration.

- **Custom DSL** (Aether-Lang) — All data processed through a manifold-native language
  with seal loops (🦭), tilde terminators, and topological convergence primitives.
  Supports both tree-walking interpretation (Bio mode) and bytecode VM (Titan mode).

- **Memory** (aegis-core) — TitanClock lock-free sharded allocator with 32 concurrent
  shards, second-chance clock eviction, and O(1) amortized allocation.

- **Security: Manifold Injection** — Topological attack surface analysis instead of
  traditional JSON/SQL injection. Operates on manifold geometry for offensive and
  defensive security research.

- **Linux-Inspired Capabilities** — Process management, filesystem, and networking
  adapted from the Linux kernel. We study and build on the best.

### 2. Epsilon-OS — Geometrical World Model

An AI assistant powered by a topological world model:

- Episodic memory stored as L2-normalized vectors on S^(D−1)
- O(1) retrieval via spherical Voronoi tessellation (TSS, Theorem 1)
- Forward dynamics via spectral contraction (SCM, Theorem 2)
- LLM synthesis (Minimax 2.7) for reasoning over retrieved context
- Betti-0 clustering for semantic grouping

```
/step <text>     Ingest observation into topological memory
/query <text>    Retrieve from memory + LLM synthesis
/dream <text>    Counterfactual rollout through latent dynamics
/memory          Memory stats: Betti-0, TSS state, cell distribution
/theorems        Run T1-T10 verification suite
```

### 3. Epsilon-IDE — Local AI Coding Assistant

A standalone app (no cloud, no API keys) with three-tier model routing:

| Tier | Model Size | Latency | Use Case |
|------|-----------|---------|----------|
| Fast | 1.5B (~1 GB) | 1–2s | Simple queries |
| Balanced | 7B (~4 GB) | 5–15s | Moderate tasks |
| Deep | 33B (CPU/SSD) | 30–120s | Complex architecture |

Includes file I/O tools, Telegram bot integration, and conversation memory.

## Architecture

```
kernel/
  epsilon/epsilon/              Rust workspace (resolver = "2")
    crates/
      aether-core/              Math foundation (no_std)
        src/
          tss.rs                SphericalVoronoiIndex<K> — O(1) cell lookup on S²
          scm.rs                SpectralContractionOperator<D> — Banach fixed-point
          topology.rs           Betti number computation (filtration-based TDA)
          manifold.rs           Point cloud, sparse graph, topological surgery
          memory.rs             Spatial clustering, SIMD-aligned allocation
          governor.rs           GeometricGovernor — PD adaptive threshold control
          state.rs              LatentState<D> — deviation metrics
          ml/                   Autograd, clustering, neural nets, gossip
      epsilon/                  Context Teleportation (bridge, manifold, teleport)
      epsilon-os/               World model + REPL shell
  aether/
    Aether-Lang/                Custom DSL
      crates/
        aether-lang/            Lexer, parser, AST, interpreter, Titan VM
        aether-kernel/          Bare-metal x86_64 microkernel
        aether-cli/             Cross-platform CLI (run .aether scripts)
        aegis-core/             TitanClock memory allocator
        aegis-cli/              Cross-platform CLI (run .aegis scripts)
    aether-verified/            Lean 4 → Rust verified theorems (T1-T10)
    aether-link/                I/O prefetching kernel (POVM, ~18ns)
    aether-nexus/               System orchestrator
  epsilon-ide/                  Local AI coding assistant (Python)
future/
  apeiron-runtime/              Experimental LLM runtime
```

## Theorems

Ten formally verified theorems govern the system. Each is implemented in Rust,
verified at boot, and backed by Lean 4 proofs.

| ID | Name | What it governs | Status |
|----|------|-----------------|--------|
| T1 | TSS  | O(1) retrieval via spherical Voronoi tessellation | **Active** — wired into `retrieve()` |
| T2 | SCM  | Spectral contraction toward attractor (Banach) | **Active** — wired into `LatentPredictor::step()` |
| T3 | GMC  | Rényi entropy bound on memory consolidation | Verified |
| T4 | AGCR | Governor convergence (eigenvalue-bounded PD control) | Verified |
| T5 | HCS  | Hyperbolic vs. Euclidean separation ratio | Verified |
| T6 | RGCS | Tangent deviation bound for sync frequency | Verified |
| T7 | PHKP | Betti-guided latency (topological persistence) | Verified |
| T8 | TEB  | Landauer energy bound per bit erasure | Verified |
| T9 | CMA  | Alignment error (Procrustes curvature + SVD) | Verified |
| T10 | WPHB | Predictive horizon (information + stability) | Verified |

**Active** = the theorem's output drives runtime decisions.
**Verified** = formula implemented and passes boot-time verification.

## How Memory Works

1. **Store**: Text is encoded to an L2-normalized vector via trigram hashing. The vector
   is assigned to a cluster (by cosine similarity threshold) and pushed into a
   `VecDeque<Episode>`. When episode count exceeds 16, the spherical Voronoi index
   activates: the first 3 dimensions are projected to (θ, φ) on S², and the episode
   is assigned to one of 8 cells.

2. **Retrieve**: With TSS active, the query vector is projected to S², the nearest
   Voronoi cell is located in O(8) = O(1), and only that cell's episodes are scored
   by cosine similarity. Falls back to full scan if the cell has fewer than k results.

3. **Evict**: FIFO via `VecDeque::pop_front()`. Cell bookkeeping is maintained through
   an absolute index scheme (`base_offset`) so Voronoi assignments survive eviction.

4. **Dream**: Counterfactual rollout. The latent predictor applies the spectral
   contraction operator (T2/SCM) to step the state toward the nearest memory attractor,
   producing a reward trace over the horizon.

## How Data Teleportation Works

Context Teleportation is Epsilon-Hollow's mechanism for transferring knowledge between
agents in O(1) instead of retransmitting O(N²) token sequences.

```
Agent A                              Agent B
┌──────────┐    S² surgery     ┌──────────┐
│ Semantic  │──────────────────▶│ Semantic  │
│ geometry  │  O(P) payload    │ geometry  │
│ on S^(E-1)│  points, not     │ on S^(E-1)│
│           │  O(N²) tokens    │           │
└──────────┘                    └──────────┘
```

1. **Encode**: The source agent's context is a point cloud on S^(E−1) (unit hypersphere).
2. **Surgery**: A hollow S² manifold (β₂ = 1) is constructed as a topological receptacle.
   The point cloud is mapped onto this receptacle via stereographic projection.
3. **Transfer**: Only the receptacle (O(P) points) crosses the boundary — not the full
   token sequence.
4. **Decode**: The receiving agent reconstructs the semantic geometry from the receptacle
   via inverse projection.

The key insight: semantic meaning lives in *geometry*, not tokens. Two agents sharing
the same manifold structure share the same understanding.

## How Aether-Lang Works

Aether-Lang is the custom DSL that all data processing in Epsilon-Hollow runs through.
It is manifold-native — every value has a topological interpretation.

**Syntax highlights:**

```aether
~~ Seal loop: iterate with topological convergence check
seal 🦭 points -> manifold {
    project(point, S2)~
    cluster(manifold, betti_threshold: 1)~
}~

~~ Tilde (~) terminates every statement
let x = compute_betti(residuals)~
let converged = x.beta_0 == 1 && x.beta_1 == 0~
```

**Two execution modes:**

| Mode | Engine | Use Case |
|------|--------|----------|
| Bio | Tree-walking interpreter | Debugging, REPL, small scripts |
| Titan | Bytecode VM with stack machine | Production, kernel-level compute |

**Pipeline:** Source → Lexer → Parser → AST → (Bio: walk directly \| Titan: compile to bytecode → VM)

## How TitanClock Memory Works

TitanClock (aegis-core) is a lock-free sharded memory allocator designed for concurrent
scientific workloads.

```
                    TitanClock Allocator
┌─────────┬─────────┬─────────┬─────────┐
│ Shard 0 │ Shard 1 │  ...    │ Shard 31│  32 shards
├─────────┼─────────┼─────────┼─────────┤
│ Clock   │ Clock   │         │ Clock   │  Second-chance
│ Hand    │ Hand    │         │ Hand    │  eviction per shard
└─────────┴─────────┴─────────┴─────────┘
```

- **32 concurrent shards**: Thread ID determines shard → zero contention in common case
- **Second-chance clock eviction**: Each entry has a reference bit. Clock hand sweeps;
  if ref bit is set, clear it and move on. If unset, evict. This approximates LRU
  without the overhead of maintaining a linked list.
- **O(1) amortized allocation**: Hash the key, index the shard, insert or evict.

## How POVM Prefetching Works

Aether-Link's I/O engine uses POVM (Positive Operator-Valued Measure) decision theory
to predict which data blocks to prefetch.

Each I/O pattern is modeled as a measurement outcome. The prefetcher maintains a
probability distribution over likely next accesses and updates it with each observed
access. Decision latency: ~18 ns per prediction.

**Profiles:**

| Profile | Strategy | Target Workload |
|---------|----------|-----------------|
| HFT | Conservative — minimize false prefetches | High-frequency trading |
| Gaming | Aggressive — prefetch broadly, tolerate waste | Game asset streaming |
| WSL2 | Balanced — bridge Linux/Windows I/O semantics | Development environments |

## How Topological Convergence Works

Instead of arbitrary loss thresholds, the ML subsystem (aether-core/ml) detects
convergence through topology:

1. **Betti numbers** (β₀, β₁) of the residual manifold are tracked per epoch.
   β₀ counts connected components; β₁ counts loops.
2. **Centroid drift** of the residual distribution is measured.
3. **Convergence** = Betti numbers stabilize AND drift approaches zero, OR error
   drops below ε.

The benchmark engine auto-escalates model complexity (Linear → Polynomial → RBF)
when the current model stalls, continuing until topological convergence is achieved.

## Build

Requires Rust 1.85+.

```bash
cargo build --workspace
cargo test --workspace        # 231 tests
cargo clippy --workspace --all-targets -- -D warnings
```

Run the world model shell:

```bash
cargo run -p epsilon-os
```

Run the Aether-Lang REPL:

```bash
cargo run -p aether-cli
```

Build the bare-metal kernel (requires nightly):

```bash
cargo build -p aether-kernel --target x86_64-unknown-none
```

## Tests

```
aether-core    70 unit + 7 proptest
epsilon        37 unit
epsilon-os     25 unit (6 TSS-specific)
aether-lang    tests via workspace
aegis-core     tests via workspace
aether-link    tests via workspace
aether-verified tests via workspace
               ─────────
               231 total, 0 failures
```

## Lean Proofs

```bash
cd kernel/aether/aether-verified/lean
lake build
```

Proves: Betti number properties, Chebyshev concentration bounds, governor convergence,
pruning correctness.

## CI

GitHub Actions (`.github/workflows/ci.yml`): fmt, clippy, test, bench-compile,
miri, cargo-audit, cargo-deny, docs, BOM validation, Python checks, Lean proofs.

## License

MIT License. Copyright (c) 2024 Teerth Sharma. See [LICENSE](LICENSE).
