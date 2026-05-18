# Theorem Reference -- T1 through T10

> Every claim in this document is traceable to a specific source file and line.

---

## T1 -- Topological State Synchronization (TSS)

**Formal Name:** Topological State Synchronization  
**Abbreviation:** T1/TSS

**Mathematical Statement:**  
Given K fixed Voronoi centroids on S^2, the nearest-centroid lookup `locate(q)` runs in O(K) time. Since K is a compile-time constant, this is O(1) amortized. Each Voronoi cell is a single connected component, so the index has Betti-0 number beta_0 = K.

**Implementation:**  
- `kernel/epsilon/epsilon/crates/aether-core/src/tss.rs` -- struct `SphericalVoronoiIndex<const K: usize>` (line 56)  
- `locate(&self, query: (f64, f64)) -> usize` (line 93): linear scan over K centroids using `great_circle_distance` (line 22), returns index of nearest  
- `betti_0(&self) -> u32` (line 134): returns `K as u32`  
- `new_verified(centroids, theta_min) -> Option<Self>` (line 81): validates all centroid pairs satisfy separation >= theta_min via `verify_separation` (line 29)

**Kernel Subsystems Using T1:**  
- **ManifoldFS:** `kernel/seal-os/src/fs/manifold_fs.rs` -- `assign_voronoi_cell()` (line 489) calls `self.voronoi.locate((theta, phi))` (line 504) to place files into cells. `store()` (line 140) calls it on every file creation. `find()` (line 257) calls it to narrow search to a single Voronoi cell.  
- **Scheduler:** `kernel/seal-os/src/process/scheduler.rs` -- `select_next_task()` (line 106) calls `self.voronoi.locate(...)` (line 108) to predict which Voronoi cell the next task belongs to.

**Lean 4 Proof:** No dedicated Lean file for T1. The separation bound is checked at runtime via `new_verified` in `tss.rs`.

**Status:** Active in kernel

---

## T2 -- Spectral Contraction Mapping (SCM)

**Formal Name:** Spectral Contraction Mapping  
**Abbreviation:** T2/SCM

**Mathematical Statement:**  
The operator T(S) = (1 - alpha) * S + alpha * S_pred is a contraction mapping with Lipschitz constant (1 - alpha) < 1 for alpha in (0, 1]. By Banach's fixed-point theorem, iterating T from any starting state converges geometrically to S_pred:  
||S_t - S_pred|| <= (1 - alpha)^t * ||S_0 - S_pred||

**Implementation:**  
- `kernel/epsilon/epsilon/crates/aether-core/src/scm.rs` -- struct `SpectralContractionOperator<const D: usize>` (line 29)  
- `apply(&self, state: &[f64; D], pred: &[f64; D]) -> [f64; D]` (line 46): componentwise `(1 - alpha) * state[i] + alpha * pred[i]`  
- `lipschitz_constant(&self) -> f64` (line 62): returns `1.0 - self.alpha`  
- `iterate(&self, state, pred, steps) -> [f64; D]` (line 71): applies `apply` `steps` times  
- `LatentPredictor<const D: usize, const A: usize>` (line 88): implements `s_{t+1} = SCM(s_t + W*a_t)` via `step()` (line 128)

**Kernel Subsystems Using T2:**  
- **ManifoldFS:** `manifold_fs.rs` line 63 creates `scm: SpectralContractionOperator::new(0.7)` (alpha=0.7). `update_prefetch_state()` (line 369) calls `self.scm.apply(&self.access_state, &pt)` (line 394) to predict next file access.  
- **Scheduler:** `scheduler.rs` line 20 creates `predictor: SpectralContractionOperator::new(0.7)`. `schedule()` (line 74) calls `self.predictor.apply(&self.predict_state, &self.tasks[idx].manifold_embedding)` (line 92) to predict next task.

**Lean 4 Proof:** No dedicated Lean file for T2/SCM specifically. The contraction property is algebraically verifiable from the definition.

**Status:** Active in kernel

---

## T3 -- Geometric Manifold Clustering (GMC)

**Formal Name:** Geometric Manifold Clustering  
**Abbreviation:** T3/GMC

**Mathematical Statement:**  
When cluster entropy exceeds a threshold, the two smallest clusters are merged to maintain balanced Voronoi partitioning. Shannon entropy H = -sum(p_i * log2(p_i)) over cluster size proportions is used as the trigger.

**Implementation:**  
- `kernel/seal-os/src/fs/manifold_fs.rs` -- `ENTROPY_MERGE_THRESHOLD: f64 = 2.0` (line 21)  
- `update_entropy(&mut self)` (line 403): computes Shannon entropy over `self.cluster_sizes`  
- `check_entropy_and_merge(&mut self)` (line 419): if `current_entropy > ENTROPY_MERGE_THRESHOLD`, sorts clusters by size (line 426), moves all files from smallest cell to second-smallest (lines 438-449), clears source cell, increments `entropy_merges` counter  
- Logged as `"T3/GMC"` event (line 455)

**Kernel Subsystems Using T3:**  
- **ManifoldFS:** Called after every `teleport()` operation (line 242). Ensures Voronoi cells remain balanced as files are moved.

**Lean 4 Proof:** None.

**Status:** Active in kernel

---

## T4 -- Adaptive Geometric Convergence Rate (AGCR)

**Formal Name:** Adaptive Geometric Convergence Rate / Geometric Governor  
**Abbreviation:** T4/AGCR

**Mathematical Statement:**  
PD controller on the state manifold:  
  epsilon(t+1) = epsilon(t) + alpha * e(t) + beta * de/dt  
where e(t) = R_target - Delta(t)/epsilon(t).  
Lyapunov stability: V(e) = e^2, and under contraction e' = rho * e with |rho| <= 1, V(rho*e) <= V(e).

**Implementation:**  
- `kernel/epsilon/epsilon/crates/aether-core/src/governor.rs` -- struct `GeometricGovernor` (line 75)  
- Constants: `TARGET_TICK_RATE = 1000.0` (line 31), `ALPHA = 0.01` (line 35), `BETA = 0.05` (line 40), `EPSILON_MIN = 0.001` (line 43), `EPSILON_MAX = 10.0` (line 46), `EPSILON_INITIAL = 0.1` (line 49)  
- `adapt(&mut self, deviation_delta: f64, dt: f64) -> f64` (line 148): computes `current_rate = deviation_delta / self.epsilon` (line 166), `error = TARGET_TICK_RATE - current_rate` (line 177), `d_error = (error - self.last_error) / dt` (line 187), applies `adjustment = alpha * error + beta * d_error` (line 198), updates `self.epsilon -= adjustment` (line 204), clamps to [EPSILON_MIN, EPSILON_MAX] (line 219)  
- `should_trigger(&self, deviation: f64) -> bool` (line 227): returns `deviation >= self.epsilon`

**Kernel Subsystems Using T4:**  
- **ManifoldFS:** `manifold_fs.rs` line 100 creates `governor: GeometricGovernor::new()`. `governor_tick()` (line 463) calls `self.governor.adapt(deviation, 0.01)`. Called from `store()` (line 177), `teleport()` (lines 209, 221).  
- **Scheduler:** `scheduler.rs` line 19 creates `governor: GeometricGovernor::new()`. `schedule()` (line 97) calls `self.governor.adapt(deviation, 0.01)`. `adaptive_timeslice()` (line 146) uses `self.governor.epsilon()` to scale timeslice: if eps < 0.5, scale = 2.0; else scale = 0.5.

**Lean 4 Proof:**  
- `kernel/aether/aether-verified/lean/AetherVerified/Governor.lean`  
- `lyapunov_descent` (line 29): proves V(rho*e) <= V(e) when |rho| <= 1  
- `geometric_bound` (line 48): proves |rho|^t * |e_0| <= |e_0|  
- `gain_margin_yields_contraction` (line 60): proves that `gainMarginRefined(dt)` implies 0 < 1 - (0.01 + 0.05/dt) < 1

**Status:** Active in kernel + Verified in Lean 4

---

## T5 -- Hyperbolic Capacity Scaling (HCS)

**Formal Name:** Hyperbolic Capacity Scaling  
**Abbreviation:** T5/HCS

**Mathematical Statement:**  
Directory tree capacity scales hyperbolically with depth. For a branching factor b and depth d >= 2, the hyperbolic ratio is:  
  ratio = ln(b) * d * (d - 1) / 2  
This models the exponential growth of addressable space in hyperbolic geometry.

**Implementation:**  
- `kernel/seal-os/src/fs/manifold_fs.rs` -- `update_hyperbolic_ratio(&mut self, depth: usize)` (line 481): computes `log(4.0) * d * (d - 1) / 2` for branching factor b=4  
- `mkdir()` (line 305): calls `compute_depth(parent_id) + 1` (line 317), updates `max_depth` and `hyperbolic_ratio` when a new maximum depth is reached (lines 318-321)  
- Logged as `"T5/HCS"` event (line 347)  
- `ManifoldFS` tracks `max_depth` (line 73) and `hyperbolic_ratio` (line 74, initialized to `f64::INFINITY` at line 103)

**Kernel Subsystems Using T5:**  
- **ManifoldFS:** Applied during `mkdir()` to track directory tree growth characteristics.

**Lean 4 Proof:** None.

**Status:** Active in kernel

---

## T6 -- Betti Error Bound

**Formal Name:** Heuristic Betti-1 Counting Bound  
**Abbreviation:** T6/Betti

**Mathematical Statement:**  
For a window-of-4 loop detector scanning a byte sequence of length n with tolerance tol:  
  betti1_heuristic(data, tol) <= beta_1_exact + n  
The heuristic count is bounded by the number of candidate window-start indices.

**Implementation:**  
- `kernel/epsilon/epsilon/crates/aether-core/src/topology.rs` -- `compute_betti_0(data: &[u8]) -> u32` (line 121) at `CLUSTER_THRESHOLD = 15` (line 33)  
- `compute_betti_1(data: &[u8]) -> u32` (line 148): window-of-4 loop detection with tolerance 5  
- `compute_shape(data: &[u8]) -> TopologicalShape` (line 177): computes (beta_0, beta_1, density)  
- `verify_shape(data: &[u8]) -> VerifyResult` (line 234): checks `DENSITY_MIN = 0.1` (line 37), `DENSITY_MAX = 0.6` (line 40), `MAX_BETTI_1 = 10` (line 44)  
- `verify_sliding_window(data, window_size)` (line 295): fail-fast sliding window, default `WINDOW_SIZE = 64` (line 35)

**Lean 4 Proof:**  
- `kernel/aether/aether-verified/lean/AetherVerified/Betti.lean`  
- `heuristic_le_length` (line 49): proves `betti1Heuristic data tol <= data.length`  
- `betti_error_bound` (line 61): proves `betti1Heuristic data tol <= beta_1 + data.length`

**Status:** Verified in Lean 4 (not directly called in kernel boot path; available for topology gatekeeper)

---

## T7 -- Chebyshev Counting Bound

**Formal Name:** Finite Markov/Chebyshev Counting Bound  
**Abbreviation:** T7/Chebyshev

**Mathematical Statement:**  
For non-negative reals y_i and threshold t > 0:  
  |{i : y_i >= t}| * t <= sum(y_i)  
Specializing to y_i = (x_i - mu)^2 and t = (k*sigma)^2 yields the classical one-sided Chebyshev:  
  |{i : x_i <= mu - k*sigma}| <= n / k^2

**Implementation:**  
- No direct Rust implementation in the kernel. The Chebyshev guard is referenced by the aether-verified proof infrastructure.

**Lean 4 Proof:**  
- `kernel/aether/aether-verified/lean/AetherVerified/Chebyshev.lean`  
- `markov_count_bound` (line 36): proves |S| * t <= sum(y_i) for S = {i : t <= y_i}  
- `chebyshev_one_sided_sq` (line 67): proves the one-sided squared form used by the Rust GC guard

**Status:** Verified only (Lean 4 proof; no kernel callsite)

---

## T8 -- Cauchy-Schwarz Pruning Bound

**Formal Name:** Cauchy-Schwarz Pruning Soundness  
**Abbreviation:** T8/Pruning

**Mathematical Statement:**  
For vectors q, k: (dot(q, k))^2 <= normSq(q) * normSq(k).  
This backs the upper bound score: <q, k> <= ||q|| * (||mu_B|| + r_B) used in attention-head pruning.

**Implementation:**  
- Referenced by `aether_pruning::upper_bound_score` in the aether-verified Rust layer.

**Lean 4 Proof:**  
- `kernel/aether/aether-verified/lean/AetherVerified/Pruning.lean`  
- `cauchy_schwarz_sq` (line 43): full inductive proof of (dotL q k)^2 <= normSqL q * normSqL k on List R  
- `upper_bound_sound` (line 159): skeleton statement (currently `trivial`; full sqrt form deferred)  
- Helper functions: `dotL` (line 24), `normSqL` (line 30), `normSqL_nonneg` (line 34)

**Status:** Verified only (Lean 4 proof of core inequality; `upper_bound_sound` is a skeleton)

---

## T9 -- Reserved / Planned

**Status:** Planned

---

## T10 -- Reserved / Planned

**Status:** Planned

---

## Theorem Status Summary Table

| ID | Name | Abbreviation | Kernel Active | Lean Verified | Source |
|----|------|-------------|---------------|---------------|--------|
| T1 | Topological State Synchronization | TSS | Yes | No | `aether-core/src/tss.rs` |
| T2 | Spectral Contraction Mapping | SCM | Yes | No | `aether-core/src/scm.rs` |
| T3 | Geometric Manifold Clustering | GMC | Yes | No | `seal-os/src/fs/manifold_fs.rs` |
| T4 | Adaptive Geometric Convergence Rate | AGCR | Yes | Yes | `aether-core/src/governor.rs`, `Governor.lean` |
| T5 | Hyperbolic Capacity Scaling | HCS | Yes | No | `seal-os/src/fs/manifold_fs.rs` |
| T6 | Betti Error Bound | Betti | No | Yes | `aether-core/src/topology.rs`, `Betti.lean` |
| T7 | Chebyshev Counting Bound | Chebyshev | No | Yes | `Chebyshev.lean` |
| T8 | Cauchy-Schwarz Pruning Bound | Pruning | No | Partial | `Pruning.lean` |
| T9 | Reserved | -- | -- | -- | -- |
| T10 | Reserved | -- | -- | -- | -- |

The `theorem_status()` method in `manifold_fs.rs` (line 538) reports T1-T5 as `"ACTIVE"` at runtime.
