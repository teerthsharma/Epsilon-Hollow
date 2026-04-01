# Epsilon Research Specification

**Geometric State Transfer via Topological Surgery on Hollow S² Manifolds**

*v0.1.0-Draft | Author: Teerth Sharma*

---

## 1. Problem Statement

The KV-cache attention mechanism in transformer architectures has complexity O(N²) in sequence length N. This is the primary latency bottleneck in long-context inference. While retrieval-augmented generation (RAG) limits the active context window, it does not eliminate the quadratic scaling within that window. Once context is cached, it cannot be reused across agent boundaries without full re-ingestion.

This work addresses the **cross-agent context reuse** problem. Given agent A with a converged internal state representing a processed N-token context, under what conditions and by what mechanism can agent B acquire an equivalent internal state at cost independent of N?

---

## 2. Approach

We model each agent's internal state as a **Riemannian manifold embedded in ℝ^D** whose topology is characterized by its Betti numbers (β₀, β₁, β₂). We enforce a specific topological constraint — **β₂ = 1** — which creates an interior void in the manifold's geometry.

A state transfer is performed by computing a topologically verified point cloud from the source agent's semantic embeddings, and injecting it into the void of the receiving agent's manifold via a guarded system call. The receiving agent's kernel verifies topological compatibility before assimilating the injected geometry into its active processing shell.

The transfer cost is **O(P)** in the number of points in the payload (P ≤ 64 by design), independent of the source context length N.

---

## 3. The Embedding-to-Manifold Bridge

### 3.1 Spherical Structure of LLM Embeddings

Let e ∈ ℝ^E be a token embedding from a transformer language model. These models are trained using cosine similarity objectives:

```
sim(u, v) = (u · v) / (‖u‖ · ‖v‖)
```

Cosine similarity is invariant to scale — it depends only on the orientation of the embedding vector. Therefore the semantically meaningful geometry of LLM embeddings lives on S^(E-1), the unit (E-1)-sphere in ℝ^E. L2 normalization makes this structure explicit.

### 3.2 Johnson-Lindenstrauss Projection

**Lemma:** For any ε ∈ (0, 1/2) and n points in ℝ^E, a matrix M ∈ ℝ^(D×E) with i.i.d. entries from N(0, 1/D) satisfies, with probability ≥ 1 - δ:

```
∀ u, v ∈ P:  (1-ε)‖u-v‖² ≤ ‖Mu-Mv‖² ≤ (1+ε)‖u-v‖²
```

provided D ≥ (4 ln(n/δ)) / (ε²/2 - ε³/3).

The projection is **seeded and deterministic**: fixing the PRNG seed fixes M. Two agents with the same seed project identical semantic inputs to identical geometric locations — a prerequisite for the cross-agent compatibility check.

### 3.3 Spherical Normalization Guarantees the Hollow Constraint

After projection to ℝ^D, each point is normalized:

```
p̂ᵢ = (M·eᵢ) / ‖M·eᵢ‖₂
```

The set {p̂ᵢ} is a point cloud on the unit 2-sphere S² ⊂ ℝ³. By the Universal Coefficient Theorem:

```
H_k(S²; ℤ) ≅ ℤ   for k ∈ {0, 2}
              0    otherwise
```

Betti numbers: **β₀ = 1** (connected), **β₁ = 0** (no tunnels), **β₂ = 1** (one enclosed void). A Vietoris-Rips filtration of a sufficiently dense sample of S² will exhibit a persistent β₂ feature. The hollow manifold constraint is not an assumption — it is a topological invariant of the sphere.

### 3.4 Minimum Sampling Density

By Niyogi-Smale-Weinberger: if P is an ε-sample of a manifold M with reach τ, then the Vietoris-Rips complex VR(P, r) recovers Hom(M) for appropriate r when ε < τ/2. For S² with τ = 1, the required density condition ε < 0.5 corresponds to a minimum of **20 projected tokens** (MIN_TOKENS = 20).

---

## 4. The Hollow Manifold Receiver

### 4.1 Definition

A `HollowCubeManifold<D>` is a pair (Shell, Void) where:
- **Shell**: a `SparseGraph<D>` representing the outer S² boundary (β₀ = 1)
- **Void**: an `Option<ManifoldPayload<D>>` — the injection receptacle

The Betti signature of the full structure is (β₀=1, β₁=0, β₂=1). The β₂=1 feature encodes the presence of the void.

### 4.2 The ManifoldPayload

A `ManifoldPayload<D>` carries:
- A fixed point cloud (up to 64 points on S²)
- The Betti signature (β₀, β₁) at construction time
- An inherited `liveness_anchor: f64` for GC protection

Payload validity requires β₀ = 1. Disconnected payloads are rejected at injection time.

### 4.3 Injection and Assimilation

`inject_into_void(payload)` performs a pre-flight check:
1. Void is unoccupied
2. Shell is non-degenerate (β₀ = 1)
3. Payload is non-empty and connected (β₀ = 1)

On success, the payload is written to the void and a wake-up interrupt is queued.

`assimilate()` completes the transfer:
1. Removes the payload from the void
2. Merges payload points into the shell's active SparseGraph
3. Marks the void as empty

The void is restored after assimilation, ready for the next injection.

---

## 5. Safety Mechanisms

### 5.1 Surgery Permit (Governor Clutch)

The `SurgeryGovernor` implements a PD-like adaptation law:

```
ε(t+1) = ε(t) + α·e(t) + β·Δe(t)
```

where e(t) is the current deviation from threshold. Instantaneous state injection causes |Δe(t)| → ∞ for one tick, which drives the β-weighted derivative term into instability.

The `SurgeryPermit` is a **one-shot RAII token** (non-Clone, non-Copy) acquired by `prepare_for_surgery()`. It snapshots the current (β, last_error) state and zeroes both fields. `complete_surgery(permit)` consumes the token and restores the original values.

The permit pattern enforces that every prepare has exactly one complete — preventing protocol violation at the type level.

### 5.2 Chebyshev Liveness Inheritance

The memory evictor computes a safety boundary using Chebyshev's inequality:

```
P(|X - μ| ≥ k·σ) ≤ 1/k²
Safe boundary:  liveness > μ - k·σ
```

Injected objects arrive with zero accumulated liveness (t_alive = 0). The `LivenessAnchor` carries the source agent's heap statistics (μ, σ, k). The receiving heap initializes the injected block at `anchor_liveness = μ + σ`, which is above the safe boundary by construction for any k ≥ 1.

At k = 2, this guarantees P(false eviction) ≤ 25%.

### 5.3 Topological Assimilation Rescan

Before merging payload points into the shell, the kernel verifies:
- Payload β₀ = 1 (connected payload)
- Shell β₀ = 1 after merge remains valid

Points are added to the SparseGraph via the ε-neighborhood construction. The void empties. The manifold is ready for the next transfer.

---

## 6. The Orchestration System Call

```
sys_teleport_context(target, payload, governor):
    1. permit = governor.prepare_for_surgery()   → β = 0
    2. target.inject_into_void(payload)?          → topology check
    3. n = target.assimilate()                   → merge points
    4. governor.complete_surgery(permit)          → β restored
    5. return TeleportResult::Success { n }
```

On failure at step 2, the governor is restored before returning the error. The derivative zeroing is always un-done regardless of injection outcome.

---

## 7. Complexity Analysis

| Phase | Cost | Dominant Factor |
|-------|------|----------------|
| Source agent context loading | O(N²) | KV-cache attention (paid once) |
| Bridge: project N tokens | O(N·E·D) | Matmul per token |
| Bridge: build ε-graph | O(P²) | P ≤ 64 points |
| Transfer: governor clutch | O(1) | — |
| Transfer: injection + verification | O(P) | Betti check |
| Transfer: assimilation | O(P·V) | Shell graph size V |
| **Total per receiving agent** | **O(P)** | **P ≪ N** |

Concretely: source agent processes N = 10⁶ tokens once (O(N²) = O(10¹²) operations). Each receiving agent pays O(64) for the transfer. The amortized cost per receiver collapses to a constant.

---

## 8. Limitations

1. **Bridge requires embedding extraction.** The `EmbeddingBridge` accepts `&[[f64; E]]`. Extracting token embeddings from a running model requires instrumentation of the inference stack, which is runtime-specific.

2. **D = 3 restricts β₂ observability.** The manifold dimension is a compile-time constant. The `estimate_betti_1` method uses an Euler characteristic approximation rather than full persistent homology. A D > 3 implementation would require a proper Vietoris-Rips filtration library.

3. **Semantic fidelity is bounded by distortion ε.** The JL projection preserves pairwise distances within (1 ± ε). It does not preserve all semantic relationships exactly. The topology (relative ordering of distances) is preserved; absolute magnitudes are not.

4. **Remote transfer is unimplemented.** `TeleportTarget::RemoteVoid { agent_id }` exists in the type system. Serialization, routing, and consensus mechanisms are not part of this reference implementation.

---

## 9. Implementation Files

| File | Lines | Purpose |
|------|-------|---------|
| `bridge.rs` | ~350 | JL projection, L2 normalization, payload construction |
| `manifold.rs` | ~380 | HollowCubeManifold, SparseGraph, Betti computation |
| `governor.rs` | ~300 | PD governor, SurgeryPermit |
| `memory.rs` | ~280 | LivenessAnchor, ChebyshevGuard |
| `teleport.rs` | ~180 | sys_teleport_context, TeleportResult |
| `lib.rs` | ~90 | Documentation, re-exports |

Total: ~1,580 lines of Rust. Zero external runtime dependencies. No `std` required (only `libm` for transcendentals).
