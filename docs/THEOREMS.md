# Theorem Reference - T1 Through T10

This is the theorem map for Seal OS. It separates three things that must not be blurred:

- Runtime application: theorem shapes real kernel decisions.
- Boot verification: `kernel/seal-os` runs no_std checks from `aether_verified` before claiming the theorem core is alive.
- Lean proof strength: the formal artifact is full, layered, partial, or placeholder.

## Boot Gate

`kernel/seal-os/src/lib.rs` links `aether_verified` and calls `verify_topology_theorems()` from `init_theorems()`. The kernel sets `THEOREM_STATES[0..10]` from those checks and panics if any fail.

Boot status string:

```text
[BOOT] All T1-T10 theorems VERIFIED; T1-T5 ACTIVE in runtime paths
```

## Summary

| ID | Name | Runtime role in Seal OS | Boot check | Lean strength |
|---|---|---|---|---|
| T1 | TSS - Topological State Synchronization | ManifoldFS lookup/placement, scheduler task cells, TopoRAM locality | Yes | Layered TSS packing algebra; separation placeholder |
| T2 | SCM - Spectral Contraction Mapping | File prefetch state, scheduler prediction, TopoRAM prefetch | Yes | Full algebraic contraction and convergence |
| T3 | GMC - Geodesic Memory Consolidation | ManifoldFS entropy merge, memory fragmentation heuristics | Yes | Bounded termination full; entropy comparison placeholder |
| T4 | AGCR - Adaptive Governor Convergence Rate | Scheduler timeslice, ManifoldFS governor, compositor pacing | Yes | Full governor/gain-margin algebra |
| T5 | HCS - Hyperbolic Capacity Separation | Directory depth ratio, memory lifetime class, power mapping | Yes | Full cross-multiplied separation identity |
| T6 | RGCS - Ring-Allreduce Gradient Coherence | ML/HFT world-model sync bound, boot-gated | Yes | Full non-negative tangent-deviation bound |
| T7 | PHKP - Persistent Homology KV Partitioning | ML cache/latency bound, boot-gated | Yes | Placeholder locality lemma; Rust latency check active |
| T8 | TEB - Thermodynamic Erasure Bound | ML energy floor, boot-gated | Yes | Full non-negative Landauer energy bound |
| T9 | CMA - Cross-Manifold Alignment | Pipeline alignment error bound, boot-gated | Yes | Full non-negative accumulation; Rust SVD/curvature check active |
| T10 | WPHB - World Predictive Horizon Bound | Predictive horizon bound, boot-gated | Yes | Full monotonic horizon inequality |

## Runtime Applications

T1 is applied by `kernel/seal-os/src/fs/voronoi_cap.rs`, `kernel/seal-os/src/fs/manifold_fs.rs`, `kernel/seal-os/src/process/scheduler.rs`, and `kernel/seal-os/src/memory/topo_ram.rs`.

T2 is applied by `kernel/seal-os/src/fs/manifold_fs.rs`, `kernel/seal-os/src/process/scheduler.rs`, and `kernel/seal-os/src/memory/topo_ram.rs`.

T3 is applied by `kernel/seal-os/src/fs/manifold_fs.rs`, `kernel/seal-os/src/memory/topo_ram.rs`, and `kernel/seal-os/src/drivers/acpi/topological_power.rs`.

T4 is applied by `kernel/seal-os/src/process/scheduler.rs`, `kernel/seal-os/src/fs/manifold_fs.rs`, `kernel/seal-os/src/wm/compositor.rs`, and `kernel/seal-os/src/memory/swap.rs`.

T5 is applied by `kernel/seal-os/src/fs/manifold_fs.rs`, `kernel/seal-os/src/memory/topo_ram.rs`, `kernel/seal-os/src/memory/virt.rs`, and `kernel/seal-os/src/drivers/acpi/topological_power.rs`.

T6-T10 are not yet hot-path runtime governors in `kernel/seal-os`; they are boot-verified HFT/ML theorem gates from `kernel/aether/aether-verified/src/aether_world.rs`.

## Source Map

| Theorem | Rust source | Lean source |
|---|---|---|
| T1 | `kernel/aether/aether-verified/src/aether_tss.rs`, `kernel/epsilon/epsilon/crates/aether-core/src/tss.rs` | `kernel/aether/aether-verified/lean/EpsilonTheorems.lean` |
| T2 | `kernel/aether/aether-verified/src/aether_scm.rs`, `kernel/epsilon/epsilon/crates/aether-core/src/scm.rs` | `kernel/aether/aether-verified/lean/EpsilonTheorems.lean` |
| T3 | `kernel/aether/aether-verified/src/aether_gmc.rs` | `kernel/aether/aether-verified/lean/EpsilonTheorems.lean` |
| T4 | `kernel/aether/aether-verified/src/aether_agcr.rs`, `kernel/epsilon/epsilon/crates/aether-core/src/governor.rs` | `kernel/aether/aether-verified/lean/AetherVerified/Governor.lean`, `EpsilonTheorems.lean` |
| T5 | `kernel/aether/aether-verified/src/aether_hcs.rs` | `kernel/aether/aether-verified/lean/EpsilonTheorems.lean` |
| T6-T10 | `kernel/aether/aether-verified/src/aether_world.rs` | `kernel/aether/aether-verified/lean/EpsilonTheorems.lean` |

Additional proof kernels:

| Kernel | Source | Purpose |
|---|---|---|
| Betti | `kernel/aether/aether-verified/src/aether_betti.rs`, `kernel/aether/aether-verified/lean/AetherVerified/Betti.lean` | Heuristic-vs-exact Betti-1 overlap bound |
| Chebyshev | `kernel/aether/aether-verified/src/aether_chebyshev.rs`, `kernel/aether/aether-verified/lean/AetherVerified/Chebyshev.lean` | One-sided Chebyshev guard |
| Pruning | `kernel/aether/aether-verified/src/aether_pruning.rs`, `kernel/aether/aether-verified/lean/AetherVerified/Pruning.lean` | Cauchy-Schwarz pruning bound; sqrt upper-bound form partial |

## Open Formalization Work

The theorem core is now wired into Seal OS boot. Remaining proof work is to replace placeholder Lean statements for TSS separation, GMC entropy comparison, PHKP locality, and the sqrt-bearing pruning upper-bound theorem with full formal machinery. Until those are done, the README should say "boot-verified theorem gate" and "proof strength tracked here," not "all proof forms have identical strength."

## Closure Criteria

A theorem is counted as fully closed only when all of these are true:

1. The Rust boot gate checks the theorem-specific invariant and fails closed.
2. The runtime path using the theorem is named in this document.
3. The Lean artifact has no theorem placeholder such as `sorry`, `admit`, or a proof reduced to `True`.
4. The Rust gate input can be traced to the proof artifact or to a documented certificate.
5. Property tests cover a meaningful theorem domain, not only one fixed constant example.

Until all five are true, the honest status is "boot-gated" or "runtime active",
not "formally complete."
