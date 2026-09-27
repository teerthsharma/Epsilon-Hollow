# Theorem Reference - T1 Through T10

This is the theorem map for Seal OS. It separates three things that must not be blurred:

- Runtime application: theorem shapes real kernel decisions.
- Boot verification: `kernel/seal-os` runs no_std checks from `aether_verified` before claiming the theorem core is alive.
- Lean proof strength: the formal artifact is full, layered, partial, or placeholder.

## Boot Gate

`kernel/seal-os/src/lib.rs` links `aether_verified` and calls `verify_topology_theorems()` from `init_theorems()`. The kernel sets `THEOREM_STATES[0..10]` from those checks. T4 is judged at the gains and step every runtime governor uses (`GOVERNOR_ALPHA`, `GOVERNOR_BETA`, `GOVERNOR_DT` = 0.01, 0.05, 0.01); there the gain margin α + β/dt is 5.01, so T4 is reported as not certified and boot continues. Any other failed check panics (commit `3c14df0`).

Boot status lines:

```text
[THEOREM] T4/AGCR NOT CERTIFIED: alpha+beta/dt=5.01 >= 1 at dt=0.01
[BOOT] 9 of 10 theorems VERIFIED; T4/AGCR NOT CERTIFIED; T1-T3, T5 ACTIVE in runtime paths
```

`seal-mkimage --check-theorem-log` requires the other nine `VERIFIED` lines, the T4 refusal with its margin and `dt`, and the `9 of 10` summary, and rejects `[THEOREM] T4/AGCR VERIFIED` while the margin is at least 1. No step size earns T4: the margin treats the plant as unit gain, while the loop the code runs has gain 1/ε² ≈ 10⁶ at equilibrium (commit `dcc35b6`), so the governor needs a redesign.

## Summary

| ID | Name | Runtime role in Seal OS | Boot check | Lean strength |
|---|---|---|---|---|
| T1 | TSS - Topological State Synchronization | ManifoldFS lookup/placement, scheduler task cells, TopoRAM locality | Yes | Layered TSS packing algebra; separation placeholder |
| T2 | SCM - Spectral Contraction Mapping | File prefetch state, scheduler prediction, TopoRAM prefetch | Yes | Full algebraic contraction and convergence |
| T3 | GMC - Geodesic Memory Consolidation | ManifoldFS entropy merge, memory fragmentation heuristics | Yes | Bounded termination full; entropy comparison placeholder |
| T4 | AGCR - Adaptive Governor Convergence Rate | Scheduler timeslice, ManifoldFS governor, compositor pacing | Yes; refused at the runtime step dt = 0.01, margin 5.01 | Full governor/gain-margin algebra |
| T5 | HCS - Hyperbolic Capacity Separation | Directory depth ratio, memory lifetime class, power mapping | Yes | Full cross-multiplied separation identity |
| T6 | RGCS - Ring-Allreduce Gradient Coherence | ML/HFT world-model sync bound, boot-gated | Yes | Full non-negative tangent-deviation bound |
| T7 | PHKP - Persistent Homology KV Partitioning | ML cache/latency bound, boot-gated | Yes | Placeholder locality lemma; Rust latency check active |
| T8 | TEB - Thermodynamic Erasure Bound | ML energy floor, boot-gated | Yes | Full non-negative Landauer energy bound |
| T9 | CMA - Cross-Manifold Alignment | Pipeline alignment error bound, boot-gated | Yes | Full non-negative accumulation; Rust SVD/curvature check active |
| T10 | WPHB - World Predictive Horizon Bound | Predictive horizon bound, boot-gated | Yes | Full monotonic horizon inequality |

## Status at runtime inputs

`init_theorems` in `kernel/seal-os/src/lib.rs` (`lib.rs:2122-2193`) evaluates `verify_topology_theorems` (`lib.rs:2195-2249`) at boot. The boot log of run 36165748105 printed all ten as `VERIFIED`. Since commit `3c14df0`, a T4 whose gain margin fails at the runtime step is reported, not fatal: boot prints `[THEOREM] T4/AGCR NOT CERTIFIED: alpha+beta/dt=5.01 >= 1 at dt=0.01` and `[BOOT] 9 of 10 theorems VERIFIED; T4/AGCR NOT CERTIFIED; T1-T3, T5 ACTIVE in runtime paths`, and any other false entry still panics (in-kernel harness 564 / 564 on that commit, local QEMU, stated in its message). What each line establishes differs per theorem. "Certified" below means the condition holds at the inputs the running kernel uses; "refused" means it was evaluated there and fails; "not checked" means it was evaluated only on fixed boot constants, or nothing at runtime consumes it. Lean sources are in `kernel/aether/aether-verified/lean/EpsilonTheorems.lean`; the Lean 4 CI job built them in run 36165748105, and `--check-lean-proof-hygiene` rejects `sorry`, `admit` and `axiom`.

Equation numbers and "section 5" refer to [RESULTS.md, Theoretical Foundation](RESULTS.md#theoretical-foundation). This table moved here from the README on 2026-09-27.

| ID | Name | Lean artifact | Boot evaluation | Status at runtime inputs, and where decided |
|---|---|---|---|---|
| T1 | TSS | `tss_packing_bound`, layered on a named cap-area hypothesis; `tss_separation_guarantee` is a `True` placeholder | Packing bound and pairwise separation of the 8 boot centroids | **Partly.** The scheduler, compositor and firewall indices (commit `87d7b10`) and the router index (commit `9702061`) build from `aether_core::tss::CUBE_CENTROIDS` (`process/scheduler.rs:267`, `wm/compositor.rs:130`, `net/firewall.rs:76`, `net/topological.rs:67`), the cube whose values the boot check evaluates from its own copy in `lib.rs`; nothing checks that the two copies agree. Since commit `2014ddb` the separation check certifies a pair only beyond its rounding radius. The ManifoldFS cells (`fs/voronoi_cap.rs:74`) still build from another centroid set. |
| T2 | SCM | `scm_contraction` proves only the inequality $1 - \alpha < 1$ for $\alpha \in (0,1)$ | One pair contracted at $\alpha = 0.1$ | **Certified**, elementarily: the operator $(1-\alpha)S + \alpha P$ contracts by $1-\alpha < 1$ at the runtime gains 0.7 (`process/scheduler.rs:280`, `fs/manifold_fs.rs:277`) and 0.3 (`net/firewall.rs:67`). |
| T3 | GMC | `gmc_bounded_termination` proved; `gmc_entropy_nonincreasing` is a `True` placeholder | Fixed constants (100, 50, 1000) and `max_merges(8) == 7` | **Not checked.** |
| T4 | AGCR | `agcr_gain_margin_stable`, conditional on (9) | Evaluated at `GOVERNOR_DT` = 0.01 (`lib.rs:2217-2223`), margin 5.01; before commit `3c14df0`, at $\Delta t = 1.0$ | **Refused**, at boot and at runtime. Every runtime call passes `GOVERNOR_DT` (`process/scheduler.rs:548`, `fs/manifold_fs.rs:810`, `wm/compositor.rs:520`, `549`), and the boot line reads `NOT CERTIFIED`. `epsilon-os` refuses it too (`epsilon-os/src/manifold_fs.rs:31-39`, `world.rs:596-604`). No step size earns it ([RESULTS.md, section 5](RESULTS.md#5-the-t4-governor-and-its-gain-margin)). |
| T5 | HCS | `hcs_separation`, an exact identity | Fixed constants | **Not checked.** |
| T6 | RGCS | `rgcs_coherence_bound` (non-negativity) | Fixed constants | **Not checked**; no runtime consumer. |
| T7 | PHKP | `phkp_perfect_locality` is a `True` placeholder | Fixed constants | **Not checked**; no runtime consumer. |
| T8 | TEB | `teb_energy_nonneg` | Landauer bound at 300 K lies in (2.8e-21, 2.9e-21) J | **Not checked**; no runtime consumer. |
| T9 | CMA | `cma_linear_accumulation` | Fixed constants | **Not checked**; no runtime consumer. |
| T10 | WPHB | `wphb_topological_advantage`, `wphb_multi_model` | Fixed constants | **Not checked**; no runtime consumer. |

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
| Betti | `kernel/aether/aether-verified/src/aether_betti.rs`, `kernel/aether/aether-verified/lean/AetherVerified/Betti.lean` | Window-of-4 oscillation count ≤ n − 3 (`oscillationCount_le_windows`); not a Betti number |
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
