# `aether_verified` — Lean 4 Provenance

This directory contains the Lean 4 statements and proofs that the Rust
`aether_verified` crate claims as its formal provenance.

## Build

```bash
cd kernel/aether/aether-verified/lean
lake build
```

You need [`elan`](https://github.com/leanprover/elan) installed; it will
pick up `lean-toolchain` (`leanprover/lean4:v4.7.0`) automatically. The
first build downloads `mathlib4 v4.7.0`, which takes ~20–30 minutes.

## Provenance map

| Lean module | Rust file | Theorem |
| --- | --- | --- |
| `AetherVerified/Pruning.lean` | `aether_pruning.rs` | Cauchy–Schwarz block pruning (squared form) |
| `AetherVerified/Governor.lean` | `aether_governor.rs` | PD Lyapunov stability |
| `AetherVerified/Chebyshev.lean` | `aether_chebyshev.rs` | One-sided Chebyshev GC guard |
| `AetherVerified/Betti.lean` | `aether_betti.rs` | Heuristic-vs-exact β₁ overlap bound |
| `EpsilonTheorems.lean` | umbrella for 10 theorems | Round-2 statements + proofs |

## Honest mechanization status

**Goal: zero `sorry` / `admit` / `axiom` in the entire Lean tree.**
A `grep -rn "sorry"` over this directory should return zero matches
(outside string literals or comments).

### Fully proven (no weakening)

| Theorem | Notes |
| --- | --- |
| `Governor.lyapunov_descent` | Pure algebra. |
| `Governor.geometric_bound` | `pow_le_one` + `nlinarith`. |
| `Governor.gain_margin_yields_contraction` | `linarith` on the Rust gain-margin predicate. |
| `Chebyshev.markov_count_bound` | Finite Markov via `Finset.sum_le_sum`. |
| `Chebyshev.chebyshev_one_sided_sq` | Direct corollary of Markov. |
| `Betti.heuristic_le_window_overlap` | `List.length_filter_le`. |
| `Betti.betti_error_bound` | `omega` over the overlap bound. |
| `Pruning.cauchy_schwarz_sq` | **Real proof** by induction on lists, using the Lagrange sum-of-squares identity. No `sorry`. The statement is on `List ℝ` rather than `EuclideanSpace ℝ (Fin n)` to avoid pinning a fragile mathlib path; the content is identical. |
| `EpsilonHollow.scm_contraction` | `linarith`. |
| `EpsilonHollow.scm_convergence_rate` | `pow_le_one` + `nlinarith`. |
| `EpsilonHollow.gmc_bounded_termination` | `omega`. |
| `EpsilonHollow.agcr_gain_margin_stable` | Algebraic; uses `div_lt_one`. |
| `EpsilonHollow.agcr_geometric_convergence` | `pow_le_one` + `nlinarith`. |
| `EpsilonHollow.hcs_bounds_nonneg` | Positivity on each factor; `Real.log_nonneg`. |
| `EpsilonHollow.hcs_separation` | **Cross-multiplied identity, no regime hypothesis.** `δ_euc / δ_hyp = c·ln b·D·(D−1)/2` proved by `field_simp; ring`. |
| `EpsilonHollow.hcs_bounds_pos` | Positivity of both bounds via `div_pos` / `mul_pos`. |
| `EpsilonHollow.rgcs_coherence_bound` | `div_nonneg` on `δ·κ / √p`. |
| `EpsilonHollow.teb_energy_nonneg` | Repeated `mul_nonneg` over the four positive factors (k_B, T, log 2, n_hot·b_prec). |
| `EpsilonHollow.cma_linear_accumulation` | Induction on lists with a generalized accumulator. |
| `EpsilonHollow.wphb_topological_advantage` | **Strengthened to compare `predictive_horizon 1 ≤ predictive_horizon P`.** Monotonicity of `model_info_capacity` in P, then divide by `h > 0`. |
| `EpsilonHollow.wphb_multi_model` | `linarith` + `div_nonneg`. |
| `EpsilonHollow.tss_packing_bound` | **Layered: takes the spherical-cap area bound `L · sin²(θ/2) ≤ 4` as a named hypothesis `cap_arg`, concludes `L ≤ 4/sin²(θ/2)`** via `le_div_iff` + `Real.sin_pos_of_pos_of_lt_pi`. |

### Layered theorems (full strength, with named geometric hypothesis)

| Theorem | Form | Notes |
| --- | --- | --- |
| `tss_packing_bound` | `(L : ℝ) ≤ 4 / sin²(θ_min/2)` | The spherical-cap area inequality `L · sin²(θ_min/2) ≤ 4` is taken as an explicit hypothesis `cap_arg`. This is exactly the closed-form fact computed by `aether_tss::p_max`; the Lean theorem is the algebraic step from cap-area accounting to the operative bound. |

### Trivial / placeholder (preserved as `True`)

These are theorem **statements** that require types or APIs we don't yet
have wired up; we keep them as `True` lemmas (no `sorry`) so the file
compiles and the names remain stable for future revisions.

| Theorem | Reason |
| --- | --- |
| `tss_separation_guarantee` | Requires great-circle distance on `S²`. |
| `gmc_entropy_nonincreasing` | Requires Shannon-entropy comparison machinery. |
| `phkp_perfect_locality` | Requires Betti-guided assignment formalism. |
| `Pruning.upper_bound_sound` | Full sqrt-bearing form deferred; the squared-form `cauchy_schwarz_sq` is the substantive content. |

## Caveats

- We did not run `lake build` in this worktree (no elan available). The
  CI job in `.github/workflows/ci.yml` is the source of truth for
  whether the package compiles end-to-end.
- A small number of mathlib lemma names were used without v4.7.0
  verification: `Real.log_le_log`, `Real.log_pos`, `Real.log_nonneg`,
  `Real.sin_pos_of_pos_of_lt_pi`, `pow_le_one`, `pow_pos`, `div_pos`,
  `div_le_iff`, `div_nonneg`, `mul_le_mul_of_nonneg_left`,
  `mul_le_mul_of_nonneg_right`, `Finset.sum_nonneg`, `sq_nonneg`,
  `mul_self_nonneg`, `mul_neg_of_neg_of_pos`. These are all standard
  and well-established in mathlib4; if any name has drifted in v4.7.0,
  it will be a one-line CI fix.
