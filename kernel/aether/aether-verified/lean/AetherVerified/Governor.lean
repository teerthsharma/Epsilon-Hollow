/-
  AetherVerified.Governor
  -----------------------
  Provenance: backs `kernel/aether/aether-verified/src/aether_governor.rs`.

  What is proved here is scalar: the Lyapunov function `V(e) = e²`
  does not increase under `e ↦ ρ · e` for `|ρ| ≤ 1`, and the iterate
  bound that follows from it.

  No theorem here is about the Rust `governor_step`, and none of these
  results implies that it decreases `|e|`. The PD step is not
  multiplication of `e` by a fixed `ρ`: its derivative term reads
  `e_prev`, and with `α = 0.01, β = 0.05, dt = 1` (where
  `gain_margin_refined` holds) the step from `ε = 0.28, e_prev = 0.5,
  δ = 0.2, r = 0.3` raises `|e|` from 0.414286 to 0.414650
  (`tests/house_governor_step_descent.rs`). The Rust crate checks
  descent of a computed step at runtime with `lyapunov_descent_holds`.
  There is no `govStep_lyapunov`.
-/

import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace AetherVerified.Governor

/-- Scalar Lyapunov function `V(e) = e²`. -/
def V (e : ℝ) : ℝ := e ^ 2

/-- **Scalar Lyapunov descent for a contraction.**

    If `e' = ρ · e` with `|ρ| ≤ 1`, then `V(e') ≤ V(e)`.
    Applies to a map of that form only; the Rust PD step is not one
    (see the module comment). -/
theorem lyapunov_descent (ρ e : ℝ) (h_ρ : |ρ| ≤ 1) :
    V (ρ * e) ≤ V e := by
  unfold V
  have hρsq : ρ ^ 2 ≤ 1 := by
    have := sq_abs ρ
    have hρ2 : |ρ| ^ 2 ≤ (1 : ℝ) ^ 2 :=
      pow_le_pow_left (abs_nonneg ρ) h_ρ 2
    simpa [sq_abs] using hρ2
  have he2 : 0 ≤ e ^ 2 := sq_nonneg _
  calc (ρ * e) ^ 2 = ρ ^ 2 * e ^ 2 := by ring
    _ ≤ 1 * e ^ 2 := by
        exact mul_le_mul_of_nonneg_right hρsq he2
    _ = e ^ 2 := by ring

/-- **Geometric envelope never exceeds its start.**

    For `|ρ| ≤ 1`, `|ρ|^t · |e₀| ≤ |e₀|`. This bounds the envelope
    `|ρ|^t · |e₀|`; it does not state that any sequence satisfies
    `|e_t| ≤ |ρ|^t · |e₀|`, and it is not a statement about the Rust
    `governor_step`. -/
theorem geometric_bound (ρ e₀ : ℝ) (t : Nat) (h_ρ_nn : 0 ≤ |ρ|) (h_ρ_lt : |ρ| ≤ 1) :
    |ρ| ^ t * |e₀| ≤ |e₀| := by
  have h_pow : |ρ| ^ t ≤ 1 := pow_le_one t h_ρ_nn h_ρ_lt
  have h_e : 0 ≤ |e₀| := abs_nonneg _
  nlinarith [pow_nonneg h_ρ_nn t]

/-- Mirrors the Rust `gain_margin_refined(dt)`: `dt ≥ 1 ∧ 0.01 + 0.05/dt < 1`,
    the gain margin of the default gains `α = 0.01, β = 0.05` only.
    `gain_margin_yields_contraction` shows the number `1 − (0.01 + 0.05/dt)`
    lies in `(0, 1)`; nothing connects that number to the Rust step. -/
def gainMarginRefined (dt : ℝ) : Prop :=
  1 ≤ dt ∧ 0.01 + 0.05 / dt < 1

theorem gain_margin_yields_contraction (dt : ℝ) (h : gainMarginRefined dt) :
    0 < (1 : ℝ) - (0.01 + 0.05 / dt) ∧ (1 - (0.01 + 0.05 / dt)) < 1 := by
  obtain ⟨hdt, hsum⟩ := h
  have hdt_pos : 0 < dt := lt_of_lt_of_le zero_lt_one hdt
  have h005 : 0 < (0.05 : ℝ) / dt := by positivity
  refine ⟨?_, ?_⟩
  · linarith
  · have h_sum_pos : 0 < (0.01 : ℝ) + 0.05 / dt := by positivity
    linarith

end AetherVerified.Governor