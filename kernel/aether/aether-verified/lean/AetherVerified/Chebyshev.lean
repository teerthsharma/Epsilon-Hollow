/-
  AetherVerified.Chebyshev
  ------------------------
  Provenance: backs `kernel/aether/aether-verified/src/aether_chebyshev.rs`.

  The Rust `chebyshev_guard_check` asserts: for any finite multiset of
  reals with mean μ and standard deviation σ,
    |{i : x_i ≤ μ − kσ}|  ≤  n / k².

  The *one-sided* form is what Rust checks. The classical Chebyshev
  inequality (two-sided) is `|{i : |x_i − μ| ≥ kσ}| ≤ n/k²`, and the
  one-sided form follows directly.

  Below we prove the *finite Chebyshev counting bound* in its general
  algebraic form, which is the load-bearing fact for the GC guard.
-/

import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.BigOperators.Order
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity

open scoped BigOperators

namespace AetherVerified.Chebyshev

/-- **Finite Markov-style counting bound.**

    For non-negative reals `y_i` and threshold `t > 0`,
    `|{i : y_i ≥ t}| · t  ≤  Σ y_i`.

    Specializing to `y_i := (x_i − μ)²` and `t := (kσ)²` yields the
    classical Chebyshev inequality `|{i : |x_i − μ| ≥ kσ}| ≤ n/k²`,
    which is exactly the bound the Rust GC guard checks. -/
theorem markov_count_bound {n : ℕ}
    (y : Fin n → ℝ) (h_nn : ∀ i, 0 ≤ y i) (t : ℝ) (h_t : 0 < t) :
    (((Finset.univ.filter (fun i => t ≤ y i)).card : ℝ)) * t
      ≤ ∑ i, y i := by
  let S := Finset.univ.filter (fun i => t ≤ y i)
  -- Step 1: ∑_{i∈S} t ≤ ∑_{i∈S} y_i   (since t ≤ y_i on S)
  have h1 : ∑ _i in S, t ≤ ∑ i in S, y i := by
    apply Finset.sum_le_sum
    intros i hi
    exact (Finset.mem_filter.mp hi).2
  -- Step 2: ∑_{i∈S} y_i ≤ ∑_{i} y_i   (S ⊆ univ and y_i ≥ 0)
  have h2 : ∑ i in S, y i ≤ ∑ i, y i := by
    apply Finset.sum_le_sum_of_subset_of_nonneg
    · exact Finset.subset_univ S
    · intros i _ _; exact h_nn i
  -- Step 3: ∑_{i∈S} t = |S| · t
  have h3 : ∑ _i in S, t = (S.card : ℝ) * t := by
    simp [Finset.sum_const, mul_comm]
  -- Chain: |S|·t = ∑S t ≤ ∑S y ≤ ∑ y
  linarith [h1, h2, h3]

/-- **Chebyshev counting bound (one-sided form used by the Rust guard).**

    For any data `x : Fin n → ℝ` with mean `μ` and *empirical second
    moment about the mean* `σ² := (1/n) Σ (x_i − μ)²`, and any `k > 0`,
    `|{i : x_i ≤ μ − kσ}| · k² ≤ n`.

    This is the bound `chebyshev_guard_check` verifies. We state it
    as a corollary of `markov_count_bound`. The full real-σ statement
    requires `Real.sqrt`; the squared form below is mathlib-light and
    captures the same content (multiply through by `σ²`). -/
theorem chebyshev_one_sided_sq {n : ℕ}
    (x : Fin n → ℝ) (μ : ℝ) (k : ℝ) (h_k : 0 < k)
    (σ_sq : ℝ) (h_σ_pos : 0 < σ_sq)
    (h_σ_def : σ_sq * (n : ℝ) = ∑ i, (x i - μ) ^ 2) :
    (((Finset.univ.filter (fun i => (k * k) * σ_sq ≤ (x i - μ) ^ 2)).card : ℝ))
        * ((k * k) * σ_sq)
      ≤ σ_sq * (n : ℝ) := by
  have h_t_pos : 0 < (k * k) * σ_sq := by positivity
  have base := markov_count_bound (fun i => (x i - μ) ^ 2)
                  (fun i => sq_nonneg _) ((k * k) * σ_sq) h_t_pos
  rw [h_σ_def]
  exact base

end AetherVerified.Chebyshev