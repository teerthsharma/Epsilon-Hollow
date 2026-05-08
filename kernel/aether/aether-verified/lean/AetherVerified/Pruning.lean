/-
  AetherVerified.Pruning
  ----------------------
  Provenance: backs `kernel/aether/aether-verified/src/aether_pruning.rs`.

  The Rust `upper_bound_score` claims:
    ⟨q, k⟩ ≤ ‖q‖ · (‖μ_B‖ + r_B)
  via Cauchy–Schwarz + triangle inequality.

  We prove the squared form of Cauchy–Schwarz on `List ℝ` by direct
  induction (avoiding mathlib's `EuclideanSpace` / `inner` API guesses):
    (∑ᵢ qᵢ·kᵢ)²  ≤  (∑ᵢ qᵢ²) · (∑ᵢ kᵢ²).
  This is the load-bearing fact for `aether_pruning::upper_bound_score`.
-/

import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

namespace AetherVerified.Pruning

/-- Real Euclidean dot product on lists (zipped pointwise). -/
def dotL : List ℝ → List ℝ → ℝ
  | [],      _       => 0
  | _,       []      => 0
  | a :: as, b :: bs => a * b + dotL as bs

/-- Squared L2 norm on a list. -/
def normSqL : List ℝ → ℝ
  | []      => 0
  | a :: as => a * a + normSqL as

lemma normSqL_nonneg : ∀ v : List ℝ, 0 ≤ normSqL v
  | [] => le_refl 0
  | a :: as => by
      have h1 : 0 ≤ a * a := mul_self_nonneg a
      have h2 : 0 ≤ normSqL as := normSqL_nonneg as
      simp [normSqL]; linarith

/-- **Cauchy–Schwarz (squared form) on lists.**
    The load-bearing inequality for `aether_pruning::upper_bound_score`. -/
theorem cauchy_schwarz_sq : ∀ q k : List ℝ,
    (dotL q k) ^ 2 ≤ normSqL q * normSqL k
  | [], k => by
      show (dotL [] k) ^ 2 ≤ normSqL [] * normSqL k
      simp [dotL, normSqL]
  | _ :: _, [] => by
      show (dotL _ []) ^ 2 ≤ normSqL _ * normSqL []
      simp [dotL, normSqL]
  | a :: as, b :: bs => by
      -- Inductive step: use IH and the rank-1 inequality
      --   2·(a·b)·(∑ as·bs) ≤ a² · (∑ bs²) + b² · (∑ as²) + 2·… handled by AM-GM.
      have ih := cauchy_schwarz_sq as bs
      have ha2 : 0 ≤ a * a := mul_self_nonneg a
      have hb2 : 0 ≤ b * b := mul_self_nonneg b
      have hAs : 0 ≤ normSqL as := normSqL_nonneg as
      have hBs : 0 ≤ normSqL bs := normSqL_nonneg bs
      -- AM-GM-style: (a·√Bs − b·√As)² ≥ 0 ⇒ 2·a·b·√(As·Bs) ≤ a²Bs + b²As.
      -- Squared form: (a²Bs + b²As)² ≥ 4·a²b²·As·Bs, so any cross term works.
      -- We just expand and use IH + amgm on (a·√Bs ± b·√As)² style — but
      -- since we lack sqrt, we rely on:
      --   (a·dotL bs … − b·dotL as …)² ≥ 0 won't quite close it either.
      -- Robust route: prove
      --   ((a*b) + dotL as bs)² ≤ (a² + normSqL as)·(b² + normSqL bs)
      -- by expanding and using IH plus
      --   2·(a*b)·(dotL as bs) ≤ a²·normSqL bs + b²·normSqL as.
      -- The latter follows from (a·X − b·Y)² ≥ 0 where X² = normSqL bs and
      -- Y² = normSqL as — but we don't have those scalars without sqrt.
      -- Instead use: 2·u·v ≤ u² + v² with u = a·b, v = dotL as bs (weak).
      -- That's not tight enough. So we use the polynomial form:
      --   for any λ ≥ 0:  (λ·a − dotL_term)² … too messy without sqrt.
      --
      -- Cleanest: use Lagrange-style sum-of-squares directly on the
      -- two-variable case after IH.
      --
      -- Concretely: let A := normSqL as, B := normSqL bs, D := dotL as bs.
      -- Goal: (a*b + D)² ≤ (a² + A)·(b² + B).
      -- Expand RHS − LHS:
      --   = a²·B + b²·A + A·B − 2·a·b·D − D²
      --   = (A·B − D²) + (a²·B − 2·a·b·D + b²·A) + (b²·B − b²·B) -- regroup
      --   = (A·B − D²) + (a·√B − b·√A)²  [if we had sqrt]
      -- Without sqrt, multiply through: it suffices to show
      --   a²·B + b²·A ≥ 2·a·b·D, i.e. a²·B − 2·a·b·D + b²·A ≥ 0.
      -- This is a quadratic in (a,b); discriminant test:
      --   the quadratic form Q(a,b) = a²·B − 2·D·ab + b²·A
      --   is PSD iff B·A ≥ D², which is exactly IH.
      -- So combine IH (A·B ≥ D²) with the PSD bound.
      have ih' : (dotL as bs) ^ 2 ≤ normSqL as * normSqL bs := ih
      -- Set abbreviations.
      set A := normSqL as with hA
      set B := normSqL bs with hB
      set D := dotL as bs with hD
      show (a * b + D) ^ 2 ≤ (a * a + A) * (b * b + B)
      -- Suffices: 2·a·b·D ≤ a²·B + b²·A.
      -- Proof of that: by IH, D² ≤ A·B, and (a·b)² ≤ ... hmm need a·b sign.
      -- Actually: (a²B − 2abD + b²A) ≥ 0 ⇔ (a·b·D)² ≤ a²·B · b²·A · …
      -- We use: (aB' − bD'·sgn)² style is hard. Use instead:
      --   a²·B + b²·A − 2·a·b·D = (a·b's symmetric form)
      -- Key algebraic identity:
      --   (a²·B + b²·A)² ≥ 4·a²·b²·A·B ≥ 4·a²·b²·D²  [by IH]
      --   ⇒ a²·B + b²·A ≥ 2·|a·b·D| ≥ 2·a·b·D.
      -- Use nlinarith with these facts.
      have hkey : 2 * (a * b * D) ≤ a * a * B + b * b * A := by
        -- Step 1: (a·b·D)² ≤ (a²·B)·(b²·A).
        have hsq : (a * b * D) ^ 2 ≤ (a * a * B) * (b * b * A) := by
          have h1 : (a * b) ^ 2 * D ^ 2 ≤ (a * b) ^ 2 * (A * B) :=
            mul_le_mul_of_nonneg_left ih' (sq_nonneg _)
          have eq1 : (a * b * D) ^ 2 = (a * b) ^ 2 * D ^ 2 := by ring
          have eq2 : (a * a * B) * (b * b * A) = (a * b) ^ 2 * (A * B) := by ring
          rw [eq1, eq2]; exact h1
        -- Step 2: (x − y)² ≥ 0 with x = a²B, y = b²A: x² + y² ≥ 2xy.
        have hxy : (a * a * B - b * b * A) ^ 2 ≥ 0 := sq_nonneg _
        -- Step 3: from x² + y² ≥ 2xy and (abD)² ≤ xy:
        --   (x+y)² = x² + 2xy + y² ≥ 4xy ≥ 4(abD)² = (2·abD)².
        have hnn_xy : 0 ≤ (a * a * B) + (b * b * A) :=
          add_nonneg (mul_nonneg ha2 hBs) (mul_nonneg hb2 hAs)
        -- (2·abD)² ≤ (x+y)² via (x+y)² = (x-y)² + 4xy ≥ 4·(abD)² = (2abD)².
        have hsq2 : (2 * (a * b * D)) ^ 2
                  ≤ ((a * a * B) + (b * b * A)) ^ 2 := by
          have id1 : ((a * a * B) + (b * b * A)) ^ 2
                   = (a * a * B - b * b * A) ^ 2
                     + 4 * ((a * a * B) * (b * b * A)) := by ring
          have id2 : (2 * (a * b * D)) ^ 2 = 4 * (a * b * D) ^ 2 := by ring
          rw [id1, id2]
          have : 4 * (a * b * D) ^ 2 ≤ 4 * ((a * a * B) * (b * b * A)) := by
            linarith [hsq]
          linarith [hxy]
        -- From t² ≤ s² with s ≥ 0: case split on sign of (s + t).
        set t : ℝ := 2 * (a * b * D) with ht_def
        set s : ℝ := (a * a * B) + (b * b * A) with hs_def
        -- s² − t² = (s−t)(s+t) ≥ 0
        have hprod : 0 ≤ (s - t) * (s + t) := by nlinarith [hsq2]
        by_cases hst : 0 ≤ s + t
        · -- (s−t)(s+t) ≥ 0 with (s+t) ≥ 0 ⇒ if s+t > 0, s−t ≥ 0; if s+t = 0, then t = −s ≤ 0 ≤ s.
          rcases eq_or_lt_of_le hst with heq | hlt
          · -- s + t = 0 ⇒ t = −s. Since s ≥ 0, t = −s ≤ 0 ≤ s.
            have ht_eq : t = -s := by linarith
            linarith
          · -- s + t > 0
            have hsmt : 0 ≤ s - t := by
              by_contra hneg
              push_neg at hneg
              have : (s - t) * (s + t) < 0 := mul_neg_of_neg_of_pos hneg hlt
              linarith
            linarith
        · -- s + t < 0 ⇒ t < −s ≤ 0 ≤ s
          push_neg at hst
          linarith
      have goal_expand :
          (a * a + A) * (b * b + B) - (a * b + D) ^ 2
            = (a * a * B + b * b * A - 2 * (a * b * D)) + (A * B - D ^ 2) := by
        ring
      have hAB : 0 ≤ A * B - D ^ 2 := by linarith [ih']
      have hpos : 0 ≤ a * a * B + b * b * A - 2 * (a * b * D) := by linarith
      linarith [goal_expand]

/-- The pruning soundness statement, in squared form (skeleton).
    Full sqrt-bearing form deferred to a future revision. -/
theorem upper_bound_sound
    (q centroid k : List ℝ) (radius : ℝ)
    (_h_rad_nonneg : 0 ≤ radius) :
    True := trivial

end AetherVerified.Pruning
