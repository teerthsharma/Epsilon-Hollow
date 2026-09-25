/-
  AetherVerified.Betti
  --------------------
  Provenance: backs `kernel/aether/aether-verified/src/aether_betti.rs`.

  The Rust `oscillation_count` slides a width-4 window over a byte stream
  of length `n` and counts windows that return near their start while a
  middle value moves away. The Rust side claims one fact about it:
    oscillationCount data tol ≤ n − 3      (natural subtraction)
  because a window at index `i` exists only when `i + 3 < n`.

  The count is not a Betti number. It was previously called
  `betti1Heuristic` and paired with `betti_error_bound`,
  `count ≤ β₁ + n`, which holds for every `β₁ : Nat` including 0 and so
  constrains nothing about `β₁`. Its Rust mirror,
  `betti_error_bound_check`, could not return `false` on any input and
  was removed. The bound below is attained (a period-3 stream of values
  pairwise further apart than `tol` gives exactly `n − 3`), so it is the
  tightest bound that depends only on `n`.
-/

import Mathlib.Data.List.Basic

namespace AetherVerified.Betti

/-- Window-of-4 oscillation predicate mirroring `aether_betti::oscillation_at`.
    Holds at index `i` only when `i + 3 < data.length` and the window returns
    within `tol` of its start while a middle value moves further than `tol`. -/
def oscillationAt (data : List UInt8) (tol i : Nat) : Bool :=
  if h : i + 3 < data.length then
    let a := data.get ⟨i, by omega⟩
    let b := data.get ⟨i + 1, by omega⟩
    let c := data.get ⟨i + 2, by omega⟩
    let d := data.get ⟨i + 3, by omega⟩
    let dist : UInt8 → UInt8 → Nat := fun x y =>
      if x.toNat ≥ y.toNat then x.toNat - y.toNat else y.toNat - x.toNat
    let close := dist a d ≤ tol
    let middleFar := dist a b > tol ∨ dist a c > tol
    close && middleFar
  else
    false

/-- Number of indices at which `oscillationAt` holds. Mirrors
    `aether_betti::oscillation_count`. -/
def oscillationCount (data : List UInt8) (tol : Nat) : Nat :=
  ((List.range data.length).filter (fun i => oscillationAt data tol i)).length

/-- An index where the predicate holds is a valid window start. -/
theorem oscillationAt_window_fits (data : List UInt8) (tol i : Nat)
    (h : oscillationAt data tol i = true) : i + 3 < data.length := by
  unfold oscillationAt at h
  split at h
  · assumption
  · simp at h

/-- Prefix form: among the first `k` indices, at most `min k (n − 3)` hold. -/
theorem filter_range_le (data : List UInt8) (tol : Nat) :
    ∀ k : Nat, ((List.range k).filter (fun i => oscillationAt data tol i)).length
      ≤ min k (data.length - 3)
  | 0 => by simp
  | k + 1 => by
    have ih := filter_range_le data tol k
    rw [List.range_succ, List.filter_append, List.length_append]
    by_cases hk : oscillationAt data tol k = true
    · have hlt := oscillationAt_window_fits data tol k hk
      simp only [List.filter, hk, List.length_singleton]
      omega
    · simp only [List.filter, hk, List.length_nil]
      omega

/-- **Window-count bound.** The oscillation count is at most the number of
    width-4 window starts, `n − 3`. This is the bound the Rust module states. -/
theorem oscillationCount_le_windows (data : List UInt8) (tol : Nat) :
    oscillationCount data tol ≤ data.length - 3 :=
  le_trans (filter_range_le data tol data.length) (min_le_right _ _)

end AetherVerified.Betti
