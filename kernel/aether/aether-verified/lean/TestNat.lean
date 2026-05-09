import AetherVerified.Betti

theorem test_nat_chain (data : List UInt8) (tol β₁ : Nat) :
    betti1Heuristic data tol ≤ β₁ + data.length := by
  have h := heuristic_le_window_overlap data tol
  cases h with
  | inl h =>
    have h1 : betti1Heuristic data tol ≤ data.length - 2 := h
    have h2 : data.length - 2 ≤ data.length := Nat.sub_le data.length 2
    have h3 : data.length ≤ β₁ + data.length := Nat.le_add_left data.length β₁
    exact Nat.le_trans h1 (Nat.le_trans h2 h3)
  | inr h =>
    have h1 : betti1Heuristic data tol ≤ β₁ + data.length := h
    exact h1
