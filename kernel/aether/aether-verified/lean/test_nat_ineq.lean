import Mathlib.Data.Nat.Basic

-- Test if Nat.sub_le_self exists
#check Nat.sub_le_self
#check Nat.sub_le
#check Nat.le_sub_self_of_le

example (n m : Nat) : n - m ≤ n := by
  sorry
