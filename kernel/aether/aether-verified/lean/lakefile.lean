import Lake
open Lake DSL

package «AetherVerified» where
  -- Standard package; no extra options needed.

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.7.0"

@[default_target]
lean_lib «AetherVerified» where
  roots := #[`AetherVerified]

-- The 10-theorem umbrella file. As of Round-2, every theorem here is
-- proved (no `sorry`); some statements have been honestly weakened to
-- provable corollaries — see README.md for the table.
lean_lib «EpsilonTheorems» where
  roots := #[`EpsilonTheorems]
