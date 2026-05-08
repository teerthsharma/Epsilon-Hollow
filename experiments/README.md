# Experiments

These tracks are experimental, not built or tested by CI. They are preserved for research continuity.

## Contents

- `apeiron-runtime/` — Apeiron runtime (separate Cargo workspace, half-built).
- `aether-lang/` — Aether-Lang language toolchain (separate Cargo workspace, half-built).
- `aether-nexus/` — Orchestrator binary depending on Aether-Lang, Epsilon, Aether-Link, and apeiron-runtime. Path deps may be broken after relocation; revisit before reintegrating.

None of these are members of the top-level workspace at the repo root. Build them in-tree if needed.
