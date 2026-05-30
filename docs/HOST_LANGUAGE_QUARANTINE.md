# Legacy host script quarantine

Seal OS runtime source is Rust, assembly, and Aether-Lang DSL. Python files that
remain in this repository are not part of the Seal OS runtime, boot path, kernel
build, theorem gate, or VM proof.

## Current rule

- `kernel/seal-os/**`: no Python.
- `kernel/epsilon/epsilon/crates/**`: no Python.
- `kernel/aether/Aether-Lang/crates/**`: no Python.
- `.gitattributes` marks Rust, assembly, and Aether-family DSL files as the
  visible OS language surface. Host scripts, JS/TS app scaffolding, Dockerfiles,
  requirements text, generated Lean build output, assets, configs, and legacy
  Python are vendored/generated/documentation so GitHub language stats do not
  pretend they are Seal OS runtime languages.
- `seal-mkimage --check-language-hygiene .` enforces the production-root ban and
  the Linguist quarantine rules.

## Why files still exist

The remaining Python files are legacy host research scaffolding, examples,
prototype app wrappers, or tests from before the Rust/Aether migration. They are
kept only when deleting them would erase behavior that still needs a Rust/Aether
replacement.

Deletion rule:

1. Read the legacy host script.
2. Add or extend the Rust/Aether replacement.
3. Add a Rust test or `seal-mkimage` audit gate that proves the replacement API.
4. Run the gate.
5. Delete the legacy host script only after the replacement passes.

## Epsilon theorem migration

`kernel/epsilon/epsilon_core/` is legacy host scaffolding. The authoritative
theorem/runtime replacement is `kernel/epsilon/epsilon/crates/aether-core`.

The migration gate currently requires Rust/Aether replacement symbols for:

- geodesic consolidation / bounded entropy-reducing merge
- cross-manifold alignment / Procrustes information preservation
- governor convergence
- hyperbolic capacity
- hyperbolic geometry
- meta-controller and constitutional safety filter
- persistent KV-cache partitioning
- spectral contraction
- spectral entropy
- thermodynamic plasticity / Landauer erasure bound
- topological state sync
- world-model predictive horizon bound

Required verification:

```powershell
cargo +stable test --manifest-path kernel\epsilon\epsilon\crates\aether-core\Cargo.toml --test legacy_migration_gate
cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml --release -- --check-aether-migration .
```

The migration gate fails if the deleted theorem Python modules reappear, if
root tests import those deleted modules, or if the Rust gate stops covering the
restored GMC entropy merge bounds, governor history, HCS capacity bounds,
TEB/Landauer energy bounds, meta-controller tool registry, Haar transform, and
PHKP tier locality, TSS empty/dynamic grid semantics, CMA alignment bounds, and
WPHB horizon formulas.

## LAAMBA Governor migration

`apps/laamba-governor/` is not part of the bare-metal Seal OS boot path, but it
still contains host prototype Python. The native runtime target is the kernel
app in `kernel/seal-os/src/apps/laamba_governor.rs`, the Aether script in
`apps/laamba-governor/native/governor.aether`, and Rust/Aether math in
`kernel/epsilon/epsilon/crates/aether-core`.

| Host file or surface | Runtime replacement target | Status |
| --- | --- | --- |
| `apps/laamba-governor/topological_governor_ml.py` | `kernel/epsilon/epsilon/crates/aether-core/src/governor.rs` plus `apps/laamba-governor/native/governor.aether` | port `VitalsExtractor`, `PolicyNet`, preview, analyze, and rank before deletion |
| `apps/laamba-governor/topological_governor_full.py` | `kernel/seal-os/src/apps/laamba_governor.rs` plus `aether-core` governor primitives | migrate usecase-level scoring into Rust/Aether fixtures |
| `apps/laamba-governor/governor_orchestrator.py` | `apps/laamba-governor/src-tauri/src/pipeline/executor.rs` without host Python subprocesses | port `OutcomeComparator` and topology score loop first |
| `apps/laamba-governor/formula_engine.py` | Aether formula AST and `.aether` manifest emission | parser/compiler still host prototype |
| `apps/laamba-governor/cli/governor_cli.py` | Rust Tauri commands returning the same JSON contracts | replace `dataset_preview`, then `run_analysis`, then `run_battle` |
| `apps/laamba-governor/engines/wrappers/*.py` | Rust engine manager commands or Aether commands | add manifest audit before deletion |
| `apps/laamba-governor/governor-studio/main.py` | Tauri UI plus kernel-native LAAMBA window | legacy studio only; no Seal OS runtime dependency |
| `apps/laamba-governor/cli/generate_seed_data.py` | checked-in fixtures or Rust fixture generator | reference-only until fixtures are native |

Next narrow conversion:

1. Replace `dataset_preview` without changing the JSON response shape.
2. Add a `seal-mkimage` or Rust test gate that bans this command from spawning
   `python`.
3. Move to `run_analysis` only after the preview command is native and green.

## Remaining work

The host quarantine is not the final state. The final state is fewer legacy host
scripts and more Rust/Aether modules with tests. Until then, quarantined Python
must stay outside production OS roots and outside boot/build/proof paths.
