# epsilon_core legacy host package

`epsilon_core/` is no longer the production theorem runtime. Seal OS treats the
Rust/Aether implementation under `kernel/epsilon/epsilon/crates/aether-core` as
the authoritative path for governor convergence, hyperbolic capacity,
hyperbolic geometry, the meta-controller, spectral contraction, spectral
entropy, and topological state sync.

The remaining host files are compatibility/reference scaffolding only. They must
not import deleted theorem modules. The Rust gate below enforces that rule and
checks the replacement Rust API:

```powershell
cargo +stable test --manifest-path kernel\epsilon\epsilon\crates\aether-core\Cargo.toml --test legacy_migration_gate
cargo +stable run --manifest-path kernel\seal-mkimage\Cargo.toml --release -- --check-aether-migration .
```

## Migration Status

| Legacy module | Current status | Rust replacement |
|---|---|---|
| `governor_convergence` | migrated | `aether_core::governor::GovernorConvergenceAnalyzer` |
| `hyperbolic_capacity` | migrated | `aether_core::hyperbolic_capacity::{HcsVerifier, h100_analysis}` |
| `hyperbolic_geometry` | migrated | `aether_core::hyperbolic_geometry::{PoincareBall, AngularMomentumTracker}` |
| `meta_controller` | migrated with host shim | `aether_core::meta_controller::{MetaController, ConstitutionalSafetyFilter}` |
| `spectral_contraction` | migrated | `aether_core::scm::{TelemetryOperator, LatentPredictor}` |
| `spectral_entropy` | migrated | `aether_core::spectral_entropy::SpectralCoherenceTracker` |
| `topological_state_sync` | migrated with host shim | `aether_core::tss::SphericalGridHashIndex` |

Compatibility shims in `agent.py`, `memory.py`, and `world_model.py` exist only
to keep old host references from importing deleted files while the repo moves to
Rust/Aether. New OS functionality belongs in Rust/Aether, not here.
