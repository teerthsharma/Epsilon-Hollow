# epsilon_core test coverage status

This index records the test status of every module in `epsilon_core/`. Tests
live in the top-level `tests/` directory and are wired into CI via the `python`
job in `.github/workflows/ci.yml`.

Statuses:

- **Tested** — direct unit tests exercise the public API with assertions.
- **Smoke-tested** — import + construct + a single forward call covered.
- **Experimental** — theorem scaffolding or unfinished module; not covered by CI.

## Modules

| Module | Status | Notes |
|---|---|---|
| `contract.py` | Tested | `tests/test_contract.py` (round-trip + schema). |
| `hyperbolic_geometry.py` | Tested | `tests/test_hyperbolic_geometry.py` (Poincare ball, EGPB). |
| `memory.py` | Tested | `tests/test_memory.py` (UnionFind + manifold memory). |
| `topological_state_sync.py` | Tested | `tests/test_topological_state_sync.py` (grid hash + bounds). |
| `perception.py` | Smoke-tested | `tests/test_perception.py` (encoder shape + determinism). |
| `world_model.py` | Smoke-tested | `tests/test_world_model.py` (construct + ingest + query). |
| `__init__.py` | n/a | Re-exports world_model. |
| `adapter.py` | Experimental | not covered by CI |
| `agent.py` | Experimental | not covered by CI |
| `angular_sparse_attention.py` | Experimental | not covered by CI |
| `cross_manifold_alignment.py` | Experimental | theorem scaffold, Lean verification pending |
| `geodesic_consolidation.py` | Experimental | theorem scaffold, Lean verification pending |
| `governor_convergence.py` | Experimental | theorem scaffold, Lean verification pending |
| `hyperbolic_capacity.py` | Experimental | theorem scaffold, Lean verification pending |
| `main.py` | Experimental | demo entry point |
| `meta_controller.py` | Experimental | not covered by CI |
| `parallel_riemannian.py` | Experimental | theorem scaffold, Lean verification pending |
| `persistent_kv_partition.py` | Experimental | theorem scaffold, Lean verification pending |
| `riemannian_optimizer.py` | Experimental | not covered by CI |
| `simulation_engine.py` | Experimental | not covered by CI |
| `spectral_contraction.py` | Experimental | theorem scaffold, Lean verification pending |
| `spectral_entropy.py` | Experimental | not covered by CI |
| `thermodynamic_plasticity.py` | Experimental | theorem scaffold, Lean verification pending |
| `world_model_horizon.py` | Experimental | theorem scaffold, Lean verification pending |

Modules tagged Experimental carry a `# Status: experimental` header where
applicable. They may compile (verified by `python -m compileall`) but their
behavior is not asserted by tests. Add coverage before relying on them.
