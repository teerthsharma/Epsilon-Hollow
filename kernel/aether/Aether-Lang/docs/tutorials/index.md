# Runnable Examples

The examples directory contains host-side scripts for exercising language and
runtime concepts. Treat them as examples unless a test or proof gate promotes a
specific behavior to an active claim.

## Example Classes

| Example class | Files | Purpose |
| --- | --- | --- |
| Basic manifold scripts | `examples/simple.aegis`, `examples/hello_manifold.aegis` | Parser/interpreter smoke paths |
| Regression demos | `examples/regression_demo.aegis`, `examples/benchmark_suite.aegis` | Regression statement and manifold workflow examples |
| Visualization demos | `examples/visualization_demo.aegis`, `examples/3d_cluster.aegis` | Render metadata and geometric examples |
| Seal OS demos | `examples/seal_demo.aegis`, `examples/seal_loop_demo.aegis` | Integration-oriented examples; require separate OS evidence |
| LLM demos | `examples/llm_demo.ag`, `examples/llm_benchmark.*` | Roadmap/prototype examples unless benchmark artifacts are present |

## Host Run Pattern

Build and run commands depend on which crate is included in the active
workspace. In the current nested checkout, several CLI crates are present but
excluded from the root workspace. Confirm the crate path before documenting a
command as supported.

```powershell
cargo test --manifest-path crates/aether-lang/Cargo.toml
```

## Promotion Rule

An example becomes documentation evidence only when it has:

- a deterministic input;
- an expected output or verifier;
- a command that runs from a clean checkout;
- a failure mode documented in the page that cites it.
