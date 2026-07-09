# Evidence Policy

Aether documentation follows the same evidence posture as the Topological ML
Toolkit docs: active behavior is tied to tests or artifacts, while broader
claims stay labeled as roadmap.

## Claim Classes

| Claim class | Allowed wording | Required evidence |
| --- | --- | --- |
| Active API | "The parser supports..." | Rust test, command, or source-level API |
| Runtime integration | "Seal OS runs an Aether runtime probe..." | Boot log marker or verifier command |
| Benchmark result | "This workload measured..." | Raw artifact plus environment and baseline |
| Optional backend | "This backend is available when..." | Runtime gate, fallback, and verification command |
| Roadmap | "The design targets..." | Clear roadmap label; no current-capability wording |

## Documentation Rules

- Do not use unverifiable superlatives.
- Do not publish timing tables with `TBD` values as if they are benchmark
  results.
- Do not convert an example script into a production support claim.
- Do not claim hardware acceleration when the active path is a CPU fallback or a
  scaffold.
- Keep host-side examples separate from kernel-runtime claims.

## Current Evidence Sources

- `cargo test` for Rust crate behavior.
- Source-level checks in `crates/aether-lang`.
- Source-level checks in `crates/aether-kernel`.
- Epsilon Hollow boot/proof tooling at the repository root when a claim crosses
  into Seal OS runtime behavior.

Evidence must match the scope of the claim. A parser unit test does not prove
runtime performance. A host-side example does not prove kernel integration.
