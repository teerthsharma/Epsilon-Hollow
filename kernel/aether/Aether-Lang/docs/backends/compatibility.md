# Backend Compatibility

Compatibility means a reader can tell what code exists, what runtime gate must
pass, what fallback applies, and what verification supports the claim.

| Backend | Status | Runtime gate | Fallback | Verification | Claim boundary |
| --- | --- | --- | --- | --- | --- |
| Rust language core | Active | Rust toolchain and `aether-lang` crate build | None for parser/interpreter tests | `cargo test --manifest-path crates/aether-lang/Cargo.toml` | Lexer, parser, interpreter, AST, type-checking scaffolding, bytecode/VM subset |
| Rust kernel-adjacent crate | Active | Rust toolchain plus `x86_64-unknown-none` target | Host-only language tests | `cargo build --manifest-path crates/aether-kernel/Cargo.toml --target x86_64-unknown-none` | Sparse scheduler, loader checks, boot topology structs, serial/allocator scaffolding |
| `aegis-core` experiments | Active experimental | Rust toolchain and crate build | Language core remains usable | `cargo test --manifest-path crates/aegis-core/Cargo.toml` | Memory and autograd experiments only |
| CLI front ends | Partial | Build the excluded CLI crate explicitly | Direct crate tests and examples | `cargo run --manifest-path crates/aether-cli/Cargo.toml -- --help` after dependency checks | Host-side CLI only |
| Seal OS app host | Repository-level active path | Epsilon Hollow boot/proof tooling and runtime marker | Host-side interpreter tests | Use the root Seal OS verifier that checks the Aether runtime marker | OS integration only for the checked boot/app path |
| Docker image | Roadmap/unverified in this nested checkout | Published image and matching source revision | Local cargo commands | Image digest plus smoke command | Distribution convenience, not runtime correctness |
| GPU/LLM acceleration | Roadmap/prototype | Hardware gate, model artifact, correctness metric, baseline | CPU or host prototype path | Benchmark artifact with environment and baseline | No speedup claim without artifact |

## Runtime Gate Policy

Importing or building the language core should not imply that optional OS,
Docker, GPU, or LLM paths are available. Optional paths require explicit
selection and their own failure reporting.

## Compatibility Promise

Active means implemented and testable under a declared gate. It does not mean
production support, all-machine availability, or broad acceleration. Stronger
claims belong in this table only after the verification column can prove them.
