# Contributing to Epsilon-Hollow

## Prerequisites

- **Rust 1.85+** (see `rust-toolchain.toml`)
- For `aether-kernel`: nightly Rust and a bare-metal target

## Before Submitting

Run all three checks from the repo root:

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

All CI jobs must pass before a PR is merged.

## Where to Contribute

- **DSL / language work** -- the Aether-Lang crates:
  `kernel/aether/Aether-Lang/crates/`

- **Core math (topology, manifolds, PD control, teleportation)** --
  `kernel/epsilon/epsilon/crates/`

- **I/O super-kernel and benchmarks** --
  `kernel/aether/aether-link/`

## Notes

- `aether-kernel` is excluded from the default workspace because it
  requires a nightly toolchain and a bare-metal target. Build it
  separately with `cargo +nightly build -p aether-kernel`.
- Keep commits focused. One logical change per commit.
- Add or update tests for any behavioural change.
- Benchmark-sensitive changes should include a local Criterion run
  showing before/after numbers (see `BENCHMARKS.md`).
