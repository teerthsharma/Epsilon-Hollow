# Contributing to AETHER-Link

First off, thank you for considering contributing to AETHER-Link! 🚀

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the existing issues. When creating a bug report, include:

- **Clear title** describing the issue
- **Steps to reproduce** the behavior
- **Expected behavior** vs actual behavior
- **System information** (OS, Rust version, CPU)
- **Benchmark output** if performance-related

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. Include:

- **Use case** - Why is this enhancement needed?
- **Proposed solution** - How should it work?
- **Alternatives** - Other solutions you've considered

### Pull Requests

1. **Fork the repo** and create your branch from `main`
2. **Add tests** for any new functionality
3. **Run the test suite**: `cargo test`
4. **Run clippy**: `cargo clippy --all-targets`
5. **Format code**: `cargo fmt`
6. **Update documentation** if needed
7. **Write a clear PR description**

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/aether-link
cd aether-link

# Build
cargo build

# Run tests
cargo test

# Run benchmarks
cargo bench

# Check formatting
cargo fmt --check

# Run linter
cargo clippy --all-targets --all-features
```

## Code Style

- Follow Rust standard conventions
- Use `rustfmt` for formatting
- Keep functions small and focused
- Document public APIs with doc comments
- Use `#[inline]` judiciously for hot paths

### Performance Guidelines

AETHER-Link is performance-critical. When contributing:

- **Benchmark your changes**: Use `cargo bench` before and after
- **Avoid allocations** in hot paths
- **Prefer `#[inline(always)]`** for sub-100ns functions
- **Test on release builds**: `cargo run --release`

## Commit Messages

Use conventional commits:

```
feat: add new telemetry feature
fix: correct epsilon clamping bug
perf: optimize fast_atan by 15%
docs: update HFT usage examples
test: add streaming workload tests
```

## License

By contributing, you agree that your contributions will be licensed under the same MIT/Apache-2.0 dual license as the project.

---

Questions? Open an issue or reach out!
