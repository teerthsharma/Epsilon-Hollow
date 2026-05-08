# aether-link

Ultra-fast I/O super-kernel with adaptive POVM-based prefetching.

See `src/lib.rs` for the full API and theory of operation.

# Benchmarks

The kernel documentation claims `~18 ns per cycle` for
`AetherLinkKernel::process_io_cycle`. The `io_cycle` bench makes that claim
reproducible.

Run:

```
cargo bench --manifest-path kernel/aether/aether-link/Cargo.toml
```

This runs three Criterion benchmarks — `io_cycle_8_lbas`, `io_cycle_16_lbas`,
and `io_cycle_64_lbas` — exercising `process_io_cycle` against LBA streams of
8, 16, and 64 entries respectively.

## Reading the results

Absolute nanosecond numbers depend on the host CPU, frequency scaling, and
build flags, so the exact figure will vary across machines. The
**load-bearing claim** is the *shape* of the result: per-cycle latency should
be approximately constant across the three stream sizes. If runtime grows
linearly (or worse) with LBA count, the constant-time invariant is broken
and the `~18 ns/cycle` headline is no longer meaningful.
