# Benchmarks

Performance benchmarks live alongside each crate, driven by [Criterion].
This document covers how to run them, what numbers to expect, and how to
read regression output from CI.

## Running locally

The primary benchmark is the `io_cycle` suite for `aether-link`:

```bash
cargo bench --bench io_cycle --manifest-path kernel/aether/aether-link/Cargo.toml
```

Run a single bench by id (substring match):

```bash
cargo bench --bench io_cycle -- io_cycle_8_lbas
```

Compile benches across the workspace without running them (mirrors CI's
`bench-compile` job):

```bash
cargo bench --workspace --no-run
```

## Expected ranges

`io_cycle_8_lbas` measures one full `process_io_cycle` call against an
8-element LBA stream. Quoting the crate docs (`kernel/aether/aether-link/src/lib.rs`):

| Component  | Latency  |
|------------|----------|
| Full Cycle | ~18 ns   |
| Telemetry  | ~1.4 ns  |
| State Prep | ~47 ns   |

On a recent x86_64 desktop you should see `io_cycle_8_lbas` land in the
**15–25 ns** band. CI runners are noisier; the regression gate uses a
**120 ns** ceiling (see `bench-regression` in `.github/workflows/ci.yml`)
to keep the signal real without flapping on shared-hardware variance.

## Interpreting CI output

The `bench-regression` CI job runs Criterion in `--output-format bencher`
mode and parses the median ns/iter for `io_cycle_8_lbas`. The full
`bench.txt` is uploaded as the `io-cycle-bench` artifact on every run.
A regression looks like:

```
REGRESSION: io_cycle_8_lbas = 187 ns/iter > threshold 120 ns/iter
```

When investigating a regression, download the artifact and compare the
median + variance against the previous run. Numbers near the threshold
on the same commit usually point at runner contention rather than a real
slowdown — re-run the workflow once before assuming a real regression.

## Adding a new benchmark

1. Add `[[bench]]` entry in the crate's `Cargo.toml`.
2. Use `criterion::Criterion::bench_function` for single-input benches.
3. If the bench should be regression-gated, extend the `bench-regression`
   job to parse the new id and pick a threshold with ~5x headroom over
   the observed local median.

## End-to-end demo

The executable harness specified in `docs/research/MOTHER_OF_ALL_DOCS.md` (§3.5)
ships as a Rust example that exercises every theorem (T1-T10) through the
`aether_verified` kernel API in a single run.

Run it with:

```bash
cargo run --example world_model_demo \
  --manifest-path kernel/aether/aether-link/Cargo.toml
```

Sample (truncated) output:

```
============================================================
   Epsilon-Hollow Theorem Verification Run
============================================================

[T1 TSS]   P_max(theta=0.5) = 65.350
           theta_min(eps=0.1) = 0.1000 rad
           packing OK: true
[T2 SCM]   trajectory: 3.500 2.450 1.715 1.200 0.840
           rho = 0.700, half-life ~ 1.94 steps
[T3 GMC]   max merges (P=1024) = 1023
           ΔH on merge = -0.09710 bits  (nonincreasing: true)
[T4 PDG]   eps after 50 steps = 0.19452 (target ratio 0.20)
           gain margin refined: true
[T5 HCS]   hyp/euc distortion = 0.0016 / 0.0975
           separation ratio = 6.2383e1, holds: true
[T6 RGCS]  Δ_T(p=8) = 3.536e-4
           NVLink sync cost (70B params): 0.544 s
[T7 PHKP]  E[T] base = 2625.0 ns, sparse(s=0.7) = 787.5 ns
           KV cache (8k ctx, 32L) = 1.000 GB
[T8 TEB]   k_B·T·ln2 @ 350K = 3.349e-21 J/bit
           E_min/update (1M hot, 16b) = 5.359e-14 J
           H100 700W headroom factor  = 1.306e7
[T9 CMA]   single-stage eps_align = 0.3122
           5-stage transitive eps = 0.2300
[T10 WPHB] horizon = 1.143e15 tokens (P=70000000000, d=8192, h=5.0 b/tok)

============================================================
All 10 theorems exercised. Bounds satisfied: 10/10.
============================================================
```

The demo exits 0 on a successful run; all 10 theorems must report
their bounds satisfied for the count to read `10/10`.

[Criterion]: https://github.com/bheisler/criterion.rs
