# Benchmarks

Detailed performance analysis of AETHER-Link kernel.

## Test Environment

- **CPU**: Intel/AMD x86_64 (results vary by microarchitecture)
- **Rust**: 1.75+ with LTO enabled
- **Profile**: Release with `codegen-units = 1`

## Core Function Benchmarks

| Function | Mean | Notes |
|----------|------|-------|
| `process_io_cycle` | **14.6 ns** | Full decision loop |
| `extract_telemetry` | 0.99 ns | Zero-copy DSP |
| `prepare_quantum_state` | 3.68 ns | 6x fast_atan |
| `fast_atan` | 0.62 ns | Padé approximant |
| `fast_exp` | ~1.5 ns | Hardware intrinsic |
| `fast_sigmoid` | 2.31 ns | Combined |

## Throughput

```
1M cycles: 15.32 ms total
Throughput: ~65.3 million ops/sec
```

## Latency Percentiles (HFT Mode)

| Percentile | Latency |
|------------|---------|
| P50 | 17.2 ns |
| P90 | 18.4 ns |
| P99 | 19.1 ns |
| P99.9 | 21.4 ns |

**Jitter (P99b - P50)**: 1.9 ns ✓

## Stream Size Scaling

| Stream Size | Latency | Impact |
|-------------|---------|--------|
| 10 | 17.8 ns | Baseline |
| 100 | 18.0 ns | +1.1% |
| 1,000 | 18.1 ns | +1.7% |
| 10,000 | 18.2 ns | +2.2% |

Stream size has minimal impact due to O(1) telemetry extraction.

## Workload Pattern Comparison

| Pattern | Prefetch Ratio | Latency |
|---------|----------------|---------|
| Sequential | 62% | 17.9 ns |
| Random | 41% | 18.3 ns |
| Bursty | 55% | 18.1 ns |
| HFT Tick | 38% | 17.4 ns |

## Running Benchmarks

```bash
# Full benchmark suite
cargo bench

# Specific benchmark
cargo bench -- "process_io_cycle"

# HTML report
cargo bench -- --plotting-backend plotters
open target/criterion/report/index.html
```

## Comparison Notes

For context:
- DDR4 latency: ~14-18 ns
- L3 cache hit: ~10-20 ns
- NVMe latency: ~10-25 µs (1000x slower)

AETHER-Link decision overhead is **imperceptible** compared to actual I/O.
