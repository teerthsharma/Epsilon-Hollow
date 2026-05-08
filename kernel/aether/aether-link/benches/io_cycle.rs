//! Benchmarks for `AetherLinkKernel::process_io_cycle`.
//!
//! The crate documentation claims `~18 ns per cycle`. This bench makes that
//! claim reproducible and shows the runtime is approximately constant in the
//! length of the LBA stream.
//!
//! Run with:
//! `cargo bench --manifest-path kernel/aether/aether-link/Cargo.toml`

use aether_link::AetherLinkKernel;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

/// Build a deterministic LBA stream of length `n` for benching.
fn make_stream(n: usize) -> Vec<u64> {
    // Mix of sequential and small jumps — exercises the telemetry path
    // without being trivially constant-foldable.
    (0..n as u64)
        .map(|i| 100u64 + i * 3 + (i % 5) * 7)
        .collect()
}

fn bench_io_cycle(c: &mut Criterion, name: &str, n: usize) {
    let stream = make_stream(n);
    c.bench_function(name, |b| {
        // Reconstruct kernel inside the iter so accumulated state from prior
        // iterations does not skew the measurement. `new_gaming` is a
        // zero-arg public constructor (see src/lib.rs:130).
        b.iter_batched(
            AetherLinkKernel::new_gaming,
            |mut kernel| {
                let result = kernel.process_io_cycle(black_box(&stream));
                black_box(result);
            },
            criterion::BatchSize::SmallInput,
        );
    });
}

fn io_cycle_8_lbas(c: &mut Criterion) {
    bench_io_cycle(c, "io_cycle_8_lbas", 8);
}

fn io_cycle_16_lbas(c: &mut Criterion) {
    bench_io_cycle(c, "io_cycle_16_lbas", 16);
}

fn io_cycle_64_lbas(c: &mut Criterion) {
    bench_io_cycle(c, "io_cycle_64_lbas", 64);
}

criterion_group!(benches, io_cycle_8_lbas, io_cycle_16_lbas, io_cycle_64_lbas);
criterion_main!(benches);
