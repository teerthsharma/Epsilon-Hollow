use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use aether_link::AetherLinkKernel;

fn bench_core_functions(c: &mut Criterion) {
    let mut kernel = AetherLinkKernel::new(0.5, 0.1, [0.1, 0.2, 0.3], 0.05);
    let lba_stream = vec![100u64, 101, 102, 105, 110, 200, 205];

    let mut group = c.benchmark_group("AETHER Core");
    group.throughput(Throughput::Elements(1));

    group.bench_function("process_io_cycle", |b| {
        b.iter(|| kernel.process_io_cycle(black_box(&lba_stream)))
    });

    group.bench_function("extract_telemetry", |b| {
        b.iter(|| kernel.extract_telemetry(black_box(&lba_stream)))
    });

    let features = kernel.extract_telemetry(&lba_stream);
    group.bench_function("prepare_quantum_state", |b| {
        b.iter(|| kernel.prepare_quantum_state(black_box(features)))
    });

    group.finish();
}

fn bench_presets(c: &mut Criterion) {
    let lba_stream: Vec<u64> = (0..100).collect();

    let mut group = c.benchmark_group("AETHER Presets");
    group.throughput(Throughput::Elements(1));

    let mut hft_kernel = AetherLinkKernel::new_hft();
    group.bench_function("HFT Cycle", |b| {
        b.iter(|| hft_kernel.process_io_cycle(black_box(&lba_stream)))
    });

    let mut gaming_kernel = AetherLinkKernel::new_gaming();
    group.bench_function("Gaming Cycle", |b| {
        b.iter(|| gaming_kernel.process_io_cycle(black_box(&lba_stream)))
    });

    group.finish();
}

fn bench_stream_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("Stream Size Scaling");

    for size in [10, 100, 1000, 10000].iter() {
        let lba_stream: Vec<u64> = (0..*size).collect();
        let mut kernel = AetherLinkKernel::default();

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &lba_stream,
            |b, stream| {
                b.iter(|| kernel.process_io_cycle(black_box(stream)))
            },
        );
    }

    group.finish();
}

fn bench_fast_math(c: &mut Criterion) {
    use aether_link::{fast_atan, fast_exp, fast_sigmoid};

    let mut group = c.benchmark_group("Fast Math");
    group.throughput(Throughput::Elements(1));

    group.bench_function("fast_atan", |b| {
        b.iter(|| fast_atan(black_box(1.5)))
    });

    group.bench_function("fast_exp", |b| {
        b.iter(|| fast_exp(black_box(-0.5)))
    });

    group.bench_function("fast_sigmoid", |b| {
        b.iter(|| fast_sigmoid(black_box(0.3)))
    });

    group.finish();
}

fn bench_throughput(c: &mut Criterion) {
    let mut kernel = AetherLinkKernel::new_hft();
    let lba_stream: Vec<u64> = (0..50).collect();

    let mut group = c.benchmark_group("Throughput");
    group.throughput(Throughput::Elements(1_000_000));

    group.bench_function("1M cycles", |b| {
        b.iter(|| {
            for _ in 0..1_000_000 {
                black_box(kernel.process_io_cycle(&lba_stream));
            }
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_core_functions,
    bench_presets,
    bench_stream_sizes,
    bench_fast_math,
    bench_throughput,
);

criterion_main!(benches);
