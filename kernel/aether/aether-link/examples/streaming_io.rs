//! Streaming I/O simulation for AETHER-Link.
//!
//! This example simulates real-time I/O processing with different
//! workload patterns to demonstrate adaptive behavior.
//!
//! Run with: cargo run --example streaming_io

use aether_link::AetherLinkKernel;
use std::time::Instant;

/// Simulates different I/O workload patterns
enum WorkloadPattern {
    Sequential,
    Random,
    Bursty,
    HftTick,
}

impl WorkloadPattern {
    fn generate(&self, base: u64, count: usize) -> Vec<u64> {
        match self {
            WorkloadPattern::Sequential => (base..base + count as u64).collect(),
            WorkloadPattern::Random => {
                // Pseudo-random using simple LCG
                let mut rng = base;
                (0..count)
                    .map(|_| {
                        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                        rng % 100000
                    })
                    .collect()
            }
            WorkloadPattern::Bursty => {
                // Bursts of sequential with gaps
                let mut result = Vec::with_capacity(count);
                let mut pos = base;
                for i in 0..count {
                    if i % 5 == 0 {
                        pos += 1000; // Jump
                    }
                    result.push(pos);
                    pos += 1;
                }
                result
            }
            WorkloadPattern::HftTick => {
                // High-frequency tick data pattern
                // Small sequential bursts with occasional jumps
                let mut result = Vec::with_capacity(count);
                let mut pos = base;
                for i in 0..count {
                    if i % 10 == 0 {
                        pos += 64; // Cache line boundary
                    }
                    result.push(pos);
                    pos += 1;
                }
                result
            }
        }
    }

    fn name(&self) -> &'static str {
        match self {
            WorkloadPattern::Sequential => "Sequential",
            WorkloadPattern::Random => "Random",
            WorkloadPattern::Bursty => "Bursty",
            WorkloadPattern::HftTick => "HFT Tick",
        }
    }
}

fn run_workload(pattern: &WorkloadPattern, cycles: usize) {
    let mut kernel = match pattern {
        WorkloadPattern::HftTick => AetherLinkKernel::new_hft(),
        _ => AetherLinkKernel::default(),
    };

    let start = Instant::now();

    for i in 0..cycles {
        let stream = pattern.generate(i as u64 * 100, 20);
        kernel.process_io_cycle(&stream);
    }

    let elapsed = start.elapsed();
    let ns_per_cycle = elapsed.as_nanos() as f64 / cycles as f64;

    println!(
        "   {:12} │ {:>8} │ {:>8} │ {:>6.1}% │ {:>7.1} ns │ {:>6.1} Mops/s",
        pattern.name(),
        kernel.cycles,
        kernel.prefetches,
        kernel.prefetch_ratio() * 100.0,
        ns_per_cycle,
        1000.0 / ns_per_cycle,
    );
}

fn main() {
    println!();
    println!("╔═══════════════════════════════════════════════════════════════════════╗");
    println!("║        🌊 AETHER-Link: Streaming I/O Simulation 🌊                    ║");
    println!("║                                                                       ║");
    println!("║  Simulates real-time I/O processing with various workload patterns   ║");
    println!("║  to demonstrate adaptive prefetching behavior.                        ║");
    println!("╚═══════════════════════════════════════════════════════════════════════╝");
    println!();

    let cycles = 100_000;
    println!("📊 Running {} I/O cycles per workload...", cycles);
    println!();

    println!("┌──────────────┬──────────┬──────────┬─────────┬─────────────┬────────────┐");
    println!("│   Workload   │  Cycles  │ Prefetch │  Ratio  │  Latency    │ Throughput │");
    println!("├──────────────┼──────────┼──────────┼─────────┼─────────────┼────────────┤");

    run_workload(&WorkloadPattern::Sequential, cycles);
    run_workload(&WorkloadPattern::Random, cycles);
    run_workload(&WorkloadPattern::Bursty, cycles);
    run_workload(&WorkloadPattern::HftTick, cycles);

    println!("└──────────────┴──────────┴──────────┴─────────┴─────────────┴────────────┘");
    println!();

    // HFT-specific demonstration
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("🏦 HFT Mode: Latency-Critical Analysis");
    println!();

    let mut hft_kernel = AetherLinkKernel::new_hft();
    let tick_stream: Vec<u64> = (0..50).collect();

    // Measure single-cycle latency
    let iterations = 1_000_000;
    let start = Instant::now();
    for _ in 0..iterations {
        std::hint::black_box(hft_kernel.process_io_cycle(&tick_stream));
    }
    let elapsed = start.elapsed();
    let ns_per_op = elapsed.as_nanos() as f64 / iterations as f64;

    println!("   Single-cycle latency: {:.1} ns", ns_per_op);
    println!("   Throughput: {:.1} million decisions/sec", 1000.0 / ns_per_op);
    println!("   Final ε: {:.4}", hft_kernel.epsilon);
    println!("   Final φ: {:.4} rad", hft_kernel.phi);
    println!();

    // Consistency check
    println!("📐 Latency Consistency (1000 samples):");
    let mut latencies = Vec::with_capacity(1000);
    for _ in 0..1000 {
        let start = Instant::now();
        for _ in 0..1000 {
            std::hint::black_box(hft_kernel.process_io_cycle(&tick_stream));
        }
        latencies.push(start.elapsed().as_nanos() as f64 / 1000.0);
    }

    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = latencies[500];
    let p99 = latencies[990];
    let p999 = latencies[999];

    println!("   P50:  {:.1} ns", p50);
    println!("   P99:  {:.1} ns", p99);
    println!("   P999: {:.1} ns", p999);
    println!("   Jitter (P99-P50): {:.1} ns", p99 - p50);
    println!();

    println!("✅ Streaming simulation complete!");
    println!("   For detailed benchmarks, run: cargo bench");
}
