use aether_core::{compute_shape, verify_shape};

fn xs(seed: u64, n: usize) -> Vec<u8> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s % 256) as u8
        })
        .collect()
}

#[test]
fn density_after_the_fix() {
    println!(
        "{:<34} {:>5} {:>6} {:>9}  {:?}",
        "input", "len", "beta_0", "density", "verdict"
    );
    let x86: Vec<u8> = vec![
        0x55, 0x48, 0x89, 0xe5, 0x48, 0x83, 0xec, 0x20, 0x89, 0x7d, 0xec, 0x89, 0x75, 0xe8, 0x48,
        0x89, 0x55, 0xe0, 0x48, 0x89, 0x4d, 0xd8, 0x44, 0x89, 0x45, 0xd4, 0x44, 0x89, 0x4d, 0xd0,
        0x8b, 0x45,
    ];
    let ascii: Vec<u8> =
        b"the quick brown fox jumps over the lazy dog and then some more text".to_vec();
    let cases: Vec<(&str, Vec<u8>)> = vec![
        ("NOP sled [0x90;64]", vec![0x90u8; 64]),
        ("x86 prologue (32B)", x86.clone()),
        ("ASCII text", ascii),
        ("random 64B", xs(1, 64)),
        ("random 256B", xs(2, 256)),
        ("random 1024B", xs(3, 1024)),
        ("random 4096B", xs(4, 4096)),
        (
            "two clusters 0/200 (64B)",
            (0..64)
                .map(|i| {
                    if i < 32 {
                        (i % 8) as u8
                    } else {
                        200 + (i % 8) as u8
                    }
                })
                .collect(),
        ),
    ];
    for (name, d) in &cases {
        let sh = compute_shape(d);
        println!(
            "{:<34} {:>5} {:>6} {:>9.5}  {:?}",
            name,
            d.len(),
            sh.betti_0,
            sh.density,
            verify_shape(d)
        );
    }
    println!();
    println!("STRUCTURAL: beta_0 <= 256 always (only 256 distinct byte values).");
    println!("So density = beta_0/len <= 256/len. Max achievable density by length:");
    for n in [64usize, 128, 256, 512, 1024, 4096] {
        println!(
            "  len={:5}  max possible density = {:.5}  (DENSITY_MIN=0.1 reachable? {})",
            n,
            256.0f64.min(n as f64) / (n as f64),
            if 256.0f64.min(n as f64) / (n as f64) >= 0.1 {
                "yes"
            } else {
                "NO - always rejected"
            }
        );
    }
}
