//! `verify_shape` rejects every input longer than 2560 bytes regardless of
//! content, because `density = beta_0 / len` cannot reach `DENSITY_MIN` there.
//!
//! beta_0 counts clusters of distinct byte VALUES, so it is bounded by 256 for
//! every input, while `len` is unbounded. The ratio has a ceiling of `256/len`,
//! which falls below 0.1 once `len > 2560`. That is a guaranteed wrong answer,
//! not a threshold that needs tuning.
use aether_core::{verify_shape, VerifyResult};

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

/// A 32-byte x86 prologue, repeated. Content that PASSES at short length.
fn prologue(reps: usize) -> Vec<u8> {
    let unit: [u8; 32] = [
        0x55, 0x48, 0x89, 0xe5, 0x48, 0x83, 0xec, 0x20, 0x89, 0x7d, 0xec, 0x89, 0x75, 0xe8, 0x48,
        0x89, 0x55, 0xe0, 0x48, 0x89, 0x4d, 0xd8, 0x44, 0x89, 0x45, 0xd4, 0x44, 0x89, 0x4d, 0xd0,
        0x8b, 0x45,
    ];
    unit.iter().copied().cycle().take(32 * reps).collect()
}

#[test]
fn long_input_is_not_reported_as_invalid_density() {
    // The same 32 bytes that pass on their own must not become "invalid" purely
    // by being repeated past 2560. Repetition adds no new byte values, so
    // beta_0 is unchanged; only `len` grows.
    let short = prologue(1);
    let long = prologue(200); // 6400 bytes, same value set
    println!("32 bytes   -> {:?}", verify_shape(&short));
    println!("6400 bytes -> {:?}", verify_shape(&long));
    assert!(
        !matches!(verify_shape(&long), VerifyResult::InvalidDensity { .. }),
        "the same byte values, repeated, became InvalidDensity purely by length"
    );
}

#[test]
fn the_gate_declines_rather_than_rejecting_when_passing_is_impossible() {
    for &n in &[4096usize, 8192, 65536] {
        let r = verify_shape(&xs(7, n));
        println!("len {n:>6} -> {r:?}");
        assert!(
            matches!(r, VerifyResult::LengthOutOfRange { .. }),
            "len {n}: expected the gate to decline, got {r:?}"
        );
    }
}

#[test]
fn short_inputs_still_get_a_real_verdict() {
    // The fix must not turn the gate off for lengths where it can work.
    let r = verify_shape(&prologue(1));
    println!("32-byte prologue -> {r:?}");
    assert!(
        !matches!(r, VerifyResult::LengthOutOfRange { .. }),
        "a 32-byte input must still be assessed, got {r:?}"
    );
}

#[test]
fn declining_is_not_passing() {
    // The one way this change could be wrong: if declining to assess were
    // treated as approval, the gate would silently become permissive on exactly
    // the inputs it can no longer judge.
    use aether_core::topology::is_shape_valid;
    for &n in &[4096usize, 8192, 65536] {
        let d = xs(11, n);
        assert!(matches!(
            verify_shape(&d),
            VerifyResult::LengthOutOfRange { .. }
        ));
        assert!(
            !is_shape_valid(&d),
            "len {n}: the gate declined to assess and is_shape_valid returned true"
        );
    }
    // And a long run of identical bytes, the case a permissive gate would wave
    // through most readily.
    let sled = vec![0x90u8; 8192];
    assert!(
        !is_shape_valid(&sled),
        "an 8192-byte NOP sled passed the gate"
    );
}
