//! The `aether-core` copy of `great_circle_distance`. Fixed in iteration 13
//! alongside the `aether-verified` copy, but only that one was tested — the
//! mutation gate caught the gap by reverting this file and seeing nothing fail.
use aether_core::geodesic_consolidation::great_circle_distance;

const PI: f64 = core::f64::consts::PI;

/// Ground truth from the dot product of the unit vectors the index builds:
/// `[sin t cos p, sin t sin p, cos t]`, the colatitude convention.
fn colatitude_truth(a: (f64, f64), b: (f64, f64)) -> f64 {
    let c = a.0.sin() * b.0.sin() * (a.1 - b.1).cos() + a.0.cos() * b.0.cos();
    c.clamp(-1.0, 1.0).acos()
}

#[test]
fn equator_quarter_turn_is_not_zero() {
    // The witness that killed the latitude form: colatitude pi/2 is the
    // equator, longitudes 0 and pi/2 are 90 degrees apart.
    let (a, b) = ((PI / 2.0, 0.0), (PI / 2.0, PI / 2.0));
    let got = great_circle_distance(a, b);
    println!("equator, 90 deg apart -> {got:.6} (want {:.6})", PI / 2.0);
    assert!((got - PI / 2.0).abs() < 1e-9,
        "returned {got} for two points a quarter turn apart");
}

#[test]
fn agrees_with_the_colatitude_convention_on_random_pairs() {
    let mut s: u64 = 0xA5A5_1234_DEAD_BEEF;
    let mut u = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; ((s >> 11) as f64) / ((1u64 << 53) as f64) };
    let mut worst = 0.0f64;
    for _ in 0..2000 {
        let a = (PI * u(), 2.0 * PI * u());
        let b = (PI * u(), 2.0 * PI * u());
        worst = worst.max((great_circle_distance(a, b) - colatitude_truth(a, b)).abs());
    }
    println!("worst disagreement over 2000 pairs: {worst:.6}");
    assert!(worst < 1e-9, "disagrees with the index convention by up to {worst} rad");
}

#[test]
fn metric_basics_hold() {
    let mut s: u64 = 0x0FF1CE_5EED;
    let mut u = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; ((s >> 11) as f64) / ((1u64 << 53) as f64) };
    for _ in 0..500 {
        let a = (PI * u(), 2.0 * PI * u());
        let b = (PI * u(), 2.0 * PI * u());
        let d = great_circle_distance(a, b);
        assert!(great_circle_distance(a, a).abs() < 1e-15, "d(p,p) must be 0");
        assert!((d - great_circle_distance(b, a)).abs() < 1e-12, "must be symmetric");
        assert!((0.0..=PI + 1e-9).contains(&d), "distance {d} outside [0, pi]");
    }
}

#[test]
fn small_separations_survive_the_regime_a_separation_check_operates_in() {
    // `verify_separation` compares against `theta_min - 1e-6`, so accuracy at
    // small separations is exactly what matters. The acos-of-dot-product form
    // returned 9.998224e-7 for a true 1e-6 (1.8e-4 relative error) and exactly
    // 0.0 for a true 1e-8. Haversine holds both.
    let theta = 1.0f64;
    for &sep in &[1e-2f64, 1e-4, 1e-6, 1e-8] {
        let d = great_circle_distance((theta, 0.0), (theta + sep, 0.0));
        let rel = (d - sep).abs() / sep;
        println!("true {sep:.0e} -> {d:.6e}  relative error {rel:.2e}");
        assert!(rel < 1e-6,
            "separation {sep:.0e} came back as {d:.6e}, relative error {rel:.2e}");
        assert!(d > 0.0, "two distinct points {sep:.0e} apart reported as coincident");
    }
}
