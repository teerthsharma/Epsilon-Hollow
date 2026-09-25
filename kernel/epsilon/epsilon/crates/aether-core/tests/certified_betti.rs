//! Host tests for `certified_betti::certified_beta0`.
//!
//! The contract: a `Certified` value is one that no threshold inside the
//! ratio-wide band centred on the scale could change; anything else is
//! `Refused` with the ambiguous pair named.
use aether_core::certified_betti::{certified_beta0, Beta0};

const RHO: f64 = 10.0;

/// Scale-safe Euclidean distance, `m * sqrt(sum((d / m)^2))` with
/// `m = max |d|`, so neither a `1e-170` nor a `1e170` difference underflows
/// or overflows when squared. It performs the same operations in the same
/// order as the library's norm, and IEEE `sqrt` is correctly rounded in
/// both, so a named height compares bit for bit at every opt-level.
fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    let d = [
        (a[0] - b[0]).abs(),
        (a[1] - b[1]).abs(),
        (a[2] - b[2]).abs(),
    ];
    let m = d[0].max(d[1]).max(d[2]);
    if m == 0.0 || !m.is_finite() {
        return m;
    }
    let (x, y, z) = (d[0] / m, d[1] / m, d[2] / m);
    m * (x * x + y * y + z * z).sqrt()
}

/// Independent reference: all-pairs union-find at a fixed threshold.
fn brute_beta0(points: &[[f64; 3]], t: f64, strict: bool) -> u32 {
    let n = points.len();
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(p: &mut [usize], mut x: usize) -> usize {
        while p[x] != x {
            x = p[x];
        }
        x
    }
    for i in 0..n {
        for j in (i + 1)..n {
            let d = dist(points[i], points[j]);
            if (strict && d < t) || (!strict && d <= t) {
                let (a, b) = (find(&mut parent, i), find(&mut parent, j));
                parent[a] = b;
            }
        }
    }
    (0..n).filter(|&i| find(&mut parent, i) == i).count() as u32
}

/// Seeded splitmix64 so every cloud is reproducible.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Clustered cloud: a few centres in a 10-cube, members jittered by a
/// per-cloud spread so some scales land in wide gaps and some do not.
fn cloud(rng: &mut Rng) -> Vec<[f64; 3]> {
    let k = 1 + (rng.next() * 5.0) as usize;
    let n = 1 + (rng.next() * 30.0) as usize;
    let spread = 10f64.powf(-6.0 + 6.0 * rng.next());
    let centres: Vec<[f64; 3]> = (0..k)
        .map(|_| [rng.next() * 10.0, rng.next() * 10.0, rng.next() * 10.0])
        .collect();
    (0..n)
        .map(|i| {
            let c = centres[i % k];
            [
                c[0] + spread * (rng.next() - 0.5),
                c[1] + spread * (rng.next() - 0.5),
                c[2] + spread * (rng.next() - 0.5),
            ]
        })
        .collect()
}

fn log_uniform_scale(rng: &mut Rng) -> f64 {
    10f64.powf(-5.0 + 7.0 * rng.next())
}

fn value(b: Beta0) -> Option<u32> {
    match b {
        Beta0::Certified { value, .. } => Some(value),
        Beta0::Refused { .. } => None,
    }
}

/// `certified_beta0`, with every certified answer checked against the
/// all-pairs reference so the invariance tests cannot pass vacuously.
fn checked(pts: &[[f64; 3]], s: f64) -> Beta0 {
    let b = certified_beta0(pts, s, RHO);
    if let Beta0::Certified { value, .. } = b {
        assert_eq!(
            value,
            brute_beta0(pts, s, true),
            "certified value disagrees with all-pairs"
        );
    }
    b
}

fn assert_refused_pair(b: Beta0, pts: &[[f64; 3]]) {
    match b {
        Beta0::Refused { i, j, height } => {
            assert!(i < j && j < pts.len(), "pair ({i},{j}) out of range");
            assert_eq!(
                height,
                dist(pts[i], pts[j]),
                "height is the named pair's distance"
            );
        }
        other => panic!("expected Refused, got {other:?}"),
    }
}

#[test]
fn chord_just_below_threshold_refuses() {
    let pts = [[0.0, 0.0, 0.0], [0.5 - 1e-9, 0.0, 0.0]];
    let b = certified_beta0(&pts, 0.5, RHO);
    assert_refused_pair(b, &pts);
}

#[test]
fn chord_just_above_threshold_refuses() {
    let pts = [[0.0, 0.0, 0.0], [0.5 + 1e-9, 0.0, 0.0]];
    let b = certified_beta0(&pts, 0.5, RHO);
    assert_refused_pair(b, &pts);
}

#[test]
fn two_tight_clusters_far_apart_certify_two() {
    let pts = [
        [0.0, 0.0, 0.0],
        [1e-4, 0.0, 0.0],
        [0.0, 1e-4, 0.0],
        [10.0, 0.0, 0.0],
        [10.0, 1e-4, 0.0],
    ];
    match certified_beta0(&pts, 0.5, RHO) {
        Beta0::Certified {
            value,
            gap_lo,
            gap_hi,
        } => {
            assert_eq!(value, 2);
            assert!(gap_lo <= 0.5 / RHO.sqrt() && gap_hi >= 0.5 * RHO.sqrt());
            assert!(gap_hi / gap_lo >= RHO);
        }
        other => panic!("expected Certified 2, got {other:?}"),
    }
}

#[test]
fn one_point_certifies_one() {
    assert_eq!(
        certified_beta0(&[[0.3, 0.4, 0.5]], 0.5, RHO),
        Beta0::Certified {
            value: 1,
            gap_lo: 0.0,
            gap_hi: f64::INFINITY
        }
    );
}

#[test]
fn empty_certifies_zero() {
    // No heights at all: the empty stretch is the whole half-line.
    let pts: [[f64; 3]; 0] = [];
    assert_eq!(
        certified_beta0(&pts, 0.5, RHO),
        Beta0::Certified {
            value: 0,
            gap_lo: 0.0,
            gap_hi: f64::INFINITY
        }
    );
}

#[test]
fn non_finite_coordinate_refuses() {
    let pts = [[0.0, 0.0, 0.0], [f64::NAN, 0.0, 0.0]];
    assert!(value(certified_beta0(&pts, 0.5, RHO)).is_none());
}

#[test]
fn permutation_invariant() {
    let mut rng = Rng(0x5EED_0001);
    for _ in 0..200 {
        let pts = cloud(&mut rng);
        let s = log_uniform_scale(&mut rng);
        let base = checked(&pts, s);
        let mut rev = pts.clone();
        rev.reverse();
        let mut shuf = pts.clone();
        for i in (1..shuf.len()).rev() {
            let j = (rng.next() * (i + 1) as f64) as usize;
            shuf.swap(i, j);
        }
        for p in [&rev, &shuf] {
            let b = checked(p, s);
            assert_eq!(value(b), value(base), "permutation changed the answer");
        }
    }
}

#[test]
fn isometry_invariant() {
    // Rotation about an oblique axis plus a translation.
    let (c, s) = (0.6f64, 0.8f64);
    let rot = |p: [f64; 3]| {
        let x = c * p[0] - s * p[1];
        let y = s * p[0] + c * p[1];
        let (y2, z2) = (c * y - s * p[2], s * y + c * p[2]);
        [x + 3.25, y2 - 1.5, z2 + 0.75]
    };
    let mut rng = Rng(0x5EED_0002);
    let mut certified = 0;
    for _ in 0..200 {
        let pts = cloud(&mut rng);
        let sc = log_uniform_scale(&mut rng);
        let a = checked(&pts, sc);
        let moved: Vec<[f64; 3]> = pts.iter().map(|&p| rot(p)).collect();
        let b = checked(&moved, sc);
        // Rounding in the rotation may move a height across the band edge,
        // turning a certificate into a refusal; it must never change a
        // certified value into a different certified value.
        if let (Some(x), Some(y)) = (value(a), value(b)) {
            assert_eq!(x, y, "isometry changed a certified value");
            certified += 1;
        }
    }
    assert!(
        certified > 50,
        "only {certified}/200 isometry pairs certified"
    );
}

#[test]
fn scale_equivariant() {
    let mut rng = Rng(0x5EED_0003);
    for _ in 0..200 {
        let pts = cloud(&mut rng);
        let s = log_uniform_scale(&mut rng);
        let base = checked(&pts, s);
        // Powers of two scale every coordinate and height exactly.
        for f in [0.125, 8.0, 1024.0] {
            let scaled: Vec<[f64; 3]> =
                pts.iter().map(|p| [p[0] * f, p[1] * f, p[2] * f]).collect();
            let b = checked(&scaled, s * f);
            assert_eq!(value(b), value(base), "scaling by {f} changed the answer");
        }
    }
}

#[test]
fn certified_agrees_with_all_pairs_union_find_across_the_band() {
    let mut rng = Rng(0x5EED_0004);
    let (mut certified, mut refused) = (0, 0);
    for _ in 0..500 {
        let pts = cloud(&mut rng);
        let s = log_uniform_scale(&mut rng);
        match certified_beta0(&pts, s, RHO) {
            Beta0::Certified {
                value,
                gap_lo,
                gap_hi,
            } => {
                certified += 1;
                assert!(gap_lo < s && s < gap_hi);
                // Every threshold in the certified band gives the same count,
                // under either comparison convention.
                for t in [s / RHO.sqrt(), s, s * RHO.sqrt()] {
                    assert_eq!(value, brute_beta0(&pts, t, true), "strict, t={t}");
                    assert_eq!(value, brute_beta0(&pts, t, false), "non-strict, t={t}");
                }
            }
            b @ Beta0::Refused { .. } => {
                refused += 1;
                assert_refused_pair(b, &pts);
            }
        }
    }
    assert!(
        certified > 100 && refused > 20,
        "certified {certified}, refused {refused}"
    );
}

#[test]
fn tiny_separation_does_not_underflow_into_a_merge() {
    // 1e-170 squared is 0 in f64; the true distance is far above the band
    // [3.2e-201, 3.2e-200], so the two points are two components.
    let pts = [[0.0, 0.0, 0.0], [1e-170, 0.0, 0.0]];
    assert_eq!(
        brute_beta0(&pts, 1e-200, true),
        2,
        "oracle must see the gap"
    );
    assert_eq!(
        certified_beta0(&pts, 1e-200, RHO),
        Beta0::Certified {
            value: 2,
            gap_lo: 0.0,
            gap_hi: 1e-170
        }
    );
}

#[test]
fn huge_separation_does_not_overflow_the_height() {
    // 1e170 squared is inf in f64. At scale 1e170 the edge sits at the band
    // centre, and the refusal must report its finite height.
    let pts = [[0.0, 0.0, 0.0], [1e170, 0.0, 0.0]];
    assert_eq!(
        certified_beta0(&pts, 1e170, RHO),
        Beta0::Refused {
            i: 0,
            j: 1,
            height: 1e170
        }
    );
    // 1e160 lies below the band around 1e170: one component, not a refusal.
    let pts = [[0.0, 0.0, 0.0], [1e160, 0.0, 0.0]];
    assert_eq!(
        certified_beta0(&pts, 1e170, RHO),
        Beta0::Certified {
            value: 1,
            gap_lo: 1e160,
            gap_hi: f64::INFINITY
        }
    );
}

#[test]
fn edge_exactly_at_a_band_end_refuses() {
    // Scale 2, ratio 4: the band is exactly [1, 4].
    for h in [1.0, 4.0] {
        assert_eq!(
            certified_beta0(&[[0.0], [h]], 2.0, 4.0),
            Beta0::Refused {
                i: 0,
                j: 1,
                height: h
            },
            "edge at band end {h}"
        );
    }
    // Just outside either end certifies.
    assert_eq!(
        certified_beta0(&[[0.0], [0.875]], 2.0, 4.0),
        Beta0::Certified {
            value: 1,
            gap_lo: 0.875,
            gap_hi: f64::INFINITY
        }
    );
    assert_eq!(
        certified_beta0(&[[0.0], [4.5]], 2.0, 4.0),
        Beta0::Certified {
            value: 2,
            gap_lo: 0.0,
            gap_hi: 4.5
        }
    );
}

#[test]
fn ratio_below_one_or_nan_is_treated_as_one() {
    for ratio in [0.25, 0.0, -3.0, f64::NAN] {
        // With the band collapsed onto the scale, 1.5 lies above it and 0.5 below.
        assert_eq!(
            certified_beta0(&[[0.0], [1.5]], 1.0, ratio),
            Beta0::Certified {
                value: 2,
                gap_lo: 0.0,
                gap_hi: 1.5
            },
            "ratio {ratio}"
        );
        assert_eq!(
            certified_beta0(&[[0.0], [0.5]], 1.0, ratio),
            Beta0::Certified {
                value: 1,
                gap_lo: 0.5,
                gap_hi: f64::INFINITY
            },
            "ratio {ratio}"
        );
        // A height equal to the scale still refuses.
        assert_eq!(
            certified_beta0(&[[0.0], [1.0]], 1.0, ratio),
            Beta0::Refused {
                i: 0,
                j: 1,
                height: 1.0
            },
            "ratio {ratio}"
        );
    }
}

#[test]
fn gap_bounds_are_the_nearest_heights_outside_the_band() {
    // Tree heights 0.125, 0.25, 7.625, 16; the band around 1 at ratio 4 is
    // [0.5, 2]. Below it the largest is 0.25; above it the smallest is 7.625.
    let pts = [[0.0], [0.125], [0.375], [8.0], [24.0]];
    assert_eq!(
        certified_beta0(&pts, 1.0, 4.0),
        Beta0::Certified {
            value: 3,
            gap_lo: 0.25,
            gap_hi: 7.625
        }
    );
    // The same heights found in another order give the same bounds.
    let pts = [[24.0], [0.375], [8.0], [0.0], [0.125]];
    assert_eq!(
        certified_beta0(&pts, 1.0, 4.0),
        Beta0::Certified {
            value: 3,
            gap_lo: 0.25,
            gap_hi: 7.625
        }
    );
}

#[test]
fn refusal_names_the_band_edge_nearest_the_scale() {
    // Band [0.25, 4] around 1. Prim finds heights 0.5, 1, 3 in that order,
    // all inside the band; the named edge is the middle one, |h - 1| = 0.
    assert_eq!(
        certified_beta0(&[[0.0], [0.5], [1.5], [4.5]], 1.0, 16.0),
        Beta0::Refused {
            i: 1,
            j: 2,
            height: 1.0
        }
    );
    // Heights 2.5 (found first) and 0.375: nearest by absolute difference is
    // 0.375 (0.625 < 1.5), although 2.5 is nearer by ratio (2.5 < 1 / 0.375).
    assert_eq!(
        certified_beta0(&[[2.875], [0.375], [0.0]], 1.0, 16.0),
        Beta0::Refused {
            i: 1,
            j: 2,
            height: 0.375
        }
    );
}

#[test]
fn non_finite_point_is_named_as_itself() {
    // An edge sits exactly at the scale, yet the input also holds a point with
    // no measurable distance. The refusal names that point as i == j, with a
    // NaN height, whether Prim meets it first (NaN) or last (infinity).
    for pts in [
        [
            [f64::NAN, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ],
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [f64::INFINITY, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ],
    ] {
        let p = pts
            .iter()
            .position(|q| q.iter().any(|c| !c.is_finite()))
            .unwrap();
        match certified_beta0(&pts, 0.5, RHO) {
            Beta0::Refused { i, j, height } => {
                assert_eq!((i, j), (p, p), "names the non-finite point");
                assert!(height.is_nan(), "height {height} is not NaN");
            }
            other => panic!("expected Refused, got {other:?}"),
        }
    }
    // One point has no pair to be ambiguous about.
    assert_eq!(
        certified_beta0(&[[f64::NAN, 0.0, 0.0]], 0.5, RHO),
        Beta0::Certified {
            value: 1,
            gap_lo: 0.0,
            gap_hi: f64::INFINITY
        }
    );
}
