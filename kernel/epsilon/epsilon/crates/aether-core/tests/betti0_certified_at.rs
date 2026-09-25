//! Host tests for `PersistenceDiagram::betti0_certified_at`.
//!
//! `betti_at(radius)` reads beta_0 at one threshold from floating-point merge
//! heights. When the rounded height lands on the radius the answer can
//! disagree with exact arithmetic, and nothing in the result says so. The
//! certified variant reads the same H0 deaths but answers only when none of
//! them lies in the ratio-wide band around the radius.
use aether_core::certified_betti::{certified_beta0, Beta0};
use aether_core::{persistent_homology, ManifoldPoint, PersistenceConfig};

const RHO: f64 = 10.0;

#[test]
fn float_rounded_merge_height_is_refused() {
    // The distance from the origin to [0.1, 0.116] rounds below the exact one,
    // under both the plain and the scale-safe norm. At that radius the exact
    // Rips complex has no edge (beta_0 = 2) while the rounded diagram has
    // already merged the pair (beta_0 = 1). (The earlier fixture [0.134,
    // 0.847] rounded low only under the plain norm the engine no longer uses.)
    let pts = [
        ManifoldPoint::<2>::new([0.0, 0.0]),
        ManifoldPoint::<2>::new([0.1, 0.116]),
    ];
    let r = 0.153_153_517_752_613_18;
    assert_eq!(
        pts[0].distance(&pts[1]),
        r,
        "the rounded distance this test pins"
    );

    let diagram = persistent_homology(&pts, PersistenceConfig::h0_only()).unwrap();
    assert_eq!(
        diagram.betti_at(r).beta_0,
        1,
        "betti_at reports the rounded answer, uncertified"
    );
    assert_eq!(diagram.betti0_certified_at(r, RHO), Err(r));
}

#[test]
fn three_points_certify_only_in_gaps() {
    let pts = [
        ManifoldPoint::<2>::new([0.0, 0.0]),
        ManifoldPoint::<2>::new([1.0, 0.0]),
        ManifoldPoint::<2>::new([3.0, 0.0]),
    ];
    let diagram = persistent_homology(&pts, PersistenceConfig::h0_only()).unwrap();

    assert_eq!(diagram.betti0_certified_at(0.1, RHO), Ok(3));
    assert_eq!(diagram.betti0_certified_at(10.0, RHO), Ok(1));
    // Band [0.379, 3.79] holds both merge heights; 1.0 is nearer 1.2.
    assert_eq!(diagram.betti0_certified_at(1.2, RHO), Err(1.0));
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

#[test]
fn agrees_with_certified_beta0_on_seeded_clouds() {
    // The H0 deaths of a Rips diagram are the minimum-spanning-tree edge
    // lengths, so the diagram path and the point path must give the same
    // certificate: the same value, or a refusal at the same height.
    let mut rng = Rng(0x5EED_0B0E);
    let (mut certified, mut refused) = (0, 0);
    for _ in 0..300 {
        let k = 1 + (rng.next() * 5.0) as usize;
        let n = 1 + (rng.next() * 30.0) as usize;
        let spread = 10f64.powf(-6.0 + 6.0 * rng.next());
        let centres: Vec<[f64; 3]> = (0..k)
            .map(|_| [rng.next() * 10.0, rng.next() * 10.0, rng.next() * 10.0])
            .collect();
        let raw: Vec<[f64; 3]> = (0..n)
            .map(|i| {
                let c = centres[i % k];
                [
                    c[0] + spread * (rng.next() - 0.5),
                    c[1] + spread * (rng.next() - 0.5),
                    c[2] + spread * (rng.next() - 0.5),
                ]
            })
            .collect();
        let pts: Vec<ManifoldPoint<3>> = raw.iter().map(|&c| ManifoldPoint::new(c)).collect();
        let diagram = persistent_homology(&pts, PersistenceConfig::h0_only()).unwrap();

        for _ in 0..5 {
            let s = 10f64.powf(-5.0 + 7.0 * rng.next());
            match (
                certified_beta0(&raw, s, RHO),
                diagram.betti0_certified_at(s, RHO),
            ) {
                (Beta0::Certified { value, .. }, Ok(v)) => {
                    assert_eq!(v, value, "scale {s}");
                    certified += 1;
                }
                (Beta0::Refused { height, .. }, Err(h)) => {
                    // Both paths measure with the same scale-safe norm, so
                    // the refused heights are the same float.
                    assert_eq!(h, height, "scale {s}");
                    refused += 1;
                }
                (a, b) => panic!("scale {s}: points say {a:?}, diagram says {b:?}"),
            }
        }
    }
    assert!(
        certified > 100 && refused > 100,
        "{certified} certified, {refused} refused"
    );
}

/// Certified beta_0 of two points on the x axis `sep` apart.
fn pair_at(sep: f64, radius: f64) -> Result<u32, f64> {
    let pts = [
        ManifoldPoint::<2>::new([0.0, 0.0]),
        ManifoldPoint::<2>::new([sep, 0.0]),
    ];
    persistent_homology(&pts, PersistenceConfig::h0_only())
        .unwrap()
        .betti0_certified_at(radius, RHO)
}

#[test]
fn tiny_separation_is_not_squared_to_zero() {
    // 1e-170 squared underflows to 0, so a plain sqrt(sum(d^2)) merges the
    // pair at height 0 and certifies one component at radius 1e-200, where
    // the two points are 1e30 radii apart. The truth is 2.
    let got = pair_at(1e-170, 1e-200);
    assert!(
        matches!(got, Ok(2) | Err(_)),
        "wrong certified count {got:?}"
    );
}

#[test]
fn huge_separation_is_not_squared_to_infinity() {
    // 1e170 squared overflows to infinity, so a plain norm never merges the
    // pair and certifies two components at radius 1e175. The truth is 1.
    let got = pair_at(1e170, 1e175);
    assert!(
        matches!(got, Ok(1) | Err(_)),
        "wrong certified count {got:?}"
    );
}

fn seeded_cloud(rng: &mut Rng, n: usize) -> Vec<[f64; 3]> {
    (0..n)
        .map(|i| {
            let c = (i % 3) as f64 * 4.0;
            [c + rng.next(), c + rng.next(), c + rng.next()]
        })
        .collect()
}

fn certify(raw: &[[f64; 3]], radius: f64) -> Result<u32, f64> {
    let pts: Vec<ManifoldPoint<3>> = raw.iter().map(|&c| ManifoldPoint::new(c)).collect();
    persistent_homology(&pts, PersistenceConfig::h0_only())
        .unwrap()
        .betti0_certified_at(radius, RHO)
}

const RADII: [f64; 5] = [1e-3, 0.05, 0.4, 2.0, 50.0];

#[test]
fn certified_count_is_permutation_invariant() {
    let mut rng = Rng(0xBE77_1000);
    for _ in 0..40 {
        let n = 3 + (rng.next() * 20.0) as usize;
        let raw = seeded_cloud(&mut rng, n);
        let mut shuffled = raw.clone();
        for i in (1..shuffled.len()).rev() {
            let j = (rng.next() * (i + 1) as f64) as usize;
            shuffled.swap(i, j);
        }
        for r in RADII {
            assert_eq!(certify(&raw, r), certify(&shuffled, r), "radius {r}");
        }
    }
}

#[test]
fn certified_count_is_isometry_invariant() {
    // Axis permutation with reflections: an isometry that rounds nothing.
    let mut rng = Rng(0xBE77_2000);
    for _ in 0..40 {
        let n = 3 + (rng.next() * 20.0) as usize;
        let raw = seeded_cloud(&mut rng, n);
        let moved: Vec<[f64; 3]> = raw.iter().map(|p| [-p[2], p[0], -p[1]]).collect();
        for r in RADII {
            // The count is invariant. A refusal height is a rounded norm whose
            // terms are summed in a different axis order, so it may move by
            // an ulp.
            match (certify(&raw, r), certify(&moved, r)) {
                (Ok(a), Ok(b)) => assert_eq!(a, b, "radius {r}"),
                (Err(a), Err(b)) => {
                    assert!(
                        (a - b).abs() <= 4.0 * f64::EPSILON * a,
                        "radius {r}: {a} vs {b}"
                    )
                }
                (a, b) => panic!("radius {r}: {a:?} vs {b:?}"),
            }
        }
    }
}

#[test]
fn certified_count_is_scale_invariant() {
    // Scaling by a power of two is exact, so the certificate must move with
    // it: the same count, or a refusal at the scaled height. 2^+-600 takes
    // the squared separations past the f64 range in both directions.
    let mut rng = Rng(0xBE77_3000);
    for _ in 0..40 {
        let n = 3 + (rng.next() * 20.0) as usize;
        let raw = seeded_cloud(&mut rng, n);
        for k in [-600, -40, 40, 600] {
            let s = libm::ldexp(1.0, k);
            let scaled: Vec<[f64; 3]> = raw.iter().map(|p| p.map(|c| c * s)).collect();
            for r in RADII {
                let want = certify(&raw, r).map_err(|h| h * s);
                assert_eq!(certify(&scaled, r * s), want, "radius {r}, scale 2^{k}");
            }
        }
    }
}
