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
    // 0.134^2 + 0.847^2 rounds so that the computed distance sits below the
    // exact one. At that radius the exact Rips complex has no edge (beta_0 = 2)
    // while the rounded diagram has already merged the pair (beta_0 = 1).
    let pts = [
        ManifoldPoint::<2>::new([0.0, 0.0]),
        ManifoldPoint::<2>::new([0.134, 0.847]),
    ];
    let r = 0.857_534_255_875_530_6;
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
                    // certified_beta0 measures with a scale-safe norm and the
                    // diagram with the plain one; the two agree to a few ulp.
                    assert!(
                        (h - height).abs() <= 4.0 * f64::EPSILON * height.abs(),
                        "scale {s}: diagram {h} vs points {height}"
                    );
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
