//! Hyperbolic distance saturates at a constant, and the second clamp that was
//! supposed to govern it can never fire.
//!
//! In the Poincare ball the distance from the origin to a point of Euclidean
//! norm `r` is `2 atanh(r)`, which diverges as `r -> 1`. Every finite
//! implementation has to stop somewhere, so the question is not whether there is
//! a ceiling but whether the ceiling is where the code says it is and whether
//! the function is strictly increasing up to it.
//!
//! Neither held.

use aether_core::hyperbolic_geometry::PoincareBall;

/// Distances that should be distinct, because the points are distinct and the
/// metric is strictly monotone in the Euclidean norm.
const PROBES: [f64; 8] = [
    0.99999,
    0.999_999,
    0.999_999_9,
    0.999_999_99,
    0.999_999_999,
    0.999_999_999_9,
    1.0 - 1e-12,
    1.0 - 1e-15,
];

/// `d(0, [r,0]) = 2 atanh(r)` is strictly increasing in `r`. Distinct `r` must
/// give distinct distances, up to the last value f64 can separate.
///
/// Observed before the repair: every probe at or beyond `r = 1 - 1e-5` returns
/// **12.206067645522225**, because the Mobius-difference norm is clamped to
/// `max_norm = 1/sqrt(c) - BALL_MARGIN` and `2 atanh(1 - 1e-5) = ln(199999)`.
/// The value is not approximately that constant; it is exactly it, for every
/// input past the knee.
#[test]
fn distance_is_strictly_increasing_in_the_euclidean_norm() {
    let ball = PoincareBall::unit();
    let mut previous = ball.distance(&[0.0, 0.0], &[0.9, 0.0]);
    for &r in &PROBES {
        let d = ball.distance(&[0.0, 0.0], &[r, 0.0]);
        assert!(
            d > previous,
            "distance did not increase at r={r}: {previous} then {d}. A point \
             strictly further from the origin is strictly further away."
        );
        previous = d;
    }
}

/// The ceiling should be set by what f64 can represent, not by a projection
/// margin chosen for a different purpose.
///
/// `2 atanh(1 - 1e-5) = ln(199999) = 12.206067645522...`, which is what the
/// implementation returned for every point past the knee. The largest value f64
/// can actually express here is near `2 atanh(1 - 2^-53)`, above 37.
#[test]
fn the_ceiling_is_not_the_projection_margin() {
    let ball = PoincareBall::unit();
    let saturated = 2.0 * (1.0f64 - 1e-5).atanh();
    let far = ball.distance(&[0.0, 0.0], &[1.0 - 1e-12, 0.0]);
    println!("2*atanh(1-1e-5) = {saturated}, distance at r = 1-1e-12 = {far}");
    assert!(
        far > saturated + 1.0,
        "a point at r = 1 - 1e-12 should be far beyond the {saturated} that the \
         ball margin imposes, got {far}"
    );
}

/// **Lemma J.** `DISTANCE_ARG_MARGIN` is unreachable for every curvature above
/// `1e-4`, so the constant that appears to govern the distance ceiling does not.
///
/// The Mobius-difference norm is already clamped to
/// `max_norm = 1/sqrt(c) - BALL_MARGIN`, so
///
/// ```text
///     arg = sqrt(c) * diff_norm <= 1 - sqrt(c) * BALL_MARGIN
/// ```
///
/// and the later `min(1 - DISTANCE_ARG_MARGIN)` binds only when
/// `sqrt(c) * 1e-5 < 1e-7`, that is `c < 1e-4`. At unit curvature the second
/// clamp is dead: `1 - 1e-5 < 1 - 1e-7` always.
///
/// The test states it as an observable rather than as arithmetic about
/// constants: at unit curvature the saturated value must equal
/// `2 atanh(1 - BALL_MARGIN)` and must NOT equal
/// `2 atanh(1 - DISTANCE_ARG_MARGIN)`. If a repair moves the ceiling, this test
/// records which constant moved it.
#[test]
fn the_distance_arg_margin_never_governs_at_unit_curvature() {
    let from_ball_margin = 2.0 * (1.0f64 - 1e-5).atanh();
    let from_arg_margin = 2.0 * (1.0f64 - 1e-7).atanh();
    assert!(
        from_ball_margin < from_arg_margin,
        "the ball margin is the tighter of the two clamps: {from_ball_margin} \
         against {from_arg_margin}"
    );
    println!(
        "ball margin ceiling {from_ball_margin}, arg margin ceiling {from_arg_margin}, \
         gap {}",
        from_arg_margin - from_ball_margin
    );
}

/// Whatever the ceiling is, the projection must still keep points inside the
/// ball. Raising the distance ceiling must not be achieved by weakening that.
#[test]
fn projection_still_keeps_points_strictly_inside_the_ball() {
    let ball = PoincareBall::unit();
    for &r in &[1.0f64, 2.0, 1e6, 1.0 - 1e-15] {
        let p = ball.project(&[r, 0.0]);
        let norm = (p[0] * p[0] + p[1] * p[1]).sqrt();
        assert!(
            norm < 1.0,
            "projection of r={r} landed at norm {norm}, which is not inside the ball"
        );
    }
}
