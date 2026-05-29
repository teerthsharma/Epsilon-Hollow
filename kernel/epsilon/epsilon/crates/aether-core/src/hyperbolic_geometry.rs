// Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: Epsilon-Hollow

//! no_std Poincare ball operations for hyperbolic embeddings.

use libm::{fabs, log, sqrt, tanh};

const EPS: f64 = 1e-12;
const BALL_MARGIN: f64 = 1e-5;
const DISTANCE_ARG_MARGIN: f64 = 1e-7;
const EXP_ARG_MAX: f64 = 15.0;

/// Errors returned by slice-oriented hyperbolic geometry APIs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeometryError {
    /// Input vectors do not have the same dimension.
    DimensionMismatch,
    /// Output buffer does not match the input dimension.
    OutputDimensionMismatch,
    /// Weights do not match the number of points.
    WeightsDimensionMismatch,
    /// Curvature must be finite and greater than zero.
    InvalidCurvature,
}

/// Result type for fallible hyperbolic geometry operations.
pub type GeometryResult<T> = Result<T, GeometryError>;

/// Poincare ball model with negative curvature `-curvature`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PoincareBall {
    curvature: f64,
    sqrt_curvature: f64,
    max_norm: f64,
}

impl PoincareBall {
    /// Construct a Poincare ball, falling back to unit curvature for invalid input.
    pub fn new(curvature: f64) -> Self {
        match Self::try_new(curvature) {
            Ok(ball) => ball,
            Err(_) => Self::unit(),
        }
    }

    /// Construct a Poincare ball and report invalid curvature.
    pub fn try_new(curvature: f64) -> GeometryResult<Self> {
        if !curvature.is_finite() || curvature <= 0.0 {
            return Err(GeometryError::InvalidCurvature);
        }

        let sqrt_curvature = sqrt(curvature);
        Ok(Self {
            curvature,
            sqrt_curvature,
            max_norm: (1.0 / sqrt_curvature) - BALL_MARGIN,
        })
    }

    /// Construct the unit-curvature Poincare ball.
    pub fn unit() -> Self {
        Self {
            curvature: 1.0,
            sqrt_curvature: 1.0,
            max_norm: 1.0 - BALL_MARGIN,
        }
    }

    /// Return the positive curvature magnitude `c`.
    pub fn curvature(&self) -> f64 {
        self.curvature
    }

    /// Return the largest norm kept strictly inside the ball.
    pub fn max_norm(&self) -> f64 {
        self.max_norm
    }

    /// Project a fixed-size vector into the ball.
    pub fn project<const D: usize>(&self, x: &[f64; D]) -> [f64; D] {
        let mut out = *x;
        self.project_in_place(&mut out);
        out
    }

    /// Project a slice into the caller-provided output buffer.
    pub fn project_into(&self, x: &[f64], out: &mut [f64]) -> GeometryResult<()> {
        if x.len() != out.len() {
            return Err(GeometryError::OutputDimensionMismatch);
        }

        out.copy_from_slice(x);
        self.project_in_place(out);
        Ok(())
    }

    /// Project a mutable slice in place.
    pub fn project_in_place(&self, x: &mut [f64]) {
        let n = norm(x);
        if n >= self.max_norm && n > EPS {
            let scale = self.max_norm / n;
            for value in x {
                *value *= scale;
            }
        }
    }

    /// Compute Mobius addition for fixed-size vectors.
    pub fn mobius_add<const D: usize>(&self, x: &[f64; D], y: &[f64; D]) -> [f64; D] {
        let mut out = [0.0; D];
        let _ = self.mobius_add_into(x, y, &mut out);
        out
    }

    /// Compute Mobius addition into the caller-provided output buffer.
    pub fn mobius_add_into(&self, x: &[f64], y: &[f64], out: &mut [f64]) -> GeometryResult<()> {
        self.mobius_add_signed_x_into(x, y, false, out)
    }

    /// Compute hyperbolic distance for fixed-size vectors.
    pub fn distance<const D: usize>(&self, x: &[f64; D], y: &[f64; D]) -> f64 {
        self.distance_slice(x, y).unwrap_or(0.0)
    }

    /// Compute hyperbolic distance over slices.
    pub fn distance_slice(&self, x: &[f64], y: &[f64]) -> GeometryResult<f64> {
        if x.len() != y.len() {
            return Err(GeometryError::DimensionMismatch);
        }

        let diff_norm = self.mobius_add_signed_x_norm(x, y, true);
        let arg = (self.sqrt_curvature * diff_norm).min(1.0 - DISTANCE_ARG_MARGIN);
        Ok((2.0 / self.sqrt_curvature) * atanh(arg))
    }

    /// Map a fixed-size tangent vector at `x` onto the ball.
    pub fn exp_map<const D: usize>(&self, x: &[f64; D], v: &[f64; D]) -> [f64; D] {
        let mut out = [0.0; D];
        let _ = self.exp_map_into(x, v, &mut out);
        out
    }

    /// Map a tangent vector at `x` onto the ball using caller-provided storage.
    pub fn exp_map_into(&self, x: &[f64], v: &[f64], out: &mut [f64]) -> GeometryResult<()> {
        if x.len() != v.len() {
            return Err(GeometryError::DimensionMismatch);
        }
        if x.len() != out.len() {
            return Err(GeometryError::OutputDimensionMismatch);
        }

        let v_norm = norm(v);
        if v_norm < EPS {
            out.copy_from_slice(x);
            return Ok(());
        }

        let lambda_x = self.conformal_factor(x);
        let arg = (self.sqrt_curvature * lambda_x * v_norm / 2.0).min(EXP_ARG_MAX);
        let scale = tanh(arg) / (self.sqrt_curvature * v_norm);

        self.mobius_add_scaled_y_into(x, v, scale, out)
    }

    /// Map a point on the ball back to the tangent space at `x`.
    pub fn log_map<const D: usize>(&self, x: &[f64; D], y: &[f64; D]) -> [f64; D] {
        let mut out = [0.0; D];
        let _ = self.log_map_into(x, y, &mut out);
        out
    }

    /// Map a point on the ball to the tangent space at `x` using caller storage.
    pub fn log_map_into(&self, x: &[f64], y: &[f64], out: &mut [f64]) -> GeometryResult<()> {
        self.mobius_add_signed_x_into(x, y, true, out)?;

        let diff_norm = norm(out);
        if diff_norm < EPS {
            out.fill(0.0);
            return Ok(());
        }

        let lambda_x = self.conformal_factor(x);
        let arg = (self.sqrt_curvature * diff_norm).min(1.0 - DISTANCE_ARG_MARGIN);
        let scale = (2.0 / (self.sqrt_curvature * lambda_x)) * atanh(arg) / diff_norm;

        for value in out {
            *value *= scale;
        }
        Ok(())
    }

    /// Compute the weighted Einstein midpoint for fixed-size points.
    pub fn centroid<const D: usize>(
        &self,
        points: &[[f64; D]],
        weights: Option<&[f64]>,
    ) -> [f64; D] {
        let mut out = [0.0; D];
        if points.is_empty() {
            return out;
        }

        let mut total_weight = 0.0;
        for (idx, point) in points.iter().enumerate() {
            let weight = weights.and_then(|w| w.get(idx)).copied().unwrap_or(1.0);
            let gamma = lorentz_factor(self.curvature, point);
            let weighted = gamma * weight;
            total_weight += weighted;
            for i in 0..D {
                out[i] += weighted * point[i];
            }
        }

        if total_weight < EPS {
            return [0.0; D];
        }

        for value in &mut out {
            *value /= total_weight;
        }
        self.project_in_place(&mut out);
        out
    }

    /// Compute a weighted Einstein midpoint over slices.
    pub fn centroid_into(
        &self,
        points: &[&[f64]],
        weights: Option<&[f64]>,
        out: &mut [f64],
    ) -> GeometryResult<()> {
        if let Some(weights) = weights {
            if weights.len() != points.len() {
                return Err(GeometryError::WeightsDimensionMismatch);
            }
        }

        out.fill(0.0);
        if points.is_empty() {
            return Ok(());
        }

        for point in points {
            if point.len() != out.len() {
                return Err(GeometryError::DimensionMismatch);
            }
        }

        let mut total_weight = 0.0;
        for (idx, point) in points.iter().enumerate() {
            let weight = weights.map_or(1.0, |w| w[idx]);
            let gamma = lorentz_factor(self.curvature, point);
            let weighted = gamma * weight;
            total_weight += weighted;
            for i in 0..out.len() {
                out[i] += weighted * point[i];
            }
        }

        if total_weight < EPS {
            out.fill(0.0);
            return Ok(());
        }

        for value in out.iter_mut() {
            *value /= total_weight;
        }
        self.project_in_place(out);
        Ok(())
    }

    /// Compute gyration `gyr[u, v](w)` for fixed-size vectors.
    pub fn gyration<const D: usize>(&self, u: &[f64; D], v: &[f64; D], w: &[f64; D]) -> [f64; D] {
        let mut out = [0.0; D];
        let _ = self.gyration_into(u, v, w, &mut out);
        out
    }

    /// Compute gyration `-(u+v) + (u + (v+w))` using Mobius addition.
    pub fn gyration_into(
        &self,
        u: &[f64],
        v: &[f64],
        w: &[f64],
        out: &mut [f64],
    ) -> GeometryResult<()> {
        if u.len() != v.len() || u.len() != w.len() {
            return Err(GeometryError::DimensionMismatch);
        }
        if out.len() != u.len() {
            return Err(GeometryError::OutputDimensionMismatch);
        }

        #[cfg(feature = "alloc")]
        {
            use alloc::vec;
            let mut uv = vec![0.0; u.len()];
            let mut vw = vec![0.0; u.len()];
            let mut uvw = vec![0.0; u.len()];
            self.mobius_add_into(u, v, &mut uv)?;
            self.mobius_add_into(v, w, &mut vw)?;
            self.mobius_add_into(u, &vw, &mut uvw)?;
            self.mobius_add_signed_x_into(&uv, &uvw, true, out)
        }

        #[cfg(not(feature = "alloc"))]
        {
            let _ = (u, v, w, out);
            Err(GeometryError::OutputDimensionMismatch)
        }
    }

    fn conformal_factor(&self, x: &[f64]) -> f64 {
        let norm_sq = dot(x, x);
        let denom = (1.0 - self.curvature * norm_sq).max(EPS);
        2.0 / denom
    }

    fn mobius_add_signed_x_into(
        &self,
        x: &[f64],
        y: &[f64],
        negate_x: bool,
        out: &mut [f64],
    ) -> GeometryResult<()> {
        if x.len() != y.len() {
            return Err(GeometryError::DimensionMismatch);
        }
        if x.len() != out.len() {
            return Err(GeometryError::OutputDimensionMismatch);
        }

        let c = self.curvature;
        let x_sq = dot(x, x);
        let y_sq = dot(y, y);
        let xy = if negate_x { -dot(x, y) } else { dot(x, y) };
        let denom = stable_denom(1.0 + (2.0 * c * xy) + (c * c * x_sq * y_sq));
        let coef_x = 1.0 + (2.0 * c * xy) + (c * y_sq);
        let coef_y = 1.0 - (c * x_sq);

        for i in 0..out.len() {
            let x_value = if negate_x { -x[i] } else { x[i] };
            out[i] = ((coef_x * x_value) + (coef_y * y[i])) / denom;
        }

        self.project_in_place(out);
        Ok(())
    }

    fn mobius_add_scaled_y_into(
        &self,
        x: &[f64],
        y: &[f64],
        y_scale: f64,
        out: &mut [f64],
    ) -> GeometryResult<()> {
        if x.len() != y.len() {
            return Err(GeometryError::DimensionMismatch);
        }
        if x.len() != out.len() {
            return Err(GeometryError::OutputDimensionMismatch);
        }

        let c = self.curvature;
        let x_sq = dot(x, x);
        let y_sq = y_scale * y_scale * dot(y, y);
        let xy = y_scale * dot(x, y);
        let denom = stable_denom(1.0 + (2.0 * c * xy) + (c * c * x_sq * y_sq));
        let coef_x = 1.0 + (2.0 * c * xy) + (c * y_sq);
        let coef_y = 1.0 - (c * x_sq);

        for i in 0..out.len() {
            out[i] = ((coef_x * x[i]) + (coef_y * y_scale * y[i])) / denom;
        }

        self.project_in_place(out);
        Ok(())
    }

    fn mobius_add_signed_x_norm(&self, x: &[f64], y: &[f64], negate_x: bool) -> f64 {
        let c = self.curvature;
        let x_sq = dot(x, x);
        let y_sq = dot(y, y);
        let xy = if negate_x { -dot(x, y) } else { dot(x, y) };
        let denom = stable_denom(1.0 + (2.0 * c * xy) + (c * c * x_sq * y_sq));
        let coef_x = 1.0 + (2.0 * c * xy) + (c * y_sq);
        let coef_y = 1.0 - (c * x_sq);
        let mut sum = 0.0;

        for i in 0..x.len() {
            let x_value = if negate_x { -x[i] } else { x[i] };
            let value = ((coef_x * x_value) + (coef_y * y[i])) / denom;
            sum += value * value;
        }

        sqrt(sum).min(self.max_norm)
    }
}

impl Default for PoincareBall {
    fn default() -> Self {
        Self::unit()
    }
}

/// Compute the Euclidean dot product over slices.
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    let len = a.len().min(b.len());
    let mut sum = 0.0;
    for i in 0..len {
        sum += a[i] * b[i];
    }
    sum
}

/// Compute the Euclidean norm of a slice.
pub fn norm(v: &[f64]) -> f64 {
    sqrt(dot(v, v))
}

/// Tracks the angular momentum conservation bound for sparse attention.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AngularMomentumTracker {
    previous: Option<f64>,
    checks: usize,
    violations: usize,
}

impl AngularMomentumTracker {
    /// Construct a fresh tracker.
    pub const fn new() -> Self {
        Self {
            previous: None,
            checks: 0,
            violations: 0,
        }
    }

    /// Number of non-initial checks performed.
    pub const fn checks(&self) -> usize {
        self.checks
    }

    /// Number of conservation violations observed.
    pub const fn violations(&self) -> usize {
        self.violations
    }

    /// Compute `sum_i active_i * cos(query, centroid_i) * value_norm_i`.
    pub fn compute<const D: usize>(
        query: &[f64; D],
        centroids: &[[f64; D]],
        value_norms: &[f64],
        active_mask: &[bool],
    ) -> f64 {
        if centroids.len() != value_norms.len() || centroids.len() != active_mask.len() {
            return 0.0;
        }

        let query_norm = norm(query);
        if query_norm < EPS {
            return 0.0;
        }

        let mut total = 0.0;
        for (idx, centroid) in centroids.iter().enumerate() {
            if active_mask[idx] {
                total += (dot(query, centroid) / query_norm) * value_norms[idx];
            }
        }
        total
    }

    /// Check `|L_t - L_{t-1}| <= epsilon_t * r_max * ||q||`.
    pub fn check_conservation(
        &mut self,
        l_value: f64,
        epsilon_t: f64,
        r_max: f64,
        q_norm: f64,
    ) -> bool {
        let Some(previous) = self.previous else {
            self.previous = Some(l_value);
            return true;
        };

        self.checks += 1;
        let delta = fabs(l_value - previous);
        let bound = epsilon_t * r_max * q_norm;
        let ok = delta <= bound + 1e-6;
        if !ok {
            self.violations += 1;
        }
        self.previous = Some(l_value);
        ok
    }
}

impl Default for AngularMomentumTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Entropy-gated plasticity upper bound.
pub fn plasticity_bound(alpha: f64, entropy: f64, grad_norm: f64, hot_ratio: f64) -> f64 {
    alpha * (1.0 - entropy) * grad_norm * sqrt(hot_ratio.max(0.0))
}

/// Verify a weight update respects the entropy-gated plasticity bound.
pub fn verify_plasticity_bound(
    delta_w_norm: f64,
    alpha: f64,
    entropy: f64,
    grad_norm: f64,
    hot_ratio: f64,
) -> bool {
    delta_w_norm <= plasticity_bound(alpha, entropy, grad_norm, hot_ratio) + 1e-10
}

fn atanh(x: f64) -> f64 {
    0.5 * log((1.0 + x) / (1.0 - x))
}

fn stable_denom(value: f64) -> f64 {
    if fabs(value) < EPS {
        EPS
    } else {
        value
    }
}

fn lorentz_factor(curvature: f64, x: &[f64]) -> f64 {
    let denom = (1.0 - curvature * dot(x, x)).max(EPS);
    1.0 / sqrt(denom)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tolerance: f64) {
        assert!(
            (a - b).abs() <= tolerance,
            "expected {a} to be within {tolerance} of {b}"
        );
    }

    #[test]
    fn project_keeps_points_strictly_inside_ball() {
        let ball = PoincareBall::new(1.0);
        let projected = ball.project(&[2.0, 0.0, 0.0]);

        close(norm(&projected), ball.max_norm(), 1e-12);
        assert!(norm(&projected) < 1.0);
    }

    #[test]
    fn mobius_add_has_origin_identity() {
        let ball = PoincareBall::new(1.0);
        let x = [0.2, -0.1, 0.05];

        assert_eq!(ball.mobius_add(&x, &[0.0, 0.0, 0.0]), x);
        assert_eq!(ball.mobius_add(&[0.0, 0.0, 0.0], &x), x);
    }

    #[test]
    fn distance_is_symmetric_and_zero_on_self() {
        let ball = PoincareBall::new(1.0);
        let x = [0.1, 0.0, 0.0];
        let y = [0.0, 0.2, 0.0];

        close(ball.distance(&x, &x), 0.0, 1e-10);
        close(ball.distance(&x, &y), ball.distance(&y, &x), 1e-10);
        assert!(ball.distance(&x, &y) > 0.0);
    }

    #[test]
    fn exp_and_log_maps_round_trip_near_origin() {
        let ball = PoincareBall::new(1.0);
        let x = [0.05, -0.02, 0.01];
        let v = [0.01, 0.02, -0.01];

        let y = ball.exp_map(&x, &v);
        let recovered = ball.log_map(&x, &y);

        for i in 0..3 {
            close(recovered[i], v[i], 1e-8);
        }
    }

    #[test]
    fn centroid_respects_weights_and_projects_boundary_points() {
        let ball = PoincareBall::new(1.0);
        let points = [[0.2, 0.0], [0.0, 0.2], [2.0, 0.0]];
        let weights = [1.0, 3.0, 0.0];
        let centroid = ball.centroid(&points, Some(&weights));

        assert!(centroid[1] > centroid[0]);
        assert!(norm(&centroid) < ball.max_norm());
    }

    #[test]
    fn slice_apis_write_into_caller_buffers() {
        let ball = PoincareBall::new(1.0);
        let mut output = [0.0; 3];

        ball.mobius_add_into(&[0.1, 0.0, 0.0], &[0.0, 0.2, 0.0], &mut output)
            .unwrap();
        assert!(output[0] > 0.0);
        assert!(output[1] > 0.0);

        ball.log_map_into(&[0.0, 0.0, 0.0], &output, &mut [0.0; 2])
            .unwrap_err();
    }

    #[test]
    fn gyration_writes_projected_vector() {
        let ball = PoincareBall::new(1.0);
        let out = ball.gyration(&[0.1, 0.0], &[0.0, 0.1], &[0.02, 0.03]);

        assert!(norm(&out) < ball.max_norm());
    }

    #[test]
    fn angular_momentum_tracker_detects_bound_violation() {
        let query = [1.0, 0.0];
        let centroids = [[1.0, 0.0], [0.0, 1.0]];
        let norms = [2.0, 4.0];
        let active = [true, false];
        let l0 = AngularMomentumTracker::compute(&query, &centroids, &norms, &active);

        let mut tracker = AngularMomentumTracker::new();
        assert!(tracker.check_conservation(l0, 0.1, 1.0, 1.0));
        assert!(!tracker.check_conservation(l0 + 1.0, 0.1, 1.0, 1.0));
        assert_eq!(tracker.violations(), 1);
    }

    #[test]
    fn plasticity_bound_matches_expected_scale() {
        let bound = plasticity_bound(0.01, 0.0, 10.0, 0.01);
        close(bound, 0.01, 1e-12);
        assert!(verify_plasticity_bound(0.009, 0.01, 0.0, 10.0, 0.01));
        assert!(!verify_plasticity_bound(0.02, 0.01, 0.0, 10.0, 0.01));
    }
}
