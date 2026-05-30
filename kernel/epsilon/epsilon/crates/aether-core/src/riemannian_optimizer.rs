// Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: Epsilon-Hollow

//! Riemannian optimizer primitives migrated from legacy host code.
//!
//! This module keeps the production theorem/runtime path in no-std Rust:
//! fixed-size arrays, no heap allocation, and explicit manifold projection /
//! retraction formulas. Distributed Stiefel synchronization remains in
//! [`crate::parallel_riemannian`]; this file owns local optimizer state.

use libm::{pow, sqrt};

const EPS: f64 = 1e-12;

/// Supported local optimizer manifolds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ManifoldType {
    /// Plain Euclidean space; no tangent projection or curved retraction.
    Euclidean,
    /// Unit sphere with tangent projection `z - <x,z>x`.
    Sphere,
    /// Stiefel manifold. Matrix helper APIs preserve the theorem formulas.
    Stiefel,
    /// Grassmann manifold. Matrix helper APIs preserve the theorem formulas.
    Grassmann,
}

/// Project `z` onto the tangent space of the unit sphere at `x`.
pub fn sphere_project_tangent<const D: usize>(x: &[f64; D], z: &[f64; D]) -> [f64; D] {
    let dot = dot_vector(x, z);
    let mut projected = [0.0_f64; D];
    for idx in 0..D {
        projected[idx] = z[idx] - dot * x[idx];
    }
    projected
}

/// Retract a tangent vector on the sphere by normalizing `x + z`.
pub fn sphere_retract<const D: usize>(x: &[f64; D], z: &[f64; D]) -> [f64; D] {
    let mut y = [0.0_f64; D];
    for idx in 0..D {
        y[idx] = x[idx] + z[idx];
    }

    let norm = vector_norm(&y);
    if norm < EPS {
        return *x;
    }

    for value in &mut y {
        *value /= norm;
    }
    y
}

/// Transport a sphere tangent vector by projecting it onto `T_y S^(D-1)`.
///
/// The legacy documentation describes this transport as:
///
/// ```text
/// Gamma_{x -> y}(v) = Pi_{T_y}(v) = v - <y,v>y
/// ```
pub fn parallel_transport_sphere<const D: usize>(
    _x: &[f64; D],
    y: &[f64; D],
    v: &[f64; D],
) -> [f64; D] {
    sphere_project_tangent(y, v)
}

/// Project `z` onto `T_x St(n,p)` with `Pi_X(Z) = Z - X sym(X^T Z)`.
pub fn stiefel_project_tangent<const N: usize, const P: usize>(
    x: &[[f64; P]; N],
    z: &[[f64; P]; N],
) -> [[f64; P]; N] {
    let mut xtz = [[0.0_f64; P]; P];
    for row in 0..N {
        for col_x in 0..P {
            for col_z in 0..P {
                xtz[col_x][col_z] += x[row][col_x] * z[row][col_z];
            }
        }
    }

    let mut sym = [[0.0_f64; P]; P];
    for row in 0..P {
        for col in 0..P {
            sym[row][col] = 0.5 * (xtz[row][col] + xtz[col][row]);
        }
    }

    let mut projected = *z;
    for row in 0..N {
        for col in 0..P {
            let mut correction = 0.0;
            for mid in 0..P {
                correction += x[row][mid] * sym[mid][col];
            }
            projected[row][col] -= correction;
        }
    }
    projected
}

/// QR-style Stiefel retraction using modified Gram-Schmidt on `x + z`.
pub fn stiefel_retract<const N: usize, const P: usize>(
    x: &[[f64; P]; N],
    z: &[[f64; P]; N],
) -> [[f64; P]; N] {
    let mut y = *x;
    for row in 0..N {
        for col in 0..P {
            y[row][col] += z[row][col];
        }
    }
    orthonormalize_columns(y)
}

/// Project `z` onto `T_x Gr(n,p)` with `Pi_X(Z) = Z - X(X^T Z)`.
pub fn grassmann_project_tangent<const N: usize, const P: usize>(
    x: &[[f64; P]; N],
    z: &[[f64; P]; N],
) -> [[f64; P]; N] {
    let mut xtz = [[0.0_f64; P]; P];
    for row in 0..N {
        for col_x in 0..P {
            for col_z in 0..P {
                xtz[col_x][col_z] += x[row][col_x] * z[row][col_z];
            }
        }
    }

    let mut projected = *z;
    for row in 0..N {
        for col in 0..P {
            let mut correction = 0.0;
            for mid in 0..P {
                correction += x[row][mid] * xtz[mid][col];
            }
            projected[row][col] -= correction;
        }
    }
    projected
}

/// QR-style Grassmann retraction using modified Gram-Schmidt on `x + z`.
pub fn grassmann_retract<const N: usize, const P: usize>(
    x: &[[f64; P]; N],
    z: &[[f64; P]; N],
) -> [[f64; P]; N] {
    stiefel_retract(x, z)
}

/// Fixed-size Riemannian SGD with optional momentum.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RiemannianSgd<const D: usize> {
    lr: f64,
    momentum: f64,
    manifold: ManifoldType,
    velocity: [f64; D],
    previous_point: Option<[f64; D]>,
    steps: u32,
    total_grad_norm: f64,
}

impl<const D: usize> RiemannianSgd<D> {
    /// Create an SGD optimizer for fixed-size vector parameters.
    pub const fn new(lr: f64, momentum: f64, manifold: ManifoldType) -> Self {
        Self {
            lr,
            momentum,
            manifold,
            velocity: [0.0; D],
            previous_point: None,
            steps: 0,
            total_grad_norm: 0.0,
        }
    }

    /// Apply one Riemannian gradient step to vector parameters.
    pub fn step_vector(&mut self, x: [f64; D], grad: [f64; D]) -> [f64; D] {
        let rgrad = project_vector(self.manifold, &x, &grad);
        self.total_grad_norm += vector_norm(&rgrad);
        self.steps = self.steps.saturating_add(1);

        let transported = self
            .previous_point
            .map(|prev| transport_vector(self.manifold, &prev, &x, &self.velocity))
            .unwrap_or([0.0; D]);

        let mut velocity = [0.0_f64; D];
        for idx in 0..D {
            velocity[idx] = if self.previous_point.is_some() {
                self.momentum * transported[idx] + rgrad[idx]
            } else {
                rgrad[idx]
            };
        }

        self.velocity = velocity;
        self.previous_point = Some(x);

        let mut step = [0.0_f64; D];
        for idx in 0..D {
            step[idx] = -self.lr * velocity[idx];
        }
        retract_vector(self.manifold, &x, &step)
    }

    /// Number of optimizer steps applied.
    pub const fn steps(&self) -> u32 {
        self.steps
    }

    /// Average norm of projected gradients observed so far.
    pub fn avg_grad_norm(&self) -> f64 {
        if self.steps == 0 {
            0.0
        } else {
            self.total_grad_norm / f64::from(self.steps)
        }
    }
}

impl<const D: usize> Default for RiemannianSgd<D> {
    fn default() -> Self {
        Self::new(0.01, 0.9, ManifoldType::Stiefel)
    }
}

/// Fixed-size Riemannian Adam with scalar second moment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RiemannianAdam<const D: usize> {
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    manifold: ManifoldType,
    first_moment: [f64; D],
    second_moment: f64,
    previous_point: Option<[f64; D]>,
    steps: u32,
}

impl<const D: usize> RiemannianAdam<D> {
    /// Create an Adam optimizer for fixed-size vector parameters.
    pub const fn new(lr: f64, beta1: f64, beta2: f64, eps: f64, manifold: ManifoldType) -> Self {
        Self {
            lr,
            beta1,
            beta2,
            eps,
            manifold,
            first_moment: [0.0; D],
            second_moment: 0.0,
            previous_point: None,
            steps: 0,
        }
    }

    /// Apply one Riemannian Adam step to vector parameters.
    pub fn step_vector(&mut self, x: [f64; D], grad: [f64; D]) -> [f64; D] {
        self.steps = self.steps.saturating_add(1);
        let rgrad = project_vector(self.manifold, &x, &grad);
        let transported = self
            .previous_point
            .map(|prev| transport_vector(self.manifold, &prev, &x, &self.first_moment))
            .unwrap_or([0.0; D]);

        for idx in 0..D {
            self.first_moment[idx] = if self.previous_point.is_some() {
                self.beta1 * transported[idx] + (1.0 - self.beta1) * rgrad[idx]
            } else {
                (1.0 - self.beta1) * rgrad[idx]
            };
        }

        self.second_moment =
            self.beta2 * self.second_moment + (1.0 - self.beta2) * squared_norm(&rgrad);

        let bias1 = 1.0 - pow(self.beta1, f64::from(self.steps));
        let bias2 = 1.0 - pow(self.beta2, f64::from(self.steps));
        let inv_bias1 = if bias1.abs() < EPS { 1.0 } else { 1.0 / bias1 };
        let v_hat = if bias2.abs() < EPS {
            self.second_moment
        } else {
            self.second_moment / bias2
        };
        let scale = self.lr / (sqrt(v_hat) + self.eps);

        let mut step = [0.0_f64; D];
        for idx in 0..D {
            step[idx] = -scale * self.first_moment[idx] * inv_bias1;
        }

        self.previous_point = Some(x);
        retract_vector(self.manifold, &x, &step)
    }

    /// Number of optimizer steps applied.
    pub const fn steps(&self) -> u32 {
        self.steps
    }
}

impl<const D: usize> Default for RiemannianAdam<D> {
    fn default() -> Self {
        Self::new(0.001, 0.9, 0.999, 1e-8, ManifoldType::Stiefel)
    }
}

fn project_vector<const D: usize>(
    manifold: ManifoldType,
    x: &[f64; D],
    grad: &[f64; D],
) -> [f64; D] {
    match manifold {
        ManifoldType::Sphere => sphere_project_tangent(x, grad),
        ManifoldType::Euclidean | ManifoldType::Stiefel | ManifoldType::Grassmann => *grad,
    }
}

fn retract_vector<const D: usize>(
    manifold: ManifoldType,
    x: &[f64; D],
    step: &[f64; D],
) -> [f64; D] {
    match manifold {
        ManifoldType::Sphere => sphere_retract(x, step),
        ManifoldType::Euclidean | ManifoldType::Stiefel | ManifoldType::Grassmann => {
            let mut y = [0.0_f64; D];
            for idx in 0..D {
                y[idx] = x[idx] + step[idx];
            }
            y
        }
    }
}

fn transport_vector<const D: usize>(
    manifold: ManifoldType,
    prev: &[f64; D],
    x: &[f64; D],
    v: &[f64; D],
) -> [f64; D] {
    match manifold {
        ManifoldType::Sphere => parallel_transport_sphere(prev, x, v),
        ManifoldType::Euclidean | ManifoldType::Stiefel | ManifoldType::Grassmann => *v,
    }
}

fn orthonormalize_columns<const N: usize, const P: usize>(mut q: [[f64; P]; N]) -> [[f64; P]; N] {
    for col in 0..P {
        for prev in 0..col {
            let mut projection = 0.0;
            for row in 0..N {
                projection += q[row][prev] * q[row][col];
            }
            for row in 0..N {
                q[row][col] -= projection * q[row][prev];
            }
        }

        let mut norm_sq = 0.0;
        for row in 0..N {
            norm_sq += q[row][col] * q[row][col];
        }
        let norm = sqrt(norm_sq);

        if norm < EPS {
            for row in 0..N {
                q[row][col] = 0.0;
            }
            if col < N {
                q[col][col] = 1.0;
            }
        } else {
            for row in 0..N {
                q[row][col] /= norm;
            }
        }
    }
    q
}

fn dot_vector<const D: usize>(a: &[f64; D], b: &[f64; D]) -> f64 {
    let mut sum = 0.0;
    for idx in 0..D {
        sum += a[idx] * b[idx];
    }
    sum
}

fn squared_norm<const D: usize>(v: &[f64; D]) -> f64 {
    dot_vector(v, v)
}

fn vector_norm<const D: usize>(v: &[f64; D]) -> f64 {
    sqrt(squared_norm(v))
}
