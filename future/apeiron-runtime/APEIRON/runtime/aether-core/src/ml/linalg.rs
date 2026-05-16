// Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: Epsilon-Hollow

//! ═══════════════════════════════════════════════════════════════════════════════
//! AEGIS Linear Algebra Library
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Complete linear algebra primitives for ML algorithms.
//! Now powered by dynamic Tensors (Bio/Civil Phase Separation).
//!
//! ═══════════════════════════════════════════════════════════════════════════════

#![allow(dead_code)]

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(feature = "std")]
use std::vec::Vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use super::tensor::{Tensor, Neuroplasticity};

// ═══════════════════════════════════════════════════════════════════════════════
// Loss Functions
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LossConfig {
    MSE,
    MAE,
    BinaryCrossEntropy,
    Hinge,
}

impl LossConfig {
    /// Compute loss value
    pub fn compute(&self, y_true: &Tensor, y_pred: &Tensor) -> f64 {
        match self {
            LossConfig::MSE => mse(y_true, y_pred),
            LossConfig::MAE => mae(y_true, y_pred),
            LossConfig::BinaryCrossEntropy => binary_cross_entropy(y_true, y_pred),
            LossConfig::Hinge => hinge_loss(y_true, y_pred),
        }
    }

    /// Compute derivative (gradient) w.r.t prediction
    pub fn derivative(&self, y_true: &Tensor, y_pred: &Tensor) -> Tensor {
        let (rows, cols) = y_true.shape;
        let n: f64 = (rows * cols) as f64;

        match self {
            LossConfig::MSE => {
                let diff = y_pred.sub(y_true);
                diff.scale(2.0 / n)
            }
            LossConfig::MAE => {
                let diff = y_pred.sub(y_true);
                diff.map(|x| if x > 0.0 { 1.0 / n } else if x < 0.0 { -1.0 / n } else { 0.0 })
            }
            LossConfig::BinaryCrossEntropy => {
                // dL/dp = (1-y)/(1-p) - y/p
                
                let true_data = y_true.get_snapshot();
                let pred_data = y_pred.get_snapshot();
                
                let len = true_data.len();
                let mut grad_data = Vec::with_capacity(len); 
                
                for i in 0..len {
                    let y = true_data[i];
                    let p = pred_data[i].clamp(1e-7, 1.0 - 1e-7); // Avoid div by zero
                    
                    let grad = -(y / p) + ((1.0 - y) / (1.0 - p));
                    grad_data.push(grad / len as f64);
                }
                Tensor::new(grad_data, y_pred.shape)
            }
            LossConfig::Hinge => {
                // L = max(0, 1 - y*p)
                // dL/dp = -y if 1 - y*p > 0 else 0
                let true_data = y_true.get_snapshot();
                let pred_data = y_pred.get_snapshot();
                
                let len = true_data.len();
                let mut grad_data = Vec::with_capacity(len);

                for i in 0..len {
                    let y = true_data[i];
                    let p = pred_data[i];
                    
                    if 1.0 - y * p > 0.0 {
                        grad_data.push(-y / len as f64);
                    } else {
                        grad_data.push(0.0);
                    }
                }
                Tensor::new(grad_data, y_pred.shape)
            }
        }
    }
}

/// Mean Squared Error
pub fn mse(y_true: &Tensor, y_pred: &Tensor) -> f64 {
    let diff = y_true.sub(y_pred);
    let (rows, cols) = y_true.shape;
    let n = (rows * cols) as f64;
    diff.mul(&diff).sum() / n
}

/// Mean Absolute Error
pub fn mae(y_true: &Tensor, y_pred: &Tensor) -> f64 {
    let diff = y_true.sub(y_pred);
    let (rows, cols) = y_true.shape;
    let n = (rows * cols) as f64;
    // Utilize high-level map/sum instead of explicit loop + clone if possible, 
    // but map returns tensor. 
    // Optimization: diff.map(|x| x.abs()).sum() / n
    diff.map(fabs).sum() / n
}

/// Root Mean Squared Error
pub fn rmse(y_true: &Tensor, y_pred: &Tensor) -> f64 {
    sqrt(mse(y_true, y_pred))
}

/// Binary Cross-Entropy
pub fn binary_cross_entropy(y_true: &Tensor, y_pred: &Tensor) -> f64 {
    let mut sum = 0.0;
    
    let true_data = y_true.get_snapshot();
    let pred_data = y_pred.get_snapshot();
    
    let n = true_data.len().min(pred_data.len());
    
    for i in 0..n {
        let p = pred_data[i].clamp(1e-7, 1.0 - 1e-7);
        let y = true_data[i];
        
        #[cfg(not(feature = "std"))]
        {
             sum -= y * log(p) + (1.0 - y) * log(1.0 - p);
        }
        #[cfg(feature = "std")]
        {
             sum -= y * p.ln() + (1.0 - y) * (1.0 - p).ln();
        }
    }
    sum / n as f64
}

/// Hinge Loss (for SVM)
pub fn hinge_loss(y_true: &Tensor, y_pred: &Tensor) -> f64 {
    let mut sum = 0.0;
    
    let true_data = y_true.get_snapshot();
    let pred_data = y_pred.get_snapshot();
    
    let n = true_data.len().min(pred_data.len());
    
    for i in 0..n {
        let margin = 1.0 - true_data[i] * pred_data[i];
        if margin > 0.0 {
            sum += margin;
        }
    }
    sum / n as f64
}

// ═══════════════════════════════════════════════════════════════════════════════
// Gradient Computation
// ═══════════════════════════════════════════════════════════════════════════════

/// Numerical gradient of f at x
pub fn numerical_gradient<F>(f: F, x: &Tensor, epsilon: f64) -> Tensor
where
    F: Fn(&Tensor) -> f64,
{
    // Clone structure
    let (rows, cols) = x.shape;
    let n = rows * cols; // Fixed shape access
    let _grad = Tensor::zeros((rows, cols));
    
    // We can't easily mutate x in place with get_snapshot/Tensor structure safely across phases without new()
    // Strategy: Create modified copies. Slow but safe.
    
    // Actually, x is &Tensor.
    // If we want to probe, we need to interact with it.
    // Efficient way: copy data once, mutate local vec, create new Tensor for probe.
    
    let base_data = x.get_snapshot();
    
    // We need to construct gradient vector
    let mut grad_data = Vec::with_capacity(n);

    for i in 0..n {
        let original = base_data[i];
        
        // Plus probe
        let mut data_plus = base_data.clone();
        data_plus[i] = original + epsilon;
        let x_plus = Tensor::new(data_plus, (rows, cols));
        
        // Minus probe
        let mut data_minus = base_data.clone();
        data_minus[i] = original - epsilon;
        let x_minus = Tensor::new(data_minus, (rows, cols));
        
        let val = (f(&x_plus) - f(&x_minus)) / (2.0 * epsilon);
        grad_data.push(val);
    }

    Tensor::new(grad_data, (rows, cols))
}

// ═══════════════════════════════════════════════════════════════════════════════
// Distance Functions
// ═══════════════════════════════════════════════════════════════════════════════

/// Euclidean distance
pub fn euclidean_distance(a: &Tensor, b: &Tensor) -> f64 {
    let diff = a.sub(b);
    sqrt(diff.mul(&diff).sum())
}

/// Manhattan distance (L1)
pub fn manhattan_distance(a: &Tensor, b: &Tensor) -> f64 {
    let diff = a.sub(b);
    // Reuse map/sum
    diff.map(fabs).sum()
}

/// Chebyshev distance (L∞)
pub fn chebyshev_distance(a: &Tensor, b: &Tensor) -> f64 {
    let diff = a.sub(b);
    let data = diff.get_snapshot();

    let mut max = 0.0;
    for &val in data.iter() {
        let abs_val = fabs(val);
        if abs_val > max {
            max = abs_val;
        }
    }
    max
}

/// RBF kernel value
pub fn rbf_kernel(a: &Tensor, b: &Tensor, gamma: f64) -> f64 {
    let dist = a.sub(b);
    let dist_sq = dist.mul(&dist).sum();
    exp(-gamma * dist_sq)
}

fn fabs(x: f64) -> f64 {
    #[cfg(feature = "std")]
    return x.abs();
    #[cfg(not(feature = "std"))]
    return libm::fabs(x);
}

fn sqrt(x: f64) -> f64 {
    #[cfg(feature = "std")]
    return x.sqrt();
    #[cfg(not(feature = "std"))]
    return libm::sqrt(x);
}

#[cfg(not(feature = "std"))]
fn log(x: f64) -> f64 {
    libm::log(x)
}

#[cfg(feature = "std")]
fn exp(x: f64) -> f64 {
    x.exp()
}
#[cfg(not(feature = "std"))]
fn exp(x: f64) -> f64 {
    libm::exp(x)
}
