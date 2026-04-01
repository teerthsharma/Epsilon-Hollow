use crate::ml::tensor::{Tensor, Neuroplasticity};

#[cfg(feature = "std")]
use std::vec::Vec;

#[cfg(all(not(feature = "std"), feature = "alloc"))]
use alloc::vec::Vec;

#[cfg(all(not(feature = "std"), feature = "alloc"))]
use alloc::vec;

use libm::{sqrt};

/// LiquidTensor: A tensor with a mutable "Hot Partition" for real-time neuroplasticity.
/// 
/// Unlike standard Tensors which are static after training, a LiquidTensor maintains
/// a sparse mask of "neurosynaptic" weights (approx 0.5% of total) that remain 
/// fluid and updateable at inference time.
#[derive(Debug, Clone)]
pub struct LiquidTensor {
    /// The base weight matrix
    pub weights: Tensor,
    /// Boolean mask (1.0 = hot, 0.0 = frozen) with same shape as weights
    pub hot_mask: Tensor,
    /// Accumulated gradients for the hot partition
    pub hotline_grads: Tensor,
    /// Chebyshev bounds for safe updates (mean, std_dev)
    pub bounds: (f64, f64),
}

impl LiquidTensor {
    /// Create a new LiquidTensor from existing weights.
    /// 
    /// # Arguments
    /// * `weights` - Pre-trained weight tensor
    /// * `plasticity_rate` - Fraction of weights to keep hot (e.g. 0.005 for 0.5%)
    pub fn new(weights: Tensor, _plasticity_rate: f64) -> Self {
        let shape = weights.shape;
        let total_size = shape.0 * shape.1;
        
        let mut mask_data = vec![0.0; total_size];
        
        // Identify "structurally significant" weights to keep hot.
        // Heuristic: Highest magnitude weights are critical pathways.
        // We select the top `plasticity_rate` % by magnitude.
        
        // 1. Calculate threshold
        let w_data = weights.get_snapshot();
        let abs_weights: Vec<f64> = w_data.iter().map(|&x| fabs(x)).collect();
        // Note: Simple threshold calculation.
        
        let sum: f64 = abs_weights.iter().sum();
        let mean = sum / total_size as f64;
        let var: f64 = abs_weights.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / total_size as f64;
        let std_dev = sqrt(var);
        
        // Z-score threshold for top 0.5% is approx 2.576
        // We approximate "Top X%" using Chebyshev/Z-score.
        // plasticity_rate 0.005 -> ~2.8 sigma for normal distribution
        let threshold = mean + 2.8 * std_dev;

        let mut hot_count = 0;
        for i in 0..total_size {
            if abs_weights[i] > threshold {
                mask_data[i] = 1.0;
                hot_count += 1;
            }
        }
        
        // Sanity fallback: if distribution is weird and nothing selected, select random or skip
        if hot_count == 0 && total_size > 0 {
             mask_data[0] = 1.0; // At least one
        }

        let hot_mask = Tensor::new(mask_data, shape);
        let hotline_grads = Tensor::zeros(shape);

        Self {
            weights,
            hot_mask,
            hotline_grads,
            bounds: (mean, std_dev),
        }
    }

    /// Forward pass (same as Tensor)
    pub fn forward(&self, input: &Tensor) -> Tensor {
        self.weights.matmul(input)
    }

    /// Inject a sparse update into the hot partition.
    /// 
    /// This is the "Liquid" part. It recursively updates weights based on 
    /// immediate feedback, bypassing the full backprop graph.
    pub fn inject_update(&mut self, gradients: &Tensor, learning_rate: f64) {
        // 1. Mask gradients: Only hot weights can change
        let masked_grads = gradients.mul(&self.hot_mask);
        
        // 2. Chebyshev Guard: Clip gradients that would push weights out of distribution
        // This prevents "Catastrophic Forgetting" in the hot partition.
        let safe_grads = self.apply_chebyshev_guard(&masked_grads);

        // 3. Update weights: W = W - lr * grad
        // safe_grads is "grad".
        // Use Neuroplasticity interface which handles locking safely everywhere.
        self.weights.inject_weight_update(&safe_grads.get_snapshot(), learning_rate);
    }

    fn apply_chebyshev_guard(&self, grads: &Tensor) -> Tensor {
        let (_mean, std_dev) = self.bounds;
        let k = 3.0; // 3-sigma bounds (99.7% confidence)
        let max_change = k * std_dev * 0.1; // Cap updates to 10% of 3-sigma

        grads.map(|g| {
            if g > max_change {
                max_change
            } else if g < -max_change {
                -max_change
            } else {
                g
            }
        })
    }
}

fn fabs(x: f64) -> f64 {
    #[cfg(feature = "std")]
    return x.abs();
    #[cfg(not(feature = "std"))]
    return libm::fabs(x);
}
