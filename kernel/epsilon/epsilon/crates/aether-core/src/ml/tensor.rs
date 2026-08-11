// Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: Epsilon-Hollow

//! ═══════════════════════════════════════════════════════════════════════════════
//! AEGIS Tensor Engine
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! N-dimensional tensor implementation with shared storage and strided access.
//! Optimized for CPU, with future hooks for wgpu.
//!
//! ═══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "alloc")]
use alloc::sync::Arc;
#[cfg(feature = "alloc")]
use alloc::vec;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;
#[cfg(not(feature = "alloc"))]
use std::sync::Arc;
#[cfg(not(feature = "alloc"))]
use std::vec;
#[cfg(not(feature = "alloc"))]
use std::vec::Vec;

use libm::sqrt;
use spin::Mutex;

/// AEGIS Tensor: N-dimensional array
#[derive(Debug, Clone)]
pub struct Tensor {
    /// Shared data storage (flat generic buffer)
    pub data: Arc<Mutex<Vec<f64>>>,
    /// Cached total element count (avoids locking for size queries)
    pub len: usize,
    /// Shape of the tensor dimensions
    pub shape: Vec<usize>,
    /// Strides for traversing the data
    pub strides: Vec<usize>,
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape
            && self.strides == other.strides
            && *self.data.lock() == *other.data.lock()
    }
}

impl Tensor {
    /// Create a new tensor from a raw vector (consuming it) and shape
    pub fn from_vec(data: Vec<f64>, shape: Vec<usize>) -> Self {
        let total_size: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            total_size,
            "Data length must match shape product"
        );

        let mut strides = vec![0; shape.len()];
        let mut stride = 1;
        for i in (0..shape.len()).rev() {
            strides[i] = stride;
            stride *= shape[i];
        }

        let len = data.len();
        Self {
            data: Arc::new(Mutex::new(data)),
            len,
            shape,
            strides,
        }
    }

    /// Create a new tensor from a slice and shape
    pub fn new(data: &[f64], shape: &[usize]) -> Self {
        let total_size: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            total_size,
            "Data length must match shape product"
        );

        let mut strides = vec![0; shape.len()];
        let mut stride = 1;
        for i in (0..shape.len()).rev() {
            strides[i] = stride;
            stride *= shape[i];
        }

        let data_vec = data.to_vec();
        let len = data_vec.len();
        Self {
            data: Arc::new(Mutex::new(data_vec)),
            len,
            shape: shape.to_vec(),
            strides,
        }
    }

    /// Create a tensor filled with zeros
    pub fn zeros(shape: &[usize]) -> Self {
        let total_size: usize = shape.iter().product();
        let data = vec![0.0; total_size];
        Self::new(&data, shape)
    }

    /// Create a tensor filled with ones
    pub fn ones(shape: &[usize]) -> Self {
        let total_size: usize = shape.iter().product();
        let data = vec![1.0; total_size];
        Self::new(&data, shape)
    }

    /// Create a tensor with Xavier initialization, using the historical
    /// default seed. Kept for existing callers; delegates to
    /// [`Tensor::kaiming_uniform_seeded`].
    pub fn kaiming_uniform(shape: &[usize]) -> Self {
        Self::kaiming_uniform_seeded(shape, 42)
    }

    /// Create a tensor with Xavier initialization from an explicit seed, so
    /// distinct layers don't draw from the same stream.
    pub fn kaiming_uniform_seeded(shape: &[usize], seed: u64) -> Self {
        let total_size: usize = shape.iter().product();
        let fan_in = if shape.len() > 1 { shape[1] } else { 1 };
        let bound = sqrt(3.0 / fan_in as f64);

        let mut rng = crate::ml::rng::Lcg::new(seed);
        let mut data: Vec<f64> = Vec::with_capacity(total_size);

        for _ in 0..total_size {
            data.push(rng.next_signed_f64() * bound);
        }

        Self::new(&data, shape)
    }

    /// Get value at index (handles strides)
    pub fn get(&self, indices: &[usize]) -> f64 {
        let offset = self.compute_offset(indices);
        self.data.lock()[offset]
    }

    /// Set value at index
    pub fn set(&self, indices: &[usize], value: f64) {
        let offset = self.compute_offset(indices);
        self.data.lock()[offset] = value;
    }

    fn compute_offset(&self, indices: &[usize]) -> usize {
        assert_eq!(indices.len(), self.shape.len(), "Index rank mismatch");
        let mut offset = 0;
        for (i, &idx) in indices.iter().enumerate() {
            assert!(idx < self.shape[i], "Index out of bounds");
            offset += idx * self.strides[i];
        }
        offset
    }

    /// Flatten tensor to 1D
    pub fn flatten(&self) -> Self {
        let total_size: usize = self.shape.iter().product();
        let mut new_data = Vec::with_capacity(total_size);
        let data = self.data.lock();

        new_data.extend_from_slice(&data);

        Self {
            data: Arc::new(Mutex::new(new_data)),
            len: total_size,
            shape: vec![total_size],
            strides: vec![1],
        }
    }

    /// Matrix multiplication (2D only for now)
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape.len(), 2, "Matmul requires 2D tensors");
        assert_eq!(other.shape.len(), 2, "Matmul requires 2D tensors");
        assert_eq!(
            self.shape[1], other.shape[0],
            "Dimension mismatch for matmul"
        );

        let m = self.shape[0];
        let k = self.shape[1];
        let n = other.shape[1];

        let result = Tensor::zeros(&[m, n]);

        if Arc::ptr_eq(&self.data, &other.data) {
            let data_a = self.data.lock();
            let mut data_c = result.data.lock();

            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0;
                    for l in 0..k {
                        let val_a = data_a[i * self.strides[0] + l * self.strides[1]];
                        let val_b = data_a[l * self.strides[0] + j * self.strides[1]];
                        sum += val_a * val_b;
                    }
                    data_c[i * result.strides[0] + j * result.strides[1]] = sum;
                }
            }

            drop(data_c);
            return result;
        }

        let data_a = self.data.lock();
        let data_b = other.data.lock();
        let mut data_c = result.data.lock();

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for l in 0..k {
                    let val_a = data_a[i * self.strides[0] + l * self.strides[1]];
                    let val_b = data_b[l * other.strides[0] + j * other.strides[1]];
                    sum += val_a * val_b;
                }
                data_c[i * result.strides[0] + j * result.strides[1]] = sum;
            }
        }

        drop(data_c);
        result
    }

    /// Element-wise addition
    pub fn add(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "Shape mismatch for add");
        let total_size: usize = self.shape.iter().product();
        let mut result_data = Vec::with_capacity(total_size);

        if Arc::ptr_eq(&self.data, &other.data) {
            let data = self.data.lock();
            for i in 0..total_size {
                result_data.push(data[i] + data[i]);
            }
            return Self::new(&result_data, &self.shape);
        }

        let data_a = self.data.lock();
        let data_b = other.data.lock();

        for i in 0..total_size {
            result_data.push(data_a[i] + data_b[i]);
        }

        Self::new(&result_data, &self.shape)
    }

    /// Element-wise multiplication
    pub fn mul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "Shape mismatch for mul");
        let total_size: usize = self.shape.iter().product();
        let mut result_data = Vec::with_capacity(total_size);

        if Arc::ptr_eq(&self.data, &other.data) {
            let data = self.data.lock();
            for i in 0..total_size {
                result_data.push(data[i] * data[i]);
            }
            return Self::new(&result_data, &self.shape);
        }

        let data_a = self.data.lock();
        let data_b = other.data.lock();

        for i in 0..total_size {
            result_data.push(data_a[i] * data_b[i]);
        }

        Self::new(&result_data, &self.shape)
    }

    /// Scalar multiplication
    pub fn scale(&self, s: f64) -> Tensor {
        let total_size: usize = self.shape.iter().product();
        let mut result_data = Vec::with_capacity(total_size);
        let data = self.data.lock();

        for i in 0..total_size {
            result_data.push(data[i] * s);
        }

        Self::new(&result_data, &self.shape)
    }

    /// Transpose (2D)
    pub fn transpose(&self) -> Tensor {
        assert_eq!(self.shape.len(), 2, "Transpose support 2D only for now");
        let rows = self.shape[0];
        let cols = self.shape[1];

        let result = Tensor::zeros(&[cols, rows]);
        let data = self.data.lock();
        let mut res_data = result.data.lock();

        for i in 0..rows {
            for j in 0..cols {
                res_data[j * result.strides[0] + i * result.strides[1]] =
                    data[i * self.strides[0] + j * self.strides[1]];
            }
        }

        drop(res_data);
        result
    }

    /// Sum all elements
    pub fn sum(&self) -> f64 {
        self.data.lock().iter().sum()
    }

    /// Element-wise subtraction
    pub fn sub(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "Shape mismatch for sub");
        let total_size: usize = self.shape.iter().product();
        let mut result_data = Vec::with_capacity(total_size);

        if Arc::ptr_eq(&self.data, &other.data) {
            return Tensor::zeros(&self.shape);
        }

        let data_a = self.data.lock();
        let data_b = other.data.lock();

        for i in 0..total_size {
            result_data.push(data_a[i] - data_b[i]);
        }

        Self::new(&result_data, &self.shape)
    }

    /// Element-wise mapping
    pub fn map<F>(&self, f: F) -> Self
    where
        F: Fn(f64) -> f64,
    {
        let total_size: usize = self.shape.iter().product();
        let mut result_data = Vec::with_capacity(total_size);
        let data = self.data.lock();

        for i in 0..total_size {
            result_data.push(f(data[i]));
        }

        Self::new(&result_data, &self.shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn different_seeds_produce_different_tensors() {
        // Pre-fix, `kaiming_uniform` hardcoded `let mut rng = 42u64;` with no
        // way to vary it, so every call produced the exact same tensor.
        let a = Tensor::kaiming_uniform_seeded(&[4, 4], 1);
        let b = Tensor::kaiming_uniform_seeded(&[4, 4], 2);
        assert_ne!(a, b, "different seeds must produce different tensors");
    }

    #[test]
    fn unseeded_entry_point_delegates_to_seed_42() {
        // Pins only that `kaiming_uniform(shape)` routes to seed 42 under the
        // current generator. It deliberately does NOT claim bit-compatibility
        // with the pre-`Lcg` output: that stream used increment `1` and a
        // full-width `u64 as f64` cast, where `Lcg` uses the Weyl increment and
        // the top 53 bits. The values genuinely changed — for seed 42 the first
        // draw moved from -0.019956656468772427 to 0.1364606532878152. Anything
        // needing the old weights bit-for-bit must reconstruct that formula.
        let default = Tensor::kaiming_uniform(&[4, 4]);
        let seeded_42 = Tensor::kaiming_uniform_seeded(&[4, 4], 42);
        assert_eq!(default, seeded_42);
    }
}
