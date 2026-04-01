// src/ml/tensor.rs
// PHASE SEPARATION: Bio-Time (Kernel) vs. Civil-Time (Daemon)

use cfg_if::cfg_if;

// =================================================================
// 1. THE VECTOR SCHISM RESOLUTION
// =================================================================
cfg_if! {
    if #[cfg(feature = "std")] {
        extern crate std;
        use std::vec::Vec;
        use std::sync::{Arc, RwLock};
        use std::fmt;
        #[cfg(feature = "rayon")]
        use rayon::prelude::*;
        // helper for math
        fn sqrt(x: f64) -> f64 { x.sqrt() }
    } else {
        extern crate alloc;
        use alloc::vec::Vec;
        use alloc::rc::Rc;
        use core::cell::RefCell;
        use alloc::fmt;
        use libm::sqrt;
    }
}

// =================================================================
// 2. THE SHARED API (The "Neuroplasticity" Contract)
// =================================================================
pub trait Neuroplasticity {
    fn get_snapshot(&self) -> Vec<f64>;
    fn inject_weight_update(&mut self, gradient: &[f64], learning_rate: f64);
    fn shape(&self) -> (usize, usize);
}

// =================================================================
// 3. PHASE A: THE BIO-TENSOR (Kernel Mode)
// =================================================================
#[cfg(not(feature = "std"))]
#[derive(Clone)]
pub struct Tensor {
    pub inner: Rc<RefCell<Vec<f64>>>,
    pub shape: (usize, usize),
}

#[cfg(not(feature = "std"))]
impl Tensor {
    pub fn new(data: Vec<f64>, shape: (usize, usize)) -> Self {
        assert_eq!(data.len(), shape.0 * shape.1, "Shape mismatch");
        Self {
            inner: Rc::new(RefCell::new(data)),
            shape,
        }
    }

    pub fn zeros(shape: (usize, usize)) -> Self {
        let size = shape.0 * shape.1;
        Self::new(alloc::vec![0.0; size], shape)
    }

    pub fn kaiming_uniform(shape: (usize, usize), seed: Option<u64>) -> Self {
        let fan_in = shape.1;
        let bound = sqrt(3.0 / fan_in as f64);
        let size = shape.0 * shape.1;
        
        let mut rng = seed.unwrap_or(42);
        let mut data = Vec::with_capacity(size);
        
        for _ in 0..size {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let r = (rng as f64 / u64::MAX as f64) * 2.0 - 1.0;
            data.push(r * bound);
        }
        
        Self::new(data, shape)
    }

    // --- Math Ops (Kernel Mode: Direct Borrow) ---

    pub fn matmul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape.1, other.shape.0, "Matmul dimension mismatch");
        let (m, k) = self.shape;
        let (_, n) = other.shape;
        
        let data_a = self.inner.borrow();
        let data_b = other.inner.borrow();
        let mut data_c = alloc::vec![0.0; m * n];

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for l in 0..k {
                    sum += data_a[i * k + l] * data_b[l * n + j];
                }
                data_c[i * n + j] = sum;
            }
        }
        Tensor::new(data_c, (m, n))
    }

    pub fn add(&self, other: &Tensor) -> Tensor {
        self.elementwise_op(other, |a, b| a + b)
    }

    pub fn sub(&self, other: &Tensor) -> Tensor {
        self.elementwise_op(other, |a, b| a - b)
    }

    pub fn mul(&self, other: &Tensor) -> Tensor {
        self.elementwise_op(other, |a, b| a * b)
    }

    pub fn scale(&self, s: f64) -> Tensor {
        let data = self.inner.borrow();
        let scaled: Vec<f64> = data.iter().map(|v| v * s).collect();
        Tensor::new(scaled, self.shape)
    }

    pub fn map<F>(&self, f: F) -> Tensor where F: Fn(f64) -> f64 {
        let data = self.inner.borrow();
        let mapped: Vec<f64> = data.iter().map(|&v| f(v)).collect();
        Tensor::new(mapped, self.shape)
    }

    pub fn transpose(&self) -> Tensor {
        let (rows, cols) = self.shape;
        let data = self.inner.borrow();
        let mut new_data = alloc::vec![0.0; data.len()];
        
        for i in 0..rows {
            for j in 0..cols {
                new_data[j * rows + i] = data[i * cols + j];
            }
        }
        Tensor::new(new_data, (cols, rows))
    }

    pub fn sum(&self) -> f64 {
        self.inner.borrow().iter().sum()
    }
    
    pub fn max(&self) -> f64 {
        self.inner.borrow().iter().fold(f64::NEG_INFINITY, |a, &b| if a > b { a } else { b })
    }

    fn elementwise_op<F>(&self, other: &Tensor, op: F) -> Tensor 
    where F: Fn(f64, f64) -> f64 {
        assert_eq!(self.shape, other.shape, "Shape mismatch for elementwise op");
        let a = self.inner.borrow();
        let b = other.inner.borrow();
        let res: Vec<f64> = a.iter().zip(b.iter()).map(|(&x, &y)| op(x, y)).collect();
        Tensor::new(res, self.shape)
    }
}

#[cfg(not(feature = "std"))]
impl Neuroplasticity for Tensor {
    fn get_snapshot(&self) -> Vec<f64> {
        self.inner.borrow().clone()
    }

    fn inject_weight_update(&mut self, gradient: &[f64], learning_rate: f64) {
        let mut data = self.inner.borrow_mut();
        for (i, g) in gradient.iter().enumerate() {
            if i < data.len() {
                data[i] -= g * learning_rate;
            }
        }
    }
    
    fn shape(&self) -> (usize, usize) { self.shape }
}

#[cfg(not(feature = "std"))]
impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BioTensor({:?})", self.shape)
    }
}


// =================================================================
// 4. PHASE B: THE CIVIL-TENSOR (Daemon Mode)
// =================================================================
#[cfg(feature = "std")]
#[derive(Clone)]
pub struct Tensor {
    pub inner: Arc<RwLock<Vec<f64>>>,
    pub shape: (usize, usize),
}

#[cfg(feature = "std")]
impl Tensor {
    pub fn new(data: Vec<f64>, shape: (usize, usize)) -> Self {
        assert_eq!(data.len(), shape.0 * shape.1, "Shape mismatch");
        Self {
            inner: Arc::new(RwLock::new(data)),
            shape,
        }
    }

    pub fn zeros(shape: (usize, usize)) -> Self {
        let size = shape.0 * shape.1;
        Self::new(vec![0.0; size], shape)
    }

    pub fn kaiming_uniform(shape: (usize, usize), seed: Option<u64>) -> Self {
        let fan_in = shape.1;
        let bound = sqrt(3.0 / fan_in as f64);
        let size = shape.0 * shape.1;
        
        let mut rng = seed.unwrap_or(42);
        let mut data = Vec::with_capacity(size);
        
        for _ in 0..size {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let r = (rng as f64 / u64::MAX as f64) * 2.0 - 1.0;
            data.push(r * bound);
        }
        
        Self::new(data, shape)
    }

    // --- Math Ops (Daemon Mode: Read Locks) ---

    pub fn matmul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape.1, other.shape.0, "Matmul dimension mismatch");
        let (m, k) = self.shape;
        let (_, n) = other.shape;
        
        let data_a = self.inner.read().expect("Lock poisoned");
        let data_b = other.inner.read().expect("Lock poisoned");
        let mut data_c = vec![0.0; m * n];

        #[cfg(feature = "rayon")]
        {
            let slice_a = data_a.as_slice();
            let slice_b = data_b.as_slice();
            data_c.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
                for (j, val) in row.iter_mut().enumerate() {
                    let mut sum = 0.0;
                    for l in 0..k {
                        sum += slice_a[i * k + l] * slice_b[l * n + j];
                    }
                    *val = sum;
                }
            });
        }

        #[cfg(not(feature = "rayon"))]
        {
            // Could be parallelized with rayon here in Daemon mode
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0;
                    for l in 0..k {
                        sum += data_a[i * k + l] * data_b[l * n + j];
                    }
                    data_c[i * n + j] = sum;
                }
            }
        }
        Tensor::new(data_c, (m, n))
    }

    pub fn add(&self, other: &Tensor) -> Tensor {
        self.elementwise_op(other, |a, b| a + b)
    }

    pub fn sub(&self, other: &Tensor) -> Tensor {
        self.elementwise_op(other, |a, b| a - b)
    }

    pub fn mul(&self, other: &Tensor) -> Tensor {
        self.elementwise_op(other, |a, b| a * b)
    }

    pub fn scale(&self, s: f64) -> Tensor {
        let data = self.inner.read().expect("Lock poisoned");
        let scaled: Vec<f64> = data.iter().map(|v| v * s).collect();
        Tensor::new(scaled, self.shape)
    }

    pub fn map<F>(&self, f: F) -> Tensor where F: Fn(f64) -> f64 {
        let data = self.inner.read().expect("Lock poisoned");
        let mapped: Vec<f64> = data.iter().map(|&v| f(v)).collect();
        Tensor::new(mapped, self.shape)
    }

    pub fn transpose(&self) -> Tensor {
        let (rows, cols) = self.shape;
        let data = self.inner.read().expect("Lock poisoned");
        let mut new_data = vec![0.0; data.len()];
        
        for i in 0..rows {
            for j in 0..cols {
                new_data[j * rows + i] = data[i * cols + j];
            }
        }
        Tensor::new(new_data, (cols, rows))
    }

    pub fn sum(&self) -> f64 {
        self.inner.read().expect("Lock poisoned").iter().sum()
    }
    
    pub fn max(&self) -> f64 {
        self.inner.read().expect("Lock poisoned").iter().fold(f64::NEG_INFINITY, |a, &b| if a > b { a } else { b })
    }

    fn elementwise_op<F>(&self, other: &Tensor, op: F) -> Tensor 
    where F: Fn(f64, f64) -> f64 {
        assert_eq!(self.shape, other.shape, "Shape mismatch for elementwise op");
        let a = self.inner.read().expect("Lock poisoned");
        let b = other.inner.read().expect("Lock poisoned");
        let res: Vec<f64> = a.iter().zip(b.iter()).map(|(&x, &y)| op(x, y)).collect();
        Tensor::new(res, self.shape)
    }
}

#[cfg(feature = "std")]
impl Neuroplasticity for Tensor {
    fn get_snapshot(&self) -> Vec<f64> {
        self.inner.read().expect("Lock poisoned").clone()
    }

    fn inject_weight_update(&mut self, gradient: &[f64], learning_rate: f64) {
        let mut data = self.inner.write().expect("Lock poisoned");
        for (i, g) in gradient.iter().enumerate() {
            if i < data.len() {
                data[i] -= g * learning_rate;
            }
        }
    }
    
    fn shape(&self) -> (usize, usize) { self.shape }
}

#[cfg(feature = "std")]
impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CivilTensor({:?})", self.shape)
    }
}

// =================================================================
// 5. THE TRANSMUTATION BRIDGE (Serialization)
// =================================================================

#[cfg(feature = "serde")]
#[derive(serde::Serialize, serde::Deserialize)]
pub struct TensorCheckpoint {
    pub data: Vec<f64>,
    pub shape: (usize, usize),
}

#[cfg(feature = "serde")]
impl Tensor {
    pub fn freeze(&self) -> TensorCheckpoint {
        TensorCheckpoint {
            data: self.get_snapshot(),
            shape: self.shape(),
        }
    }

    pub fn thaw(checkpoint: TensorCheckpoint) -> Self {
        Self::new(checkpoint.data, checkpoint.shape)
    }
}
