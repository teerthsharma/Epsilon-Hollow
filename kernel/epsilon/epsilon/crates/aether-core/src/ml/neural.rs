// Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: Epsilon-Hollow

//! ═══════════════════════════════════════════════════════════════════════════════
//! AEGIS Neural Network Library
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Neural networks with topological regularization and seal-loop training.
//! Now powered by dynamic Tensors and proper Optimizers.
//!
//! ═══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

// NOTE: do not `use std::f64;` here. That imports the *module*, which shadows
// the `f64` primitive in path position, so `f64::NEG_INFINITY` and `f64::MAX`
// resolve to the deprecated module constants rather than the associated
// constants on the type. Under `-D warnings` — which the miri job uses — that
// is a hard compile error, and it is why every other file in this crate writes
// the same expressions without incident.

use super::linalg::LossConfig;
use super::tensor::Tensor;

// ═══════════════════════════════════════════════════════════════════════════════
// Activation Functions
// ═══════════════════════════════════════════════════════════════════════════════

/// Activation function types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Activation {
    ReLU,
    Sigmoid,
    Tanh,
    Linear,
    LeakyReLU,
    Softmax,
}

impl Activation {
    /// Apply activation to a tensor
    pub fn apply(&self, x: &Tensor) -> Tensor {
        match self {
            Activation::Softmax => {
                let data_borrow = x.data.lock();
                let max_val = data_borrow.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let mut sum = 0.0;
                let data: Vec<f64> = data_borrow
                    .iter()
                    .map(|&v| {
                        let e = exp(v - max_val);
                        sum += e;
                        e
                    })
                    .collect();

                let normalized: Vec<f64> = data.iter().map(|&v| v / sum.max(1e-10)).collect();
                Tensor::new(&normalized, &x.shape)
            }
            _ => x.map(|v| self.apply_scalar(v)),
        }
    }

    /// Apply to single value
    pub fn apply_scalar(&self, x: f64) -> f64 {
        match self {
            Activation::ReLU => {
                if x > 0.0 {
                    x
                } else {
                    0.0
                }
            }
            Activation::Sigmoid => 1.0 / (1.0 + exp(-x.clamp(-500.0, 500.0))),
            Activation::Tanh => {
                let e_pos = exp(x.clamp(-500.0, 500.0));
                let e_neg = exp((-x).clamp(-500.0, 500.0));
                (e_pos - e_neg) / (e_pos + e_neg)
            }
            Activation::Linear => x,
            Activation::LeakyReLU => {
                if x > 0.0 {
                    x
                } else {
                    0.01 * x
                }
            }
            Activation::Softmax => x, // Should not be called on scalar
        }
    }

    /// Derivative for backprop
    pub fn derivative(&self, x: &Tensor) -> Tensor {
        match self {
            Activation::Softmax => Tensor::zeros(&x.shape), // Handled specially
            _ => x.map(|v| self.derivative_scalar(v)),
        }
    }

    fn derivative_scalar(&self, x: f64) -> f64 {
        match self {
            Activation::ReLU => {
                if x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Activation::Sigmoid => {
                let s = self.apply_scalar(x);
                s * (1.0 - s)
            }
            Activation::Tanh => {
                let t = self.apply_scalar(x);
                1.0 - t * t
            }
            Activation::Linear => 1.0,
            Activation::LeakyReLU => {
                if x > 0.0 {
                    1.0
                } else {
                    0.01
                }
            }
            Activation::Softmax => 1.0,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Optimizers
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub enum OptimizerConfig {
    SGD {
        learning_rate: f64,
        momentum: f64,
    },
    Adam {
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
    },
}

#[derive(Debug, Clone)]
pub enum OptimizerState {
    SGD {
        velocity_w: Tensor,
        velocity_b: Tensor,
    },
    Adam {
        m_w: Tensor,
        v_w: Tensor,
        m_b: Tensor,
        v_b: Tensor,
        t: u64,
    },
    None,
}

// ═══════════════════════════════════════════════════════════════════════════════
// Dense Layer
// ═══════════════════════════════════════════════════════════════════════════════

/// Dense (fully connected) layer
#[derive(Debug, Clone)]
pub struct DenseLayer {
    pub weights: Tensor, // [output_size, input_size]
    pub biases: Tensor,  // [output_size]
    pub input_size: usize,
    pub output_size: usize,
    pub activation: Activation,

    // Cache for backprop
    last_input: Option<Tensor>,
    last_z: Option<Tensor>,

    // Optimizer State
    opt_state: OptimizerState,
}

impl DenseLayer {
    pub fn new(
        input_size: usize,
        output_size: usize,
        activation: Activation,
        seed: Option<u64>,
    ) -> Self {
        // Xavier initialization
        let scale = sqrt(2.0 / (input_size + output_size) as f64);

        let mut rng = super::rng::Lcg::new(seed.unwrap_or(42));
        let mut w_data = Vec::with_capacity(input_size * output_size);
        for _ in 0..(input_size * output_size) {
            w_data.push(rng.next_signed_f64() * scale);
        }

        let weights = Tensor::new(&w_data, &[output_size, input_size]);
        let biases = Tensor::zeros(&[output_size, 1]);

        Self {
            weights,
            biases,
            input_size,
            output_size,
            activation,
            last_input: None,
            last_z: None,
            opt_state: OptimizerState::None,
        }
    }

    pub fn init_optimizer(&mut self, config: &OptimizerConfig) {
        match config {
            OptimizerConfig::SGD { .. } => {
                self.opt_state = OptimizerState::SGD {
                    velocity_w: Tensor::zeros(&self.weights.shape),
                    velocity_b: Tensor::zeros(&self.biases.shape),
                };
            }
            OptimizerConfig::Adam { .. } => {
                self.opt_state = OptimizerState::Adam {
                    m_w: Tensor::zeros(&self.weights.shape),
                    v_w: Tensor::zeros(&self.weights.shape),
                    m_b: Tensor::zeros(&self.biases.shape),
                    v_b: Tensor::zeros(&self.biases.shape),
                    t: 0,
                };
            }
        }
    }

    /// Forward pass
    pub fn forward(&mut self, input: &Tensor) -> Tensor {
        self.last_input = Some(input.clone());

        // z = W * x + b
        // weights: [out, in], input: [in] -> [out]

        let wx = self.weights.matmul(input);
        let z = wx.add(&self.biases);

        self.last_z = Some(z.clone());
        self.activation.apply(&z)
    }

    /// Backward pass
    pub fn backward(&mut self, grad_output: &Tensor, config: &OptimizerConfig) -> Tensor {
        let last_z = self
            .last_z
            .as_ref()
            .expect("Forward must be called before backward")
            .clone();
        let last_input = self
            .last_input
            .as_ref()
            .expect("Forward must be called before backward")
            .clone();

        let act_deriv = self.activation.derivative(&last_z);
        let delta = grad_output.mul(&act_deriv);

        // Gradients
        // dW = delta * input^T
        // delta: [out], input: [in]

        let mut dw_data = Vec::with_capacity(self.output_size * self.input_size);
        {
            let delta_data = delta.data.lock();
            let input_data = last_input.data.lock();

            for i in 0..self.output_size {
                for j in 0..self.input_size {
                    dw_data.push(delta_data[i] * input_data[j]);
                }
            }
        }
        let grad_w = Tensor::new(&dw_data, &self.weights.shape);
        let grad_b = delta.clone();

        // Compute input gradient for next layer
        // dx = W^T * delta
        let w_t = self.weights.transpose();
        let grad_input = w_t.matmul(&delta);

        self.update_weights(&grad_w, &grad_b, config);

        grad_input
    }

    fn update_weights(&mut self, grad_w: &Tensor, grad_b: &Tensor, config: &OptimizerConfig) {
        match config {
            OptimizerConfig::SGD {
                learning_rate,
                momentum,
            } => {
                if let OptimizerState::SGD {
                    velocity_w,
                    velocity_b,
                } = &mut self.opt_state
                {
                    *velocity_w = velocity_w
                        .scale(*momentum)
                        .sub(&grad_w.scale(*learning_rate));
                    *velocity_b = velocity_b
                        .scale(*momentum)
                        .sub(&grad_b.scale(*learning_rate));

                    self.weights = self.weights.add(velocity_w);
                    self.biases = self.biases.add(velocity_b);
                }
            }
            OptimizerConfig::Adam {
                learning_rate,
                beta1,
                beta2,
                epsilon,
            } => {
                if let OptimizerState::Adam {
                    m_w,
                    v_w,
                    m_b,
                    v_b,
                    t,
                } = &mut self.opt_state
                {
                    *t += 1;
                    let t_val = *t as f64;

                    // Weights
                    *m_w = m_w.scale(*beta1).add(&grad_w.scale(1.0 - beta1));
                    *v_w = v_w
                        .scale(*beta2)
                        .add(&grad_w.mul(grad_w).scale(1.0 - beta2));

                    let m_hat_w = m_w.scale(1.0 / (1.0 - pow(*beta1, t_val)));
                    let v_hat_w = v_w.scale(1.0 / (1.0 - pow(*beta2, t_val)));

                    let update_w = m_hat_w
                        .mul(&v_hat_w.map(|x| 1.0 / (sqrt(x) + epsilon)))
                        .scale(*learning_rate);
                    self.weights = self.weights.sub(&update_w);

                    // Biases
                    *m_b = m_b.scale(*beta1).add(&grad_b.scale(1.0 - beta1));
                    *v_b = v_b
                        .scale(*beta2)
                        .add(&grad_b.mul(grad_b).scale(1.0 - beta2));

                    let m_hat_b = m_b.scale(1.0 / (1.0 - pow(*beta1, t_val)));
                    let v_hat_b = v_b.scale(1.0 / (1.0 - pow(*beta2, t_val)));

                    let update_b = m_hat_b
                        .mul(&v_hat_b.map(|x| 1.0 / (sqrt(x) + epsilon)))
                        .scale(*learning_rate);
                    self.biases = self.biases.sub(&update_b);
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Multi-Layer Perceptron
// ═══════════════════════════════════════════════════════════════════════════════

/// Multi-Layer Perceptron neural network
#[derive(Debug, Clone)]
pub struct MLP {
    pub layers: Vec<DenseLayer>,
    pub config: OptimizerConfig,
    pub loss: LossConfig,
}

impl MLP {
    pub fn new(config: OptimizerConfig, loss: LossConfig) -> Self {
        Self {
            layers: Vec::new(),
            config,
            loss,
        }
    }

    /// Add a dense layer
    pub fn add_layer(
        &mut self,
        input_size: usize,
        output_size: usize,
        activation: Activation,
        seed: Option<u64>,
    ) {
        let mut layer = DenseLayer::new(input_size, output_size, activation, seed);
        layer.init_optimizer(&self.config);
        self.layers.push(layer);
    }

    /// Forward pass through all layers
    pub fn forward(&mut self, input: &Tensor) -> Tensor {
        let mut current = input.clone();
        for layer in &mut self.layers {
            current = layer.forward(&current);
        }
        current
    }

    /// Predict (Forward without mutating state if possible? No, dense layer caches input)
    pub fn predict(&mut self, input: &Tensor) -> Tensor {
        self.forward(input)
    }

    /// Train on single sample (returns loss)
    pub fn train_step(&mut self, input: &Tensor, target: &Tensor) -> f64 {
        // Forward
        let output = self.forward(input);

        // Loss
        let loss = self.loss.compute(target, &output);

        // Initial gradient
        let grad = self.loss.derivative(target, &output);

        // Backward
        let mut current_grad = grad;
        for layer in self.layers.iter_mut().rev() {
            current_grad = layer.backward(&current_grad, &self.config);
        }

        loss
    }

    /// Train for `epochs` full passes over `x`/`y` and report what happened.
    ///
    /// [`TrainingResult::converged`] is true only when the run *ended* on a
    /// plateau: the last [`CONVERGENCE_PATIENCE`] epochs each failed to beat the
    /// best mean loss seen so far by more than [`CONVERGENCE_TOL`]. Comparing
    /// against the best rather than the previous epoch means a loss oscillating
    /// without net progress counts as stalled, which is what it is.
    ///
    /// Exhausting the epoch budget while still improving is a normal outcome,
    /// not an error: it reports `converged == false` and the caller decides.
    /// Training is never cut short — the full budget always runs, so the
    /// reported `final_loss` is the loss of the network the caller now holds.
    ///
    /// `converged` describes the optimizer, not the model. A run that stalls at
    /// a useless loss — a saturated or dead network that stops moving — has
    /// converged in this sense. Judging the result is the job of `final_loss`.
    pub fn fit(&mut self, x: &[Tensor], y: &[Tensor], epochs: usize) -> TrainingResult {
        let mut result = TrainingResult::default();
        let n_samples = x.len();
        let mut best_loss = f64::MAX;
        let mut epochs_without_improvement = 0usize;

        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            for i in 0..n_samples {
                total_loss += self.train_step(&x[i], &y[i]);
            }
            let avg_loss = total_loss / n_samples as f64;

            if best_loss - avg_loss > CONVERGENCE_TOL {
                best_loss = avg_loss;
                epochs_without_improvement = 0;
            } else {
                best_loss = best_loss.min(avg_loss);
                epochs_without_improvement += 1;
            }

            if epoch < 100 {
                result.loss_history.push(avg_loss);
            }
            result.final_loss = avg_loss;
        }

        result.epochs = epochs as u32;
        // A run that diverged to NaN or infinity stops "improving" too, and
        // would otherwise claim the plateau. It has not converged to anything.
        // ponytail: a *finite* stall at a bad loss (saturated sigmoid pinned at
        // 0.5) still reports true; separating "stopped" from "stopped somewhere
        // useful" needs a problem-scale reference this signature does not have,
        // so callers pair this with `final_loss`, as `test_mlp_xor` does.
        result.converged =
            epochs_without_improvement >= CONVERGENCE_PATIENCE && result.final_loss.is_finite();
        result
    }
}

/// Smallest mean-loss improvement over the best epoch so far that still counts
/// as progress in [`MLP::fit`].
///
/// Fixed rather than passed in: every call site in the tree
/// (`seal-os::ml_engine::demo_train_mlp`, the `aether-lang` interpreter's `train`
/// method) calls `fit(x, y, epochs)`, so a fourth parameter would break both to
/// hand them a value neither has an opinion about. The pair below matches
/// scikit-learn's `MLPClassifier` defaults (`tol`, `n_iter_no_change`), which is
/// where the convention comes from.
pub const CONVERGENCE_TOL: f64 = 1e-4;

/// Consecutive epochs without an improvement of [`CONVERGENCE_TOL`] before
/// [`MLP::fit`] calls the run converged.
pub const CONVERGENCE_PATIENCE: usize = 10;

/// Training result
#[derive(Debug, Clone)]
pub struct TrainingResult {
    pub epochs: u32,
    pub final_loss: f64,
    pub converged: bool,
    pub loss_history: Vec<f64>,
}

impl Default for TrainingResult {
    fn default() -> Self {
        Self {
            epochs: 0,
            final_loss: f64::MAX,
            converged: false,
            loss_history: Vec::new(),
        }
    }
}

pub use OptimizerConfig::*;

fn pow(base: f64, exp: f64) -> f64 {
    #[cfg(feature = "std")]
    return base.powf(exp);
    #[cfg(not(feature = "std"))]
    return libm::pow(base, exp);
}

fn sqrt(x: f64) -> f64 {
    #[cfg(feature = "std")]
    return x.sqrt();
    #[cfg(not(feature = "std"))]
    return libm::sqrt(x);
}

fn exp(x: f64) -> f64 {
    #[cfg(feature = "std")]
    return x.exp();
    #[cfg(not(feature = "std"))]
    return libm::exp(x);
}

#[cfg(test)]
mod tests {
    use super::OptimizerConfig;
    use super::*;

    #[test]
    fn dense_layer_weights_match_shared_lcg_implementation() {
        // Pre-fix, DenseLayer ran its own inline copy of the LCG recurrence
        // (`rng = rng * A + 1`, then `rng as f64 / u64::MAX as f64`). This
        // pins its first weight to the shared `Lcg::next_signed_f64` draw,
        // which uses a different additive constant and reads the top 53
        // bits instead of converting the raw state -- the whole point of
        // migrating every call site onto one implementation.
        let layer = DenseLayer::new(2, 3, Activation::Linear, Some(7));
        let scale = sqrt(2.0 / (2 + 3) as f64);
        let mut rng = super::super::rng::Lcg::new(7);
        let expected_first_weight = rng.next_signed_f64() * scale;
        let actual_first_weight = layer.weights.get(&[0, 0]);
        assert!(
            (actual_first_weight - expected_first_weight).abs() < 1e-12,
            "DenseLayer must draw its weights from the shared Lcg implementation: got {actual_first_weight}, expected {expected_first_weight}"
        );
    }

    /// The XOR training set, pinned to the shapes the seeded runs below expect.
    fn xor_problem() -> (Vec<Tensor>, Vec<Tensor>) {
        (
            vec![
                Tensor::new(&[0.0, 0.0], &[2, 1]),
                Tensor::new(&[0.0, 1.0], &[2, 1]),
                Tensor::new(&[1.0, 0.0], &[2, 1]),
                Tensor::new(&[1.0, 1.0], &[2, 1]),
            ],
            vec![
                Tensor::new(&[0.0], &[1, 1]),
                Tensor::new(&[1.0], &[1, 1]),
                Tensor::new(&[1.0], &[1, 1]),
                Tensor::new(&[0.0], &[1, 1]),
            ],
        )
    }

    /// A freshly initialised 2-8-1 net on the pinned seed pair (42, 43).
    fn xor_mlp() -> MLP {
        let config = OptimizerConfig::SGD {
            learning_rate: 0.1,
            momentum: 0.9,
        };
        let mut mlp = MLP::new(config, LossConfig::MSE);
        mlp.add_layer(2, 8, Activation::Tanh, Some(42));
        mlp.add_layer(8, 1, Activation::Sigmoid, Some(43));
        mlp
    }

    #[test]
    fn test_mlp_xor() {
        let mut mlp = xor_mlp();
        let (x, y) = xor_problem();

        let result = mlp.fit(&x, &y, 500);
        // No longer a tautology: `fit` computes this from the loss trajectory,
        // and the same schedule reports `false` for any budget under 117 epochs
        // (see `fit_reports_false_when_the_epoch_budget_runs_out`). 500 clears
        // that boundary by 4.3x.
        assert!(
            result.converged,
            "XOR training did not reach a loss plateau within 500 epochs (final loss {})",
            result.final_loss
        );

        // Bound justified by measurement, not by guess. Sweeping 64 seed pairs
        // (`(2s+42, 2s+43)` for `s` in `0..64`) through this exact topology and
        // schedule puts every final loss in
        // `[2.2214669674426596e-4, 4.126699150326758e-4]`; the pinned pair
        // (42, 43) lands at `2.615295509201837e-4`. A 1e-3 ceiling therefore
        // clears the worst observed seed by 2.4x — wide enough that reseeding or
        // cross-target float reassociation cannot flake it — while still failing
        // on any regression that costs so much as a factor of 4. The 0.1 this
        // replaces was ~380x the true loss: an untrained network emits a
        // constant 0.5 and scores 0.2505, so 0.1 would not even have caught
        // training being switched off entirely, which is what a zeroed learning
        // rate demonstrates.
        assert!(
            result.final_loss < 1e-3,
            "XOR training regressed: final loss {} exceeds 1e-3 (measured range over 64 seeds: 2.22e-4 ..= 4.13e-4)",
            result.final_loss
        );

        // Second, independent failure mode. The loss is an average, so it can be
        // dragged under a threshold by three easy samples while the fourth is
        // inverted; thresholding the decision boundary per sample cannot be. A
        // NaN prediction fails here too, since every comparison against it is
        // false.
        for (xi, yi) in x.iter().zip(y.iter()) {
            let predicted = mlp.predict(xi).get(&[0, 0]);
            let expected = yi.get(&[0, 0]);
            assert_eq!(
                predicted > 0.5,
                expected > 0.5,
                "XOR({}, {}) misclassified: predicted {predicted}, expected {expected}",
                xi.get(&[0, 0]),
                xi.get(&[1, 0])
            );
        }
    }

    #[test]
    fn fit_reports_false_when_the_epoch_budget_runs_out() {
        // `converged` is only evidence if it can be `false`. Three budgets on
        // one problem: none at all, too few, and enough.
        let (x, y) = xor_problem();

        // At 50 epochs the net sits at a loss of 0.114, 437x its 500-epoch
        // value: still in the steep part of the curve. Every per-epoch
        // improvement before epoch 96 exceeds 2.653e-4 -- 2.6x
        // `CONVERGENCE_TOL` -- so the patience window cannot close here by
        // accident. Sweeping the budget 0..=500 puts the true boundary at 117,
        // and it never reverts to `false` after that, so 50 sits 2.3x clear of
        // it on one side and the 500 below sits 4.3x clear on the other.
        let short = xor_mlp().fit(&x, &y, 50);
        assert!(
            !short.converged,
            "50 epochs is not enough for XOR (final loss {}), yet fit reported convergence",
            short.final_loss
        );

        assert!(
            !xor_mlp().fit(&x, &y, 0).converged,
            "a run of zero epochs cannot have converged"
        );

        let long = xor_mlp().fit(&x, &y, 500);
        assert!(
            long.converged,
            "500 epochs drives XOR to a plateau (final loss {}), yet fit reported no convergence",
            long.final_loss
        );
    }

    #[test]
    fn test_mlp_large_scale() {
        // Fix 3.1: Verify we can have > 64 neurons
        let config = OptimizerConfig::Adam {
            learning_rate: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        };
        let mut mlp = MLP::new(config, LossConfig::BinaryCrossEntropy);

        // Input 100 -> Hidden 128 -> Output 10
        mlp.add_layer(100, 128, Activation::ReLU, Some(1));
        mlp.add_layer(128, 10, Activation::Softmax, Some(2));

        let input = Tensor::new(&vec![0.5; 100], &[100, 1]);
        let output = mlp.forward(&input);

        assert_eq!(output.shape, vec![10, 1]);
        assert!((output.sum() - 1.0).abs() < 1e-5);
    }
}
