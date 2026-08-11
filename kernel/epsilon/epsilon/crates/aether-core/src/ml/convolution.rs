// Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: Epsilon-Hollow

//! ═══════════════════════════════════════════════════════════════════════════════
//! AEGIS Convolutional Layers
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! 2D Convolution implementation for image processing and spatial pattern recognition.
//!
//! ═══════════════════════════════════════════════════════════════════════════════

#![allow(rustdoc::broken_intra_doc_links)]

use crate::ml::neural::Activation;
use crate::ml::rng::Lcg;
use libm::sqrt;

/// Maximum kernel size (e.g., 3x3, 5x5)
const MAX_KERNEL_SIZE: usize = 5;
/// Maximum input channels
const MAX_CHANNELS_IN: usize = 3;
/// Maximum output channels (filters)
const MAX_CHANNELS_OUT: usize = 8;
/// Maximum image dimension
const MAX_IMG_DIM: usize = 32;

/// 2D Convolutional Layer
#[derive(Debug, Clone)]
pub struct Conv2D {
    /// Filters [out_channel][in_channel][k_y][k_x]
    pub weights: [[[[f64; MAX_KERNEL_SIZE]; MAX_KERNEL_SIZE]; MAX_CHANNELS_IN]; MAX_CHANNELS_OUT],
    /// Biases [out_channel]
    pub biases: [f64; MAX_CHANNELS_OUT],
    /// Input channels
    pub in_channels: usize,
    /// Output channels
    pub out_channels: usize,
    /// Kernel size (k x k)
    pub kernel_size: usize,
    /// Stride
    pub stride: usize,
    /// Padding
    pub padding: usize,
    /// Activation
    pub activation: Activation,

    // Cache for backprop
    pub last_input: [[[[f64; MAX_IMG_DIM]; MAX_IMG_DIM]; MAX_CHANNELS_IN]; 1], // Batch size 1 for now
    pub last_output_dim: (usize, usize),
}

impl Conv2D {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        activation: Activation,
        seed: Option<u64>,
    ) -> Self {
        let in_c = in_channels.min(MAX_CHANNELS_IN);
        let out_c = out_channels.min(MAX_CHANNELS_OUT);
        let k_size = kernel_size.min(MAX_KERNEL_SIZE);

        // Kaiming/He initialization
        let scale = sqrt(2.0 / (in_c * k_size * k_size) as f64);

        let mut weights =
            [[[[0.0; MAX_KERNEL_SIZE]; MAX_KERNEL_SIZE]; MAX_CHANNELS_IN]; MAX_CHANNELS_OUT];
        let mut rng = Lcg::new(seed.unwrap_or(42));

        for channel_out in weights.iter_mut().take(out_c) {
            for channel_in in channel_out.iter_mut().take(in_c) {
                for row in channel_in.iter_mut().take(k_size) {
                    for val in row.iter_mut().take(k_size) {
                        *val = rng.next_signed_f64() * scale;
                    }
                }
            }
        }

        Self {
            weights,
            biases: [0.0; MAX_CHANNELS_OUT],
            in_channels: in_c,
            out_channels: out_c,
            kernel_size: k_size,
            stride,
            padding,
            activation,
            last_input: [[[[0.0; MAX_IMG_DIM]; MAX_IMG_DIM]; MAX_CHANNELS_IN]; 1],
            last_output_dim: (0, 0),
        }
    }

    /// Forward pass
    /// input: [channels][height][width]
    pub fn forward(
        &mut self,
        input: &[[[f64; MAX_IMG_DIM]; MAX_IMG_DIM]; MAX_CHANNELS_IN],
        input_h: usize,
        input_w: usize,
    ) -> (
        [[[f64; MAX_IMG_DIM]; MAX_IMG_DIM]; MAX_CHANNELS_OUT],
        usize,
        usize,
    ) {
        // Cache input
        self.last_input[0] = *input;

        let output_h = (input_h + 2 * self.padding - self.kernel_size) / self.stride + 1;
        let output_w = (input_w + 2 * self.padding - self.kernel_size) / self.stride + 1;
        self.last_output_dim = (output_h, output_w);

        let mut output = [[[0.0; MAX_IMG_DIM]; MAX_IMG_DIM]; MAX_CHANNELS_OUT];

        for (o, channel_out) in output.iter_mut().enumerate().take(self.out_channels) {
            for (y, row_out) in channel_out.iter_mut().enumerate().take(output_h) {
                for (x, val_out) in row_out.iter_mut().enumerate().take(output_w) {
                    let mut sum = self.biases[o];

                    // Convolve
                    let in_y_origin = (y * self.stride) as isize - self.padding as isize;
                    let in_x_origin = (x * self.stride) as isize - self.padding as isize;

                    for (c, input_channel) in input.iter().enumerate().take(self.in_channels) {
                        for ky in 0..self.kernel_size {
                            for kx in 0..self.kernel_size {
                                let in_y = in_y_origin + ky as isize;
                                let in_x = in_x_origin + kx as isize;

                                if in_y >= 0
                                    && in_y < input_h as isize
                                    && in_x >= 0
                                    && in_x < input_w as isize
                                {
                                    sum += input_channel[in_y as usize][in_x as usize]
                                        * self.weights[o][c][ky][kx];
                                }
                            }
                        }
                    }

                    // Activation
                    *val_out = self.activation.apply_scalar(sum);
                }
            }
        }

        (output, output_h, output_w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv2d_initialization() {
        let conv = Conv2D::new(1, 1, 3, 1, 1, Activation::ReLU, None);
        assert_eq!(conv.weights.len(), MAX_CHANNELS_OUT); // Array size fixed
                                                          // Check params
        assert_eq!(conv.kernel_size, 3);
        assert_eq!(conv.stride, 1);
        assert_eq!(conv.padding, 1);
    }

    #[test]
    fn test_conv2d_forward_shape() {
        let mut conv = Conv2D::new(1, 1, 3, 1, 1, Activation::ReLU, None);
        let input = [[[0.5; MAX_IMG_DIM]; MAX_IMG_DIM]; MAX_CHANNELS_IN];

        // 10x10 input
        // Output size: (10 + 2*1 - 3) / 1 + 1 = 10
        let (_, h, w) = conv.forward(&input, 10, 10);

        assert_eq!(h, 10);
        assert_eq!(w, 10);
    }

    #[test]
    fn different_seeds_produce_different_weights() {
        // Pre-fix, the seed was hardcoded to `42u64` with no parameter to
        // vary it, so two Conv2D layers were always initialised identically.
        // A network with two conv layers needs them to start decorrelated.
        let a = Conv2D::new(1, 1, 3, 1, 1, Activation::ReLU, Some(1));
        let b = Conv2D::new(1, 1, 3, 1, 1, Activation::ReLU, Some(2));
        assert_ne!(
            a.weights, b.weights,
            "different seeds must produce different initial weights"
        );
    }

    #[test]
    fn none_seed_delegates_to_42() {
        // Pins only that `None` routes to seed 42 under the current generator.
        // It does NOT establish compatibility with the pre-`Lcg` stream — that
        // used increment `1` and a full-width `u64 as f64` cast, so the weights
        // this constructor produces did change. See the note on
        // `Tensor::unseeded_entry_point_delegates_to_seed_42`.
        let default = Conv2D::new(1, 1, 3, 1, 1, Activation::ReLU, None);
        let explicit_42 = Conv2D::new(1, 1, 3, 1, 1, Activation::ReLU, Some(42));
        assert_eq!(default.weights, explicit_42.weights);
    }
}
