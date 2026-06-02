// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Spectral contraction (relaxation) kernel.
//!
//! Computes one Euler step of spectral flow:
//!   x_{t+1}[i] = (1 - alpha) * x_t[i] + alpha * target[i]
//!
//! Arguments:
//!   0: __global const double* state   — current state vector x_t
//!   1: __global const double* target  — target attractor
//!   2: __global double* output        — next state vector x_{t+1}
//!   3: int dim
//!   4: double alpha

__kernel void spectral_step(
    __global const double* state,
    __global const double* target,
    __global double* output,
    int dim,
    double alpha
) {
    int gid = get_global_id(0);
    if (gid >= dim) return;

    double x = state[gid];
    double t = target[gid];
    double one_minus_alpha = 1.0 - alpha;

    output[gid] = one_minus_alpha * x + alpha * t;
}
