// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Johnson–Lindenstrauss random projection kernel.
//!
//! Projects a batch of high-dimensional sparse vectors into 3-D
//! using a pseudo-random Gaussian matrix generated on-the-fly from a seed.
//!
//! Arguments:
//!   0: __global const double* input   — flattened [n_vectors][DIM_IN] array
//!   1: __global double* output        — flattened [n_vectors][3] array
//!   2: int n_vectors
//!   3: int dim_in                      (typically 128)
//!   4: int seed

inline uint lcg(uint* state) {
    *state = (*state * 1103515245u + 12345u);
    return *state;
}

inline double rand_gaussian_approx(uint* state) {
    uint u = lcg(state);
    double v = ((double)(u & 0x7FFFFFFF) / (double)0x7FFFFFFF) * 2.0 - 1.0;
    return v * 0.7978845608;
}

__kernel void jl_project(
    __global const double* input,
    __global double* output,
    int n_vectors,
    int dim_in,
    int seed
) {
    int gid = get_global_id(0);
    if (gid >= n_vectors) return;

    uint rng_state = (uint)(seed + gid * 73856093);

    double out0 = 0.0;
    double out1 = 0.0;
    double out2 = 0.0;

    int base = gid * dim_in;

    for (int d = 0; d < dim_in; d++) {
        double val = input[base + d];
        out0 += val * rand_gaussian_approx(&rng_state);
        out1 += val * rand_gaussian_approx(&rng_state);
        out2 += val * rand_gaussian_approx(&rng_state);
    }

    int out_base = gid * 3;
    output[out_base + 0] = out0;
    output[out_base + 1] = out1;
    output[out_base + 2] = out2;
}
