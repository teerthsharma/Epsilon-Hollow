// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Spherical Voronoi cell assignment kernel.
//!
//! Assigns each point on S² to the nearest centroid using great-circle distance.
//!
//! Arguments (packed in argument buffer):
//!   0: __global const float2* points   — (theta, phi) per point
//!   1: __global const float2* centroids — (theta, phi) per cell
//!   2: __global int* cell_ids           — output cell index per point
//!   3: int n_points
//!   4: int n_cells

typedef struct {
    float theta;
    float phi;
} SpherePoint;

__kernel void voronoi_assign(
    __global const SpherePoint* points,
    __global const SpherePoint* centroids,
    __global int* cell_ids,
    int n_points,
    int n_cells
) {
    int gid = get_global_id(0);
    if (gid >= n_points) return;

    float best_dist = 1e38f;
    int best_cell = 0;

    float p_theta = points[gid].theta;
    float p_phi   = points[gid].phi;

    float sin_pt = sin(p_theta);
    float cos_pt = cos(p_theta);

    for (int c = 0; c < n_cells; c++) {
        float ct = centroids[c].theta;
        float cp = centroids[c].phi;

        float cos_dist = sin_pt * sin(ct) + cos_pt * cos(ct) * cos(p_phi - cp);
        cos_dist = fmin(1.0f, fmax(-1.0f, cos_dist));
        float dist = acos(cos_dist);

        if (dist < best_dist) {
            best_dist = dist;
            best_cell = c;
        }
    }
    cell_ids[gid] = best_cell;
}
