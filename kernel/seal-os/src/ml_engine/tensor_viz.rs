// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Tensor-to-manifold conversion — renders multi-dimensional data as 3D hyperbolic geometry.
//!
//! Trading data becomes geometry. Profit is green peaks. Loss is red valleys.

use alloc::vec::Vec;
use crate::graphics::topo_render::{BoundingBox, Camera, TopoMesh};
use crate::wm::window::Window;

/// A simple CPU-side tensor (f32 data + shape).
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// A TopoMesh augmented with tensor metadata.
pub struct TensorMesh {
    pub mesh: TopoMesh,
    pub value_range: (f32, f32),
    pub dimensions: Vec<usize>,
}

// ---------------------------------------------------------------------------
// CSV / Matrix parsing
// ---------------------------------------------------------------------------

/// Parse a CSV string into a 2-D tensor.
pub fn parse_csv(data: &str) -> Tensor {
    let mut values = Vec::new();
    let mut rows = 0usize;
    let mut cols = 0usize;
    for line in data.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut line_cols = 0usize;
        for token in line.split(',') {
            let token = token.trim();
            if let Ok(v) = token.parse::<f32>() {
                values.push(v);
                line_cols += 1;
            }
        }
        if line_cols > 0 {
            cols = cols.max(line_cols);
            rows += 1;
        }
    }
    // NOTE: we trust well-formed CSV; rectangular padding not performed.
    Tensor {
        shape: vec![rows, cols],
        data: values,
    }
}

// ---------------------------------------------------------------------------
// Tensor → Point Cloud
// ---------------------------------------------------------------------------

/// Convert a tensor into a 3-D point cloud.
///
/// * 1-D: X = index, Y = value, Z = 0
/// * 2-D: X = column, Z = row, Y = value
/// * 3-D+: X/Y/Z = first three dimension indices (value drives colour only)
pub fn tensor_to_point_cloud(tensor: &Tensor) -> Vec<[f32; 3]> {
    if tensor.shape.is_empty() || tensor.data.is_empty() {
        return Vec::new();
    }

    let total = tensor.data.len();
    let mut points = Vec::with_capacity(total);

    if tensor.shape.len() == 1 {
        let n = tensor.shape[0];
        let (min_v, max_v) = tensor_value_range(tensor);
        let range = if max_v > min_v { max_v - min_v } else { 1.0 };
        for i in 0..n {
            let x = if n > 1 {
                (2.0 * i as f32 / (n - 1) as f32) - 1.0
            } else {
                0.0
            };
            let y = 2.0 * (tensor.data[i] - min_v) / range - 1.0;
            points.push([x, y, 0.0]);
        }
    } else if tensor.shape.len() == 2 {
        let rows = tensor.shape[0];
        let cols = tensor.shape[1];
        let (min_v, max_v) = tensor_value_range(tensor);
        let range = if max_v > min_v { max_v - min_v } else { 1.0 };
        for i in 0..rows {
            for j in 0..cols {
                let x = if cols > 1 {
                    (2.0 * j as f32 / (cols - 1) as f32) - 1.0
                } else {
                    0.0
                };
                let z = if rows > 1 {
                    (2.0 * i as f32 / (rows - 1) as f32) - 1.0
                } else {
                    0.0
                };
                let y = 2.0 * (tensor.data[i * cols + j] - min_v) / range - 1.0;
                points.push([x, y, z]);
            }
        }
    } else {
        // Higher-D: first three dimension indices become X, Y, Z.
        let d0 = tensor.shape.get(0).copied().unwrap_or(1);
        let d1 = tensor.shape.get(1).copied().unwrap_or(1);
        let d2 = tensor.shape.get(2).copied().unwrap_or(1);
        let stride0 = tensor.shape[1..].iter().product::<usize>().max(1);
        let stride1 = tensor.shape[2..].iter().product::<usize>().max(1);
        let stride2 = tensor.shape[3..].iter().product::<usize>().max(1);

        for i in 0..total {
            let i0 = (i / stride0) % d0;
            let i1 = (i / stride1) % d1;
            let i2 = (i / stride2) % d2;
            let x = if d0 > 1 {
                (2.0 * i0 as f32 / (d0 - 1) as f32) - 1.0
            } else {
                0.0
            };
            let y = if d1 > 1 {
                (2.0 * i1 as f32 / (d1 - 1) as f32) - 1.0
            } else {
                0.0
            };
            let z = if d2 > 1 {
                (2.0 * i2 as f32 / (d2 - 1) as f32) - 1.0
            } else {
                0.0
            };
            points.push([x, y, z]);
        }
    }

    points
}

fn tensor_value_range(tensor: &Tensor) -> (f32, f32) {
    let mut min = f32::MAX;
    let mut max = f32::MIN;
    for &v in &tensor.data {
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
    }
    if min > max {
        min = 0.0;
        max = 1.0;
    }
    (min, max)
}

// ---------------------------------------------------------------------------
// Point Cloud → Mesh
// ---------------------------------------------------------------------------

/// Build a `TensorMesh` from a point cloud and per-point scalar values.
///
/// For grid-like data each cell becomes two triangles.  Vertex colours are
/// derived from `values` using the Lypnos Guard palette:
///   loss  → red,   zero → grey,   profit → green.
pub fn point_cloud_to_mesh(points: &[[f32; 3]], values: &[f32]) -> TensorMesh {
    let total = points.len();
    let (rows, cols) = find_grid_dimensions(total);
    point_cloud_to_mesh_grid(points, values, rows, cols)
}

/// Same as `point_cloud_to_mesh` but with explicit grid dimensions.
pub fn point_cloud_to_mesh_grid(
    points: &[[f32; 3]],
    values: &[f32],
    rows: usize,
    cols: usize,
) -> TensorMesh {
    let (min_val, max_val) = if values.is_empty() {
        (0.0f32, 1.0f32)
    } else {
        let mut min = f32::MAX;
        let mut max = f32::MIN;
        for &v in values {
            if v.is_nan() || v.is_infinite() {
                continue;
            }
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
        }
        (min, max)
    };

    let mut mesh = TopoMesh {
        vertices: Vec::new(),
        triangles: Vec::new(),
        normals: Vec::new(),
        spherical_embedding: Vec::new(),
        bbox: BoundingBox {
            min: [f32::MAX, f32::MAX, f32::MAX],
            max: [f32::MIN, f32::MIN, f32::MIN],
        },
        vertex_colors: Vec::new(),
    };

    // Copy vertices
    mesh.vertices.extend_from_slice(points);

    // Bounding box
    for v in points {
        for i in 0..3 {
            if v[i] < mesh.bbox.min[i] {
                mesh.bbox.min[i] = v[i];
            }
            if v[i] > mesh.bbox.max[i] {
                mesh.bbox.max[i] = v[i];
            }
        }
    }

    // Vertex colours
    let range = if max_val > min_val {
        max_val - min_val
    } else {
        1.0
    };
    for &v in values {
        let norm = 2.0 * (v - min_val) / range - 1.0;
        mesh.vertex_colors.push(value_to_color(norm));
    }
    while mesh.vertex_colors.len() < points.len() {
        mesh.vertex_colors.push(0x808080);
    }

    // Spherical embeddings (same deterministic pattern as topo_ram)
    for (idx, v) in points.iter().enumerate() {
        let mut emb = [0u16; 32];
        for a in 0..32 {
            emb[a] = ((idx.wrapping_mul(1103515245)
                .wrapping_add(12345)
                .wrapping_add(a.wrapping_mul(65537))
                .wrapping_add((v[0].abs() * 1000.0) as usize)
                .wrapping_add((v[1].abs() * 1000.0) as usize)
                .wrapping_add((v[2].abs() * 1000.0) as usize))
                % 65536) as u16;
        }
        mesh.spherical_embedding.push(emb);
    }

    // Grid triangulation
    if rows > 1 && cols > 1 && rows * cols == points.len() {
        for i in 0..rows - 1 {
            for j in 0..cols - 1 {
                let a = (i * cols + j) as u32;
                let b = (i * cols + j + 1) as u32;
                let c = ((i + 1) * cols + j) as u32;
                let d = ((i + 1) * cols + j + 1) as u32;
                mesh.triangles.push([a, b, c]);
                mesh.triangles.push([b, d, c]);
            }
        }
    }

    // Normals
    mesh.normals = compute_vertex_normals(&mesh.vertices, &mesh.triangles);

    TensorMesh {
        mesh,
        value_range: (min_val, max_val),
        dimensions: vec![rows, cols],
    }
}

fn find_grid_dimensions(n: usize) -> (usize, usize) {
    if n == 0 {
        return (0, 0);
    }
    let sqrt = libm::sqrtf(n as f32) as usize;
    for cols in (1..=sqrt).rev() {
        if n % cols == 0 {
            return (n / cols, cols);
        }
    }
    (n, 1)
}

fn value_to_color(normalized: f32) -> u32 {
    // normalized ∈ [−1, 1]
    if normalized < 0.0 {
        let t = (-normalized).min(1.0);
        let r = (0x88u32 + ((0xFF - 0x88) as f32 * t) as u32).min(0xFF);
        let g = ((0x80u32 as f32) * (1.0 - t)) as u32;
        let b = ((0x80u32 as f32) * (1.0 - t)) as u32;
        (r << 16) | (g << 8) | b
    } else if normalized > 0.0 {
        let t = normalized.min(1.0);
        let r = ((0x80u32 as f32) * (1.0 - t)) as u32;
        let g = (0x88u32 + ((0xFF - 0x88) as f32 * t) as u32).min(0xFF);
        let b = ((0x80u32 as f32) * (1.0 - t)) as u32;
        (r << 16) | (g << 8) | b
    } else {
        0x808080
    }
}

fn compute_vertex_normals(vertices: &[[f32; 3]], triangles: &[[u32; 3]]) -> Vec<[f32; 3]> {
    let mut normals = vec![[0.0f32, 0.0, 0.0]; vertices.len()];
    let mut counts = vec![0u32; vertices.len()];

    for tri in triangles {
        let v0 = vertices[tri[0] as usize];
        let v1 = vertices[tri[1] as usize];
        let v2 = vertices[tri[2] as usize];
        let e0 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
        let e1 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
        let face_normal = [
            e0[1] * e1[2] - e0[2] * e1[1],
            e0[2] * e1[0] - e0[0] * e1[2],
            e0[0] * e1[1] - e0[1] * e1[0],
        ];
        let len = libm::sqrtf(
            face_normal[0] * face_normal[0]
                + face_normal[1] * face_normal[1]
                + face_normal[2] * face_normal[2],
        );
        let face_normal = if len > 0.0 {
            [
                face_normal[0] / len,
                face_normal[1] / len,
                face_normal[2] / len,
            ]
        } else {
            [0.0, 0.0, 1.0]
        };

        for i in 0..3 {
            let idx = tri[i] as usize;
            normals[idx][0] += face_normal[0];
            normals[idx][1] += face_normal[1];
            normals[idx][2] += face_normal[2];
            counts[idx] += 1;
        }
    }

    for i in 0..normals.len() {
        if counts[i] > 0 {
            let n = [
                normals[i][0] / counts[i] as f32,
                normals[i][1] / counts[i] as f32,
                normals[i][2] / counts[i] as f32,
            ];
            let len = libm::sqrtf(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
            if len > 0.0 {
                normals[i] = [n[0] / len, n[1] / len, n[2] / len];
            }
        }
    }

    normals
}

// ---------------------------------------------------------------------------
// High-level render
// ---------------------------------------------------------------------------

/// Render a tensor directly to a window.
pub fn render_tensor(tensor: &Tensor, camera: &Camera, target: &mut Window) {
    let points = tensor_to_point_cloud(tensor);
    if points.is_empty() {
        return;
    }
    let tensor_mesh = if tensor.shape.len() == 2 {
        point_cloud_to_mesh_grid(&points, &tensor.data, tensor.shape[0], tensor.shape[1])
    } else {
        point_cloud_to_mesh(&points, &tensor.data)
    };
    crate::graphics::topo_render::set_camera(*camera);
    crate::graphics::topo_render::render_mesh(&tensor_mesh.mesh, target);
}
