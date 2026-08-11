// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Tensor-to-manifold conversion — renders multi-dimensional data as 3D hyperbolic geometry.
//!
//! Trading data becomes geometry. Profit is green peaks. Loss is red valleys.

use crate::graphics::topo_render::{BoundingBox, Camera, TopoMesh};
use crate::wm::window::Window;
use alloc::vec::Vec;

/// A simple CPU-side tensor (f32 data + shape).
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl Tensor {
    /// A tensor with no shape and no data. Every consumer already treats an
    /// empty buffer as "nothing to draw", so this is the fail-closed result for
    /// input that cannot be read as a rectangle.
    pub fn empty() -> Self {
        Tensor {
            shape: Vec::new(),
            data: Vec::new(),
        }
    }

    /// True when `shape` describes exactly the buffer that backs it.
    ///
    /// Every index in this module that is computed from `shape` rather than
    /// from `data.len()` is only sound while this holds, so it gates the one
    /// entry point those indices live behind. The product is computed with
    /// `checked_mul` so an overflowing shape cannot wrap into a value the
    /// buffer appears to satisfy.
    pub fn is_rectangular(&self) -> bool {
        !self.shape.is_empty()
            && self
                .shape
                .iter()
                .try_fold(1usize, |acc, &d| acc.checked_mul(d))
                == Some(self.data.len())
    }
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
///
/// The file content is attacker-chosen, so the shape is never widened past what
/// the buffer actually holds. A row whose numeric field count disagrees with the
/// first data row makes the whole file unreadable as a matrix, and the result is
/// `Tensor::empty()` rather than a rectangle the buffer cannot fill. Padding the
/// gap with zeros was rejected: `tensor info` reports min/max/mean and the
/// renderer paints zero as break-even grey, so invented cells would be read as
/// data. Refusing renders nothing, which is honest.
///
/// Lines with no numeric field at all — a text header, a blank line — are skipped
/// and take no part in the row count, as before.
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
        if line_cols == 0 {
            continue;
        }
        if rows == 0 {
            cols = line_cols;
        } else if line_cols != cols {
            return Tensor::empty();
        }
        rows += 1;
    }
    if rows == 0 || cols == 0 {
        return Tensor::empty();
    }
    // `rows * cols == values.len()` by construction: every counted row pushed
    // exactly `cols` values.
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
///
/// A tensor whose shape does not match its buffer draws nothing: `Tensor` has
/// public fields, so `parse_csv` is not the only way one can be built.
pub fn tensor_to_point_cloud(tensor: &Tensor) -> Vec<[f32; 3]> {
    if tensor.data.is_empty() || !tensor.is_rectangular() {
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
        let d0 = tensor.shape.first().copied().unwrap_or(1);
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
            emb[a] = ((idx
                .wrapping_mul(1103515245)
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

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(any(test, feature = "test-mode"))]
pub mod tests {
    use super::*;
    use crate::testing::TestResult;
    use crate::{test_assert, test_assert_eq};

    /// Shape and buffer agree: the shape describes the buffer exactly, or there
    /// is no shape and no buffer.
    fn agrees(t: &Tensor) -> bool {
        t.is_rectangular() || (t.shape.is_empty() && t.data.is_empty())
    }

    /// The named defect: `1,2,3\n4,5\n` used to give shape [2, 3] over a 5-value
    /// buffer, so rendering read `data[5]`. Under `panic = "abort"` that is the
    /// machine, and the file content is entirely attacker-chosen.
    fn test_ragged_csv_refused() -> TestResult {
        let t = parse_csv("1,2,3\n4,5\n");
        test_assert!(agrees(&t), "ragged CSV produced a tensor that lies");
        test_assert!(t.data.is_empty(), "ragged CSV must be refused, not padded");
        test_assert!(tensor_to_point_cloud(&t).is_empty());
        TestResult::Pass
    }

    /// Same defect through a different door: one non-numeric field shortens a row.
    fn test_non_numeric_field_refused() -> TestResult {
        let t = parse_csv("1,abc,3\n4,5,6\n");
        test_assert!(agrees(&t));
        test_assert!(t.data.is_empty());
        let grown = parse_csv("1\n2,3\n4,5,6\n");
        test_assert!(agrees(&grown));
        test_assert!(grown.data.is_empty());
        TestResult::Pass
    }

    /// A rectangular file is unchanged, header row still skipped.
    fn test_well_formed_csv_unchanged() -> TestResult {
        let t = parse_csv("1,2,3\n4,5,6\n");
        test_assert_eq!(t.shape.len(), 2);
        test_assert_eq!(t.shape[0], 2);
        test_assert_eq!(t.shape[1], 3);
        test_assert_eq!(t.data.len(), 6);
        test_assert_eq!(tensor_to_point_cloud(&t).len(), 6);
        let headed = parse_csv("date,open,close\n1,2,3\n4,5,6\n");
        test_assert_eq!(headed.data.len(), 6);
        test_assert_eq!(headed.shape[1], 3);
        TestResult::Pass
    }

    /// Degenerate end of the same defect: nothing numeric anywhere.
    fn test_degenerate_inputs_empty() -> TestResult {
        for src in ["", "\n\n\n", "   \n\t\n", "a,b\nc,d\n", ",,,\n"] {
            let t = parse_csv(src);
            test_assert!(agrees(&t), "degenerate input produced a tensor that lies");
            test_assert!(tensor_to_point_cloud(&t).is_empty());
        }
        TestResult::Pass
    }

    /// `Tensor` has public fields, so a shape that outruns its buffer can still
    /// be built by hand. Every index computed from `shape` sits behind this gate.
    fn test_overlong_shape_draws_nothing() -> TestResult {
        let one_d = Tensor {
            shape: vec![5],
            data: vec![1.0, 2.0],
        };
        test_assert!(!one_d.is_rectangular());
        test_assert!(tensor_to_point_cloud(&one_d).is_empty());

        let two_d = Tensor {
            shape: vec![3, 4],
            data: vec![1.0, 2.0, 3.0],
        };
        test_assert!(!two_d.is_rectangular());
        test_assert!(tensor_to_point_cloud(&two_d).is_empty());

        // An overflowing product must not wrap into a value the buffer satisfies.
        let overflow = Tensor {
            shape: vec![usize::MAX, 4, 4],
            data: vec![1.0, 2.0, 3.0, 4.0],
        };
        test_assert!(!overflow.is_rectangular());
        test_assert!(tensor_to_point_cloud(&overflow).is_empty());
        TestResult::Pass
    }

    /// Guards the `run_tensor_render_bench` gate in lib.rs against this change.
    fn test_bench_fixture_shape() -> TestResult {
        let mut csv = alloc::string::String::new();
        for row in 0..32usize {
            for col in 0..32usize {
                csv.push((b'0' + ((row + col) % 10) as u8) as char);
                if col + 1 < 32 {
                    csv.push(',');
                }
            }
            csv.push('\n');
        }
        let t = parse_csv(&csv);
        test_assert_eq!(t.shape[0], 32);
        test_assert_eq!(t.shape[1], 32);
        test_assert_eq!(t.data.len(), 1024);
        let points = tensor_to_point_cloud(&t);
        test_assert_eq!(points.len(), 1024);
        let mesh = point_cloud_to_mesh_grid(&points, &t.data, 32, 32);
        test_assert_eq!(mesh.mesh.triangles.len(), 31 * 31 * 2);
        TestResult::Pass
    }

    pub fn register_all() {
        crate::testing::register_test("tensor_viz::ragged_csv_refused", test_ragged_csv_refused);
        crate::testing::register_test(
            "tensor_viz::non_numeric_field_refused",
            test_non_numeric_field_refused,
        );
        crate::testing::register_test(
            "tensor_viz::well_formed_csv_unchanged",
            test_well_formed_csv_unchanged,
        );
        crate::testing::register_test(
            "tensor_viz::degenerate_inputs_empty",
            test_degenerate_inputs_empty,
        );
        crate::testing::register_test(
            "tensor_viz::overlong_shape_draws_nothing",
            test_overlong_shape_draws_nothing,
        );
        crate::testing::register_test("tensor_viz::bench_fixture_shape", test_bench_fixture_shape);
    }
}
