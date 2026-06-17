// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Topological 3D Render Driver — software rasterizer driven by T1–T5 theorems.
//!
//! Implements a full software 3D pipeline: hyperbolic projection (T5), spectral LOD (T2),
//! Betti mesh integrity (T3), Voronoi spatial partition (T1), and adaptive quality governor (T4).

use crate::graphics::htek;
use crate::wm::window::Window;
use alloc::vec::Vec;

/// Per-vertex manifold embedding (16 points on S², quantized)
pub struct TopoMesh {
    pub vertices: Vec<[f32; 3]>,
    pub triangles: Vec<[u32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub spherical_embedding: Vec<[u16; 32]>,
    pub bbox: BoundingBox,
    pub vertex_colors: Vec<u32>,
}

pub struct BoundingBox {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

#[derive(Clone, Copy)]
pub struct Camera {
    pub position: [f32; 3],
    pub look_at: [f32; 3],
    pub up: [f32; 3],
    pub fov_deg: f32,
    pub near: f32,
    pub far: f32,
}

pub struct RenderState {
    pub camera: Camera,
    pub last_frame_ms: u32,
    pub quality_level: u8,
    pub frame_time_history: [u32; 5],
    pub frame_time_idx: usize,
    pub prev_view_dir: Option<[f32; 3]>,
    pub prev_lod_triangles: Option<Vec<[u32; 3]>>,
}

/// Screen-space Voronoi cell (8×8 grid = 64 cells)
pub struct VoronoiCell {
    pub x: u16,
    pub y: u16,
    pub w: u16,
    pub h: u16,
    pub triangle_indices: Vec<usize>,
}

static RENDER_STATE: spin::Mutex<Option<RenderState>> = spin::Mutex::new(None);

pub fn with_render_state<F, R>(f: F) -> Option<R>
where
    F: FnOnce(&mut RenderState) -> R,
{
    RENDER_STATE.lock().as_mut().map(f)
}

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

fn vec3_sub(a: &[f32; 3], b: &[f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn vec3_dot(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn vec3_cross(a: &[f32; 3], b: &[f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn vec3_normalize(v: &[f32; 3]) -> [f32; 3] {
    let len = libm::sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if len > 0.0 {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        [0.0, 0.0, 0.0]
    }
}

fn edge_fn(v0: &[f32; 4], v1: &[f32; 4], p: &[f32; 2]) -> f32 {
    (v1[0] - v0[0]) * (p[1] - v0[1]) - (v1[1] - v0[1]) * (p[0] - v0[0])
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

pub fn init() {
    let state = RenderState {
        camera: Camera {
            position: [0.0, 0.0, 5.0],
            look_at: [0.0, 0.0, 0.0],
            up: [0.0, 1.0, 0.0],
            fov_deg: 60.0,
            near: 0.1,
            far: 100.0,
        },
        last_frame_ms: 0,
        quality_level: 2,
        frame_time_history: [16; 5],
        frame_time_idx: 0,
        prev_view_dir: None,
        prev_lod_triangles: None,
    };
    *RENDER_STATE.lock() = Some(state);
}

pub fn set_camera(cam: Camera) {
    with_render_state(|state| {
        state.camera = cam;
    });
}

pub fn adaptive_quality(frame_ms: u32) {
    with_render_state(|state| {
        apply_adaptive_quality(state, frame_ms);
    });
}

pub fn project_vertex(
    v: &[f32; 3],
    cam: &Camera,
    screen_w: u32,
    screen_h: u32,
) -> Option<[f32; 4]> {
    hyperbolic_project(v, cam, screen_w, screen_h)
}

pub fn render_mesh(mesh: &TopoMesh, target: &mut Window) {
    let frame_start = crate::drivers::interrupts::ticks();

    let mut guard = RENDER_STATE.lock();
    let state = match guard.as_mut() {
        Some(s) => s,
        None => return,
    };

    let cw = target.client_width();
    let ch = target.client_height();
    if cw == 0 || ch == 0 {
        return;
    }

    // -----------------------------------------------------------------------
    // 1. T5 — Hyperbolic projection of all vertices
    // -----------------------------------------------------------------------
    let mut projected: Vec<[f32; 4]> = Vec::with_capacity(mesh.vertices.len());
    for v in &mesh.vertices {
        if let Some(p) = hyperbolic_project(v, &state.camera, cw, ch) {
            projected.push(p);
        } else {
            projected.push([f32::NAN, f32::NAN, f32::NAN, f32::NAN]);
        }
    }

    // -----------------------------------------------------------------------
    // 2. T2 — Spectral LOD
    // -----------------------------------------------------------------------
    let view_dir = vec3_normalize(&vec3_sub(&state.camera.look_at, &state.camera.position));
    let reuse_lod = if let Some(prev_dir) = state.prev_view_dir {
        vec3_dot(&view_dir, &prev_dir) > 0.95 && state.prev_lod_triangles.is_some()
    } else {
        false
    };

    let lod_triangles = if reuse_lod {
        if let Some(prev) = &state.prev_lod_triangles {
            prev.clone()
        } else {
            compute_lod(mesh, &state.camera, state.quality_level)
        }
    } else {
        compute_lod(mesh, &state.camera, state.quality_level)
    };

    // -----------------------------------------------------------------------
    // 3. T3 — Betti mesh integrity
    // -----------------------------------------------------------------------
    let final_triangles = if !lod_triangles.is_empty() && betti_one_check(&lod_triangles) {
        lod_triangles
    } else if let Some(prev) = &state.prev_lod_triangles {
        prev.clone()
    } else {
        mesh.triangles.clone()
    };

    state.prev_view_dir = Some(view_dir);
    state.prev_lod_triangles = Some(final_triangles.clone());

    // Backface cull after projection
    let mut visible_triangles: Vec<[u32; 3]> = Vec::with_capacity(final_triangles.len());
    for tri in &final_triangles {
        let i0 = tri[0] as usize;
        let i1 = tri[1] as usize;
        let i2 = tri[2] as usize;
        if projected[i0][0].is_nan() || projected[i1][0].is_nan() || projected[i2][0].is_nan() {
            continue;
        }
        if backface_cull_hyperbolic(&projected[i0], &projected[i1], &projected[i2]) {
            continue;
        }
        visible_triangles.push(*tri);
    }

    let quality = state.quality_level;

    // -----------------------------------------------------------------------
    // 4 & 5 — T1 Voronoi cells + rasterization
    // -----------------------------------------------------------------------
    if quality == 0 {
        // Wireframe only
        for tri in &visible_triangles {
            let p0 = projected[tri[0] as usize];
            let p1 = projected[tri[1] as usize];
            let p2 = projected[tri[2] as usize];
            htek::draw_aa_line(
                target,
                p0[0] as i32,
                p0[1] as i32,
                p1[0] as i32,
                p1[1] as i32,
                0xFFFFFF,
                255,
            );
            htek::draw_aa_line(
                target,
                p1[0] as i32,
                p1[1] as i32,
                p2[0] as i32,
                p2[1] as i32,
                0xFFFFFF,
                255,
            );
            htek::draw_aa_line(
                target,
                p2[0] as i32,
                p2[1] as i32,
                p0[0] as i32,
                p0[1] as i32,
                0xFFFFFF,
                255,
            );
        }
    } else {
        let mut depth_buffer = vec![f32::MAX; (cw * ch) as usize];
        let light = vec3_normalize(&[0.3, -0.5, -0.8]);

        if quality >= 2 {
            let mut cells = build_voronoi_cells(cw, ch);
            let cell_w = cw / 8;
            let cell_h = ch / 8;

            for (tri_idx, tri) in visible_triangles.iter().enumerate() {
                let p0 = projected[tri[0] as usize];
                let p1 = projected[tri[1] as usize];
                let p2 = projected[tri[2] as usize];
                let min_x = p0[0].min(p1[0]).min(p2[0]);
                let min_y = p0[1].min(p1[1]).min(p2[1]);
                let max_x = p0[0].max(p1[0]).max(p2[0]);
                let max_y = p0[1].max(p1[1]).max(p2[1]);

                let c0 = (min_x as u32 / cell_w).min(7) as usize;
                let c1 = (max_x as u32 / cell_w).min(7) as usize;
                let r0 = (min_y as u32 / cell_h).min(7) as usize;
                let r1 = (max_y as u32 / cell_h).min(7) as usize;

                for row in r0..=r1 {
                    for col in c0..=c1 {
                        let idx = row * 8 + col;
                        cells[idx].triangle_indices.push(tri_idx);
                    }
                }
            }

            for cell in &cells {
                let cx0 = cell.x as i32;
                let cy0 = cell.y as i32;
                let cx1 = (cell.x as i32 + cell.w as i32 - 1).min(cw as i32 - 1);
                let cy1 = (cell.y as i32 + cell.h as i32 - 1).min(ch as i32 - 1);
                if cx0 > cx1 || cy0 > cy1 {
                    continue;
                }
                for &tri_idx in &cell.triangle_indices {
                    let tri = visible_triangles[tri_idx];
                    rasterize_triangle(
                        target,
                        &projected,
                        mesh,
                        tri,
                        &light,
                        quality,
                        &mut depth_buffer,
                        cx0,
                        cy0,
                        cx1,
                        cy1,
                    );
                }
            }
        } else {
            // Quality 1: flat shading, no cells
            for tri in &visible_triangles {
                let p0 = projected[tri[0] as usize];
                let p1 = projected[tri[1] as usize];
                let p2 = projected[tri[2] as usize];
                let min_x = p0[0].min(p1[0]).min(p2[0]).max(0.0) as i32;
                let min_y = p0[1].min(p1[1]).min(p2[1]).max(0.0) as i32;
                let max_x = p0[0].max(p1[0]).max(p2[0]).min(cw as f32 - 1.0) as i32;
                let max_y = p0[1].max(p1[1]).max(p2[1]).min(ch as f32 - 1.0) as i32;
                rasterize_triangle(
                    target,
                    &projected,
                    mesh,
                    *tri,
                    &light,
                    quality,
                    &mut depth_buffer,
                    min_x,
                    min_y,
                    max_x,
                    max_y,
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // 6. T4 — Governor: adaptive quality for next frame
    // -----------------------------------------------------------------------
    let frame_end = crate::drivers::interrupts::ticks();
    let frame_ms = frame_end.wrapping_sub(frame_start) as u32;
    apply_adaptive_quality(state, frame_ms);
}

// ---------------------------------------------------------------------------
// T5 — Hyperbolic projection
// ---------------------------------------------------------------------------

pub fn hyperbolic_project(
    v: &[f32; 3],
    cam: &Camera,
    screen_w: u32,
    screen_h: u32,
) -> Option<[f32; 4]> {
    let fwd = vec3_normalize(&vec3_sub(&cam.look_at, &cam.position));
    let right = vec3_normalize(&vec3_cross(&fwd, &cam.up));
    let up = vec3_cross(&right, &fwd);

    let to_v = vec3_sub(v, &cam.position);
    let x_cam = vec3_dot(&to_v, &right);
    let y_cam = vec3_dot(&to_v, &up);
    let z_cam = vec3_dot(&to_v, &fwd);

    if z_cam <= cam.near {
        return None;
    }

    let fov_rad = cam.fov_deg * (core::f32::consts::PI / 180.0);
    let sinh_fov = libm::sinhf(fov_rad);
    let cosh_fov = libm::coshf(fov_rad);

    let denom = cosh_fov - z_cam / cam.far;
    if denom <= 0.0 {
        return None;
    }

    let x_proj = x_cam * sinh_fov / denom;
    let y_proj = y_cam * sinh_fov / denom;

    let scale_x = screen_w as f32 / (2.0 * sinh_fov);
    let scale_y = screen_h as f32 / (2.0 * sinh_fov);

    let sx = screen_w as f32 * 0.5 + x_proj * scale_x;
    let sy = screen_h as f32 * 0.5 - y_proj * scale_y;

    Some([sx, sy, z_cam, 1.0 / z_cam])
}

/// Backface cull in hyperbolic screen space.
///
/// Uses the standard 2-D cross product after projection.  The hyperbolic
/// intent is documented: distortion is already baked into the vertex
/// coordinates, so the sign test remains valid for the projected triangle.
pub fn backface_cull_hyperbolic(v0: &[f32; 4], v1: &[f32; 4], v2: &[f32; 4]) -> bool {
    let ax = v1[0] - v0[0];
    let ay = v1[1] - v0[1];
    let bx = v2[0] - v0[0];
    let by = v2[1] - v0[1];
    let cross = ax * by - ay * bx;
    cross <= 0.0
}

// ---------------------------------------------------------------------------
// T1 — Voronoi spatial partition
// ---------------------------------------------------------------------------

fn build_voronoi_cells(screen_w: u32, screen_h: u32) -> [VoronoiCell; 64] {
    let cell_w = screen_w / 8;
    let cell_h = screen_h / 8;
    core::array::from_fn(|idx| {
        let row = idx / 8;
        let col = idx % 8;
        VoronoiCell {
            x: (col as u32 * cell_w) as u16,
            y: (row as u32 * cell_h) as u16,
            w: cell_w as u16,
            h: cell_h as u16,
            triangle_indices: Vec::new(),
        }
    })
}

// ---------------------------------------------------------------------------
// T2 — Spectral LOD
// ---------------------------------------------------------------------------

fn compute_lod(mesh: &TopoMesh, camera: &Camera, quality: u8) -> Vec<[u32; 3]> {
    let threshold = match quality {
        0 => 0.0,
        1 => 25.0,
        2 => 10.0,
        3 => 4.0,
        4 => 1.0,
        _ => 0.0,
    };

    // Vertex degree centrality
    let mut degrees = vec![0u32; mesh.vertices.len()];
    for tri in &mesh.triangles {
        degrees[tri[0] as usize] += 1;
        degrees[tri[1] as usize] += 1;
        degrees[tri[2] as usize] += 1;
    }
    let max_degree = degrees.iter().copied().max().unwrap_or(1).max(1);

    // Camera basis for approximate screen-space area
    let fwd = vec3_normalize(&vec3_sub(&camera.look_at, &camera.position));
    let right = vec3_normalize(&vec3_cross(&fwd, &camera.up));
    let up = vec3_cross(&right, &fwd);

    let mut result = Vec::with_capacity(mesh.triangles.len());
    for tri in &mesh.triangles {
        let v0 = mesh.vertices[tri[0] as usize];
        let v1 = mesh.vertices[tri[1] as usize];
        let v2 = mesh.vertices[tri[2] as usize];

        let t0 = vec3_sub(&v0, &camera.position);
        let t1 = vec3_sub(&v1, &camera.position);
        let t2 = vec3_sub(&v2, &camera.position);

        let x0 = vec3_dot(&t0, &right);
        let y0 = vec3_dot(&t0, &up);
        let x1 = vec3_dot(&t1, &right);
        let y1 = vec3_dot(&t1, &up);
        let x2 = vec3_dot(&t2, &right);
        let y2 = vec3_dot(&t2, &up);

        let area = ((x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)).abs() * 0.5;
        // LOD thresholds are expressed in approximate screen-pixel units, not
        // camera-plane square units. 4096 is a conservative 64x64 reference
        // surface that keeps quality 2 from collapsing dense proof fixtures.
        let screen_area_estimate = area * 4096.0;

        let avg_deg =
            (degrees[tri[0] as usize] + degrees[tri[1] as usize] + degrees[tri[2] as usize]) as f32
                / (3.0 * max_degree as f32);
        let importance = screen_area_estimate * (1.0 + avg_deg);

        if importance >= threshold || threshold == 0.0 {
            result.push(*tri);
        }
    }

    result
}

// ---------------------------------------------------------------------------
// T3 — Betti mesh integrity
// ---------------------------------------------------------------------------

fn betti_one_check(triangles: &[[u32; 3]]) -> bool {
    // Simplified manifold check: no edge may be shared by more than 2 triangles.
    let mut edges: Vec<((u32, u32), u32)> = Vec::new();
    for tri in triangles {
        let e0 = if tri[0] < tri[1] {
            (tri[0], tri[1])
        } else {
            (tri[1], tri[0])
        };
        let e1 = if tri[1] < tri[2] {
            (tri[1], tri[2])
        } else {
            (tri[2], tri[1])
        };
        let e2 = if tri[2] < tri[0] {
            (tri[2], tri[0])
        } else {
            (tri[0], tri[2])
        };
        for e in [e0, e1, e2] {
            let mut found = false;
            for (edge, count) in edges.iter_mut() {
                if *edge == e {
                    *count += 1;
                    found = true;
                    break;
                }
            }
            if !found {
                edges.push((e, 1));
            }
        }
    }
    for &(_, count) in &edges {
        if count > 2 {
            return false;
        }
    }
    true
}

// ---------------------------------------------------------------------------
// T4 — Governor (adaptive quality)
// ---------------------------------------------------------------------------

fn apply_adaptive_quality(state: &mut RenderState, frame_ms: u32) {
    state.frame_time_history[state.frame_time_idx] = frame_ms;
    state.frame_time_idx = (state.frame_time_idx + 1) % 5;
    let avg = state.frame_time_history.iter().sum::<u32>() / 5;
    if avg > 20 && state.quality_level > 0 {
        state.quality_level -= 1;
    } else if avg < 12 && state.quality_level < 4 {
        if state.frame_time_history.iter().all(|&t| t < 12) {
            state.quality_level += 1;
        }
    }
    state.last_frame_ms = frame_ms;
}

// ---------------------------------------------------------------------------
// Rasterization
// ---------------------------------------------------------------------------

fn rasterize_triangle(
    target: &mut Window,
    projected: &[[f32; 4]],
    mesh: &TopoMesh,
    tri: [u32; 3],
    light: &[f32; 3],
    quality: u8,
    depth_buffer: &mut [f32],
    bbox_x0: i32,
    bbox_y0: i32,
    bbox_x1: i32,
    bbox_y1: i32,
) {
    let p0 = projected[tri[0] as usize];
    let p1 = projected[tri[1] as usize];
    let p2 = projected[tri[2] as usize];

    let cw = target.client_width();

    // Face normal and flat-shading colour
    let v0 = mesh.vertices[tri[0] as usize];
    let v1 = mesh.vertices[tri[1] as usize];
    let v2 = mesh.vertices[tri[2] as usize];
    let e0 = vec3_sub(&v1, &v0);
    let e1 = vec3_sub(&v2, &v0);
    let face_normal = vec3_normalize(&vec3_cross(&e0, &e1));
    let face_brightness = vec3_dot(&face_normal, light).max(0.0).min(1.0);
    let flat_gray = (255.0 * face_brightness) as u8;
    let flat_color = ((flat_gray as u32) << 16) | ((flat_gray as u32) << 8) | (flat_gray as u32);

    // Vertex colour support
    let use_vertex_colors = !mesh.vertex_colors.is_empty();
    let mut vert_colors_rgb = [[0u8; 3]; 3];
    if use_vertex_colors {
        for i in 0..3 {
            let c = mesh.vertex_colors[tri[i] as usize];
            vert_colors_rgb[i] = [(c >> 16) as u8, (c >> 8) as u8, c as u8];
        }
    }

    // Pre-compute per-vertex colours for Gouraud (grayscale fallback)
    let mut vert_colors = [[0u8; 3]; 3];
    if quality >= 3 {
        for i in 0..3 {
            let n = if (tri[i] as usize) < mesh.normals.len() {
                mesh.normals[tri[i] as usize]
            } else {
                face_normal
            };
            let b = vec3_dot(&n, light).max(0.0).min(1.0);
            let g = (255.0 * b) as u8;
            vert_colors[i] = [g, g, g];
        }
    }

    // Edge lengths for AA (quality 4)
    let (len0, len1, len2) = if quality >= 4 {
        let l0 = libm::sqrtf((p2[0] - p1[0]) * (p2[0] - p1[0]) + (p2[1] - p1[1]) * (p2[1] - p1[1]));
        let l1 = libm::sqrtf((p0[0] - p2[0]) * (p0[0] - p2[0]) + (p0[1] - p2[1]) * (p0[1] - p2[1]));
        let l2 = libm::sqrtf((p1[0] - p0[0]) * (p1[0] - p0[0]) + (p1[1] - p0[1]) * (p1[1] - p0[1]));
        (l0, l1, l2)
    } else {
        (1.0, 1.0, 1.0)
    };

    let area = edge_fn(&p0, &p1, &[p2[0], p2[1]]);
    if area <= 0.0 {
        return;
    }

    for py in bbox_y0..=bbox_y1 {
        for px in bbox_x0..=bbox_x1 {
            let p = [px as f32 + 0.5, py as f32 + 0.5];
            let w0 = edge_fn(&p1, &p2, &p);
            let w1 = edge_fn(&p2, &p0, &p);
            let w2 = edge_fn(&p0, &p1, &p);
            if w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0 {
                let b0 = w0 / area;
                let b1 = w1 / area;
                let b2 = w2 / area;

                let inv_z = b0 * p0[3] + b1 * p1[3] + b2 * p2[3];
                if inv_z <= 0.0 {
                    continue;
                }
                let z = 1.0 / inv_z;
                let idx = (py as u32 * cw + px as u32) as usize;
                if z < depth_buffer[idx] {
                    depth_buffer[idx] = z;

                    let color = match quality {
                        1 | 2 => {
                            if use_vertex_colors {
                                let r = ((vert_colors_rgb[0][0] as u32
                                    + vert_colors_rgb[1][0] as u32
                                    + vert_colors_rgb[2][0] as u32)
                                    / 3) as u8;
                                let g = ((vert_colors_rgb[0][1] as u32
                                    + vert_colors_rgb[1][1] as u32
                                    + vert_colors_rgb[2][1] as u32)
                                    / 3) as u8;
                                let b = ((vert_colors_rgb[0][2] as u32
                                    + vert_colors_rgb[1][2] as u32
                                    + vert_colors_rgb[2][2] as u32)
                                    / 3) as u8;
                                ((r as u32) << 16) | ((g as u32) << 8) | (b as u32)
                            } else {
                                flat_color
                            }
                        }
                        3 => {
                            if use_vertex_colors {
                                let r = (vert_colors_rgb[0][0] as f32 * b0
                                    + vert_colors_rgb[1][0] as f32 * b1
                                    + vert_colors_rgb[2][0] as f32 * b2)
                                    as u8;
                                let g = (vert_colors_rgb[0][1] as f32 * b0
                                    + vert_colors_rgb[1][1] as f32 * b1
                                    + vert_colors_rgb[2][1] as f32 * b2)
                                    as u8;
                                let b = (vert_colors_rgb[0][2] as f32 * b0
                                    + vert_colors_rgb[1][2] as f32 * b1
                                    + vert_colors_rgb[2][2] as f32 * b2)
                                    as u8;
                                ((r as u32) << 16) | ((g as u32) << 8) | (b as u32)
                            } else {
                                let r = (vert_colors[0][0] as f32 * b0
                                    + vert_colors[1][0] as f32 * b1
                                    + vert_colors[2][0] as f32 * b2)
                                    as u8;
                                ((r as u32) << 16) | ((r as u32) << 8) | (r as u32)
                            }
                        }
                        _ => {
                            // Phong-like per-pixel normal interpolation
                            let n0 = if (tri[0] as usize) < mesh.normals.len() {
                                mesh.normals[tri[0] as usize]
                            } else {
                                face_normal
                            };
                            let n1 = if (tri[1] as usize) < mesh.normals.len() {
                                mesh.normals[tri[1] as usize]
                            } else {
                                face_normal
                            };
                            let n2 = if (tri[2] as usize) < mesh.normals.len() {
                                mesh.normals[tri[2] as usize]
                            } else {
                                face_normal
                            };
                            let nx = n0[0] * b0 + n1[0] * b1 + n2[0] * b2;
                            let ny = n0[1] * b0 + n1[1] * b1 + n2[1] * b2;
                            let nz = n0[2] * b0 + n1[2] * b1 + n2[2] * b2;
                            let nlen = libm::sqrtf(nx * nx + ny * ny + nz * nz);
                            let (nx, ny, nz) = if nlen > 0.0 {
                                (nx / nlen, ny / nlen, nz / nlen)
                            } else {
                                (0.0, 0.0, 0.0)
                            };
                            let b = (nx * light[0] + ny * light[1] + nz * light[2])
                                .max(0.0)
                                .min(1.0);
                            if use_vertex_colors {
                                let r = (vert_colors_rgb[0][0] as f32 * b0
                                    + vert_colors_rgb[1][0] as f32 * b1
                                    + vert_colors_rgb[2][0] as f32 * b2)
                                    as u8;
                                let g = (vert_colors_rgb[0][1] as f32 * b0
                                    + vert_colors_rgb[1][1] as f32 * b1
                                    + vert_colors_rgb[2][1] as f32 * b2)
                                    as u8;
                                let b_col = (vert_colors_rgb[0][2] as f32 * b0
                                    + vert_colors_rgb[1][2] as f32 * b1
                                    + vert_colors_rgb[2][2] as f32 * b2)
                                    as u8;
                                let lit_r = ((r as f32 * b) as u8).min(255);
                                let lit_g = ((g as f32 * b) as u8).min(255);
                                let lit_b = ((b_col as f32 * b) as u8).min(255);
                                ((lit_r as u32) << 16) | ((lit_g as u32) << 8) | (lit_b as u32)
                            } else {
                                let g = (255.0 * b) as u8;
                                ((g as u32) << 16) | ((g as u32) << 8) | (g as u32)
                            }
                        }
                    };

                    if quality >= 4 {
                        let d0 = w0 / len0;
                        let d1 = w1 / len1;
                        let d2 = w2 / len2;
                        let min_dist = d0.min(d1).min(d2);
                        let coverage = (min_dist + 0.5).min(1.0).max(0.0);
                        let alpha = (coverage * 255.0) as u8;
                        htek::set_pixel_blended(target, px as u32, py as u32, color, alpha);
                    } else {
                        target.set_client_pixel(px as u32, py as u32, color);
                    }
                }
            }
        }
    }
}
