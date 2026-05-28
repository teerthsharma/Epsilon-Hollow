// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Lypnos Guard 3D Tensor Renderer — renders tensors as hyperbolic manifolds.
//!
//! v1: hard-coded demo tensor with auto-rotating camera and text overlay.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use spin::Mutex;

use crate::graphics::htek;
use crate::graphics::topo_render::{self, Camera};
use crate::ml_engine::tensor_viz::{self, Tensor, TensorMesh};
use crate::wm::window::Window;

/// Pending CSV data to load on next render (set by shell).
static PENDING_CSV: Mutex<Option<String>> = Mutex::new(None);

pub struct TensorViewer {
    pub tensor_mesh: Option<TensorMesh>,
    pub camera_angle: f32,
    pub stats: TensorStats,
    pub has_loaded_pending: bool,
}

pub struct TensorStats {
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub shape: String,
}

impl TensorViewer {
    pub fn new() -> Self {
        let demo = demo_trading_tensor();
        let stats = compute_stats(&demo);
        let points = tensor_viz::tensor_to_point_cloud(&demo);
        let mesh =
            tensor_viz::point_cloud_to_mesh_grid(&points, &demo.data, demo.shape[0], demo.shape[1]);
        Self {
            tensor_mesh: Some(mesh),
            camera_angle: 0.0,
            stats,
            has_loaded_pending: false,
        }
    }

    pub fn tick(&mut self) {
        self.camera_angle += 0.02;
        if self.camera_angle > 2.0 * core::f32::consts::PI {
            self.camera_angle -= 2.0 * core::f32::consts::PI;
        }

        // Load pending CSV if any (first time only after launch)
        if !self.has_loaded_pending {
            if let Some(csv) = PENDING_CSV.lock().take() {
                let tensor = tensor_viz::parse_csv(&csv);
                if !tensor.data.is_empty() {
                    self.stats = compute_stats(&tensor);
                    let points = tensor_viz::tensor_to_point_cloud(&tensor);
                    let mesh = if tensor.shape.len() == 2 {
                        tensor_viz::point_cloud_to_mesh_grid(
                            &points,
                            &tensor.data,
                            tensor.shape[0],
                            tensor.shape[1],
                        )
                    } else {
                        tensor_viz::point_cloud_to_mesh(&points, &tensor.data)
                    };
                    self.tensor_mesh = Some(mesh);
                }
                self.has_loaded_pending = true;
            }
        }
    }

    pub fn render_to_window(&self, win: &mut Window) {
        let cw = win.client_width();
        let ch = win.client_height();

        // Clear background
        for y in 0..ch {
            for x in 0..cw {
                win.set_client_pixel(x, y, 0x000000);
            }
        }

        if let Some(ref tm) = self.tensor_mesh {
            let radius = 4.0;
            let cam = Camera {
                position: [
                    radius * libm::cosf(self.camera_angle),
                    2.5,
                    radius * libm::sinf(self.camera_angle),
                ],
                look_at: [0.0, 0.0, 0.0],
                up: [0.0, 1.0, 0.0],
                fov_deg: 60.0,
                near: 0.1,
                far: 100.0,
            };
            topo_render::set_camera(cam);
            topo_render::render_mesh(&tm.mesh, win);

            // Stats overlay
            let text = format!(
                "shape={}  min={:.2}  max={:.2}  mean={:.2}",
                self.stats.shape, self.stats.min, self.stats.max, self.stats.mean
            );
            htek::render_text_small(win, 4, 4, &text, 0xFFFFFF);
        } else {
            htek::render_text_small(win, 4, 4, "No tensor loaded", 0xFFFFFF);
        }
    }

    pub fn mouse_click(&mut self, _x: u32, _y: u32, _pressed: bool) {}
    pub fn mouse_move(&mut self, _x: u32, _y: u32) {}
}

/// Set CSV data to be loaded by the viewer on its next tick.
pub fn set_pending_csv(data: &str) {
    *PENDING_CSV.lock() = Some(String::from(data));
}

fn demo_trading_tensor() -> Tensor {
    // 16×16 sine-wave surface — simulates a periodic trading pattern.
    let rows = 16usize;
    let cols = 16usize;
    let mut data = Vec::with_capacity(rows * cols);
    for i in 0..rows {
        for j in 0..cols {
            let x = i as f32 / rows as f32 * 2.0 * core::f32::consts::PI;
            let y = j as f32 / cols as f32 * 2.0 * core::f32::consts::PI;
            let v = libm::sinf(x) * libm::cosf(y) + 0.3 * libm::sinf(3.0 * x);
            data.push(v);
        }
    }
    Tensor {
        shape: vec![rows, cols],
        data,
    }
}

fn compute_stats(tensor: &Tensor) -> TensorStats {
    let mut min = f32::MAX;
    let mut max = f32::MIN;
    let mut sum = 0.0f32;
    for &v in &tensor.data {
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
        sum += v;
    }
    let mean = if !tensor.data.is_empty() {
        sum / tensor.data.len() as f32
    } else {
        0.0
    };
    let shape_str = tensor
        .shape
        .iter()
        .map(|s| format!("{}", s))
        .collect::<Vec<_>>()
        .join("x");
    TensorStats {
        min,
        max,
        mean,
        shape: shape_str,
    }
}

pub fn main() {
    loop {
        crate::process::scheduler::yield_current();
    }
}
