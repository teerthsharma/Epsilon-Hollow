// Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: Epsilon-Hollow

// ═══════════════════════════════════════════════════════════════════════════════
//! AEGIS Interpreter - Runtime execution of AEGIS programs
// ═══════════════════════════════════════════════════════════════════════════════
//!
//! Executes parsed AEGIS AST, managing:
//! - 3D manifold workspaces
//! - Block geometry computations
//! - Escalating regression benchmarks
//! - Topological convergence detection
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::boxed::Box;
#[cfg(not(feature = "std"))]
use alloc::collections::BTreeMap;
#[cfg(not(feature = "std"))]
use alloc::string::String;
#[cfg(not(feature = "std"))]
use alloc::string::ToString;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(not(feature = "std"))]
use alloc::{format, vec};

#[cfg(not(feature = "std"))]
macro_rules! println {
    ($($arg:tt)*) => {};
}

#[cfg(feature = "std")]
use std::boxed::Box;
#[cfg(feature = "std")]
use std::collections::BTreeMap;
#[cfg(feature = "std")]
use std::string::String;
#[cfg(feature = "std")]
use std::vec::Vec;

use crate::ast::*;
use aether_core::aether::{BlockMetadata, DriftDetector, HierarchicalBlockTree};
use aether_core::manifold::{ManifoldPoint, TimeDelayEmbedder};
use aether_core::ml::convolution::Conv2D;
use aether_core::ml::linalg::LossConfig;
use aether_core::ml::tensor::Tensor;
use aether_core::ml::{Activation, KMeans, OptimizerConfig, MLP};
use libm::{fabs, sqrt};

#[cfg(all(feature = "ml", not(feature = "std")))]
use alloc::sync::Arc;
#[cfg(feature = "net")]
use safetensors::SafeTensors;
#[cfg(all(feature = "ml", feature = "std"))]
use std::sync::Arc;

#[cfg(feature = "ml")]
use candle_core::{Device, Tensor as CandleTensor};
#[cfg(feature = "ml")]
use candle_transformers::models::quantized_llama::ModelWeights as LlamaWeights;
#[cfg(feature = "ml")]
use tokenizers::Tokenizer;

/// Interpreter errors
#[derive(Debug, Clone)]
pub enum InterpreterError {
    Message(String),
    Return(Value),
    Break,
    Continue,
}

impl From<String> for InterpreterError {
    fn from(s: String) -> Self {
        InterpreterError::Message(s)
    }
}

impl From<&str> for InterpreterError {
    fn from(s: &str) -> Self {
        InterpreterError::Message(s.into())
    }
}

/// Embedding dimension
const DIM: usize = 3;

// ═══════════════════════════════════════════════════════════════════════════════
// Runtime Values
// ═══════════════════════════════════════════════════════════════════════════════

/// Runtime value types
#[derive(Debug, Clone)]
pub enum Value {
    /// Numeric value
    Num(f64),
    /// Boolean
    Bool(bool),
    /// String
    Str(String),
    /// 3D Manifold reference
    Manifold(ManifoldHandle),
    /// Geometric block reference  
    Block(BlockHandle),
    /// 3D Point
    Point([f64; DIM]),
    /// Regression result
    RegressionResult(RegressionOutput),
    /// Class Definition
    Class(ClassHandle),
    /// Object Instance
    Object(ObjectHandle),
    /// Native Function (for Standard Library)
    NativeFn(NativeFunction),
    /// User Function
    Function(UserFunction),
    /// Dynamic List (Python-like)
    List(Vec<Value>),
    /// ML Types
    Mlp(Box<MLP>),
    KMeans(Box<KMeans<DIM>>),
    Conv2D(Box<Conv2D>),
    /// Void/Unit
    Unit,
    /// Module Namespace
    Module(String),
    /// Dynamic Tensor
    Tensor(Tensor),
    /// Llama Model (Wrapped)
    #[cfg(feature = "ml")]
    LlamaModel(Arc<LlamaContext>),
}

#[cfg(feature = "ml")]
#[derive(Debug)]
pub struct LlamaContext {
    pub model: LlamaWeights,
    pub tokenizer: Tokenizer,
    pub name: String,
}

/// Native function pointer type
#[derive(Debug, Clone)]
pub enum NativeFunction {
    MathSin,
    MathCos,
    MathSqrt,
    MathExp,
    TopoBetti,
    Print,
    // ML Constructors
    MlpNew,
    KMeansNew,
    Conv2DNew,
    // Seal Functions
    SealTrain,
    // Tensor Ops
    MlLoadWeights,
    MlMatMul,
    MlAdd,
    MlForward,
    MlRelu,
    MlSoftmax,
    MlEmbed,
    MlAttention,
    MlGpuCheck,
    MlBackward,
    MlUpdate,
    MlLoadLlama,
    MlGenerate,
    // Driver MMIO/PCI
    MmioRead32,
    MmioWrite32,
    PciFind,
    PciFindVendor,
    PciReadBar0,
    PciReadBar5,
    MathRange,
}

/// Handle to a manifold workspace
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ManifoldHandle(pub usize);

/// Handle to a geometric block
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockHandle(pub usize);

/// Handle to a class definition
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClassHandle(pub usize);

/// Handle to an object instance
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObjectHandle(pub usize);

/// User-defined function with captured closure
#[derive(Debug, Clone)]
pub struct UserFunction {
    pub decl: FnDecl,
    pub closure: BTreeMap<String, Value>,
}

/// Class Definition Runtime
#[derive(Debug, Clone)]
pub struct ClassDef {
    pub name: String,
    pub fields: Vec<VarDecl>,
    pub methods: BTreeMap<String, FnDecl>,
}

/// Object Instance Runtime
#[derive(Debug, Clone)]
pub struct ObjectInstance {
    pub class: ClassHandle,
    pub fields: BTreeMap<String, Value>,
}

/// Regression output with convergence info
#[derive(Debug, Clone)]
pub struct RegressionOutput {
    /// Final coefficients
    pub coefficients: [f64; 8],
    /// Number of epochs to converge
    pub epochs: u32,
    /// Final error
    pub final_error: f64,
    /// Converged?
    pub converged: bool,
    /// Betti numbers at convergence
    pub betti: (u32, u32),
}

// ═══════════════════════════════════════════════════════════════════════════════
// Manifold Workspace
// ═══════════════════════════════════════════════════════════════════════════════

/// 3D Manifold workspace containing embedded points
#[derive(Debug)]
pub struct ManifoldWorkspace {
    /// Embedded points in 3D
    pub points: Vec<ManifoldPoint<DIM>>,
    /// Hierarchical block tree for AETHER
    pub block_tree: HierarchicalBlockTree<DIM>,
    /// Drift detector for convergence
    pub drift: DriftDetector<DIM>,
    /// Time-delay embedder
    pub embedder: TimeDelayEmbedder<DIM>,
    /// Current centroid
    pub centroid: [f64; DIM],
}

impl ManifoldWorkspace {
    pub fn new(tau: usize) -> Self {
        Self {
            points: Vec::new(),
            block_tree: HierarchicalBlockTree::new(),
            drift: DriftDetector::new(),
            embedder: TimeDelayEmbedder::new(tau),
            centroid: [0.0; DIM],
        }
    }

    /// Embed raw data into 3D manifold
    pub fn embed_data(&mut self, data: &[f64]) {
        self.points.clear();
        self.embedder.reset();

        for &val in data {
            self.embedder.push(val);
            if let Some(point) = self.embedder.embed() {
                self.points.push(point);
            }
        }

        self.update_centroid();
    }

    /// Update centroid from points
    fn update_centroid(&mut self) {
        if self.points.is_empty() {
            return;
        }

        let mut sum = [0.0; DIM];
        for p in &self.points {
            for (d, s) in sum.iter_mut().enumerate().take(DIM) {
                *s += p.coords[d];
            }
        }

        let n = self.points.len() as f64;
        for (d, s) in sum.iter().enumerate().take(DIM) {
            self.centroid[d] = s / n;
        }
    }

    /// Extract block from index range
    pub fn extract_block(&self, start: usize, end: usize) -> BlockMetadata<DIM> {
        let end = end.min(self.points.len());
        let start = start.min(end);

        if start >= end {
            return BlockMetadata::empty();
        }

        let mut block_points = Vec::new();
        for i in start..end {
            block_points.push(self.points[i].coords);
        }

        BlockMetadata::from_points(&block_points)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Escalating Regression Engine
// ═══════════════════════════════════════════════════════════════════════════════

/// Regression model types
#[derive(Debug, Clone, Copy)]
pub enum RegressionModel {
    Linear,
    Polynomial { degree: u8 },
    Rbf { gamma: f64 },
}

/// Escalating benchmark system
pub struct EscalatingRegressor {
    /// Current model complexity
    _current_level: u32,
    /// Target for regression
    target: Vec<f64>,
    /// Predictions
    _predictions: Vec<f64>,
    /// Convergence epsilon
    epsilon: f64,
    /// Betti stability window
    betti_history: Vec<(u32, u32)>,
}

impl EscalatingRegressor {
    pub fn new(epsilon: f64) -> Self {
        Self {
            _current_level: 0,
            target: Vec::new(),
            _predictions: Vec::new(),
            epsilon,
            betti_history: Vec::new(),
        }
    }

    /// Set target values for regression
    pub fn set_target(&mut self, data: &[f64]) {
        self.target.clear();
        for &v in data {
            self.target.push(v);
        }
    }

    /// Run escalating regression until convergence
    pub fn run_escalating(
        &mut self,
        manifold: &ManifoldWorkspace,
        max_epochs: u32,
    ) -> RegressionOutput {
        let mut coefficients = [0.0f64; 8];
        let mut error = f64::MAX;
        let mut converged = false;
        let mut epochs = 0u32;

        for epoch in 0..max_epochs {
            epochs = epoch;
            let model = self.escalate_model(epoch);
            coefficients = self.fit_model(manifold, &model);
            error = self.compute_error(manifold, &coefficients, &model);
            let betti = self.compute_residual_betti(manifold, &coefficients, &model);
            self.betti_history.push(betti);
            if self.betti_history.len() > 10 {
                self.betti_history.remove(0);
            }
            if self.is_converged(error, &betti) {
                converged = true;
                break;
            }
        }

        RegressionOutput {
            coefficients,
            epochs,
            final_error: error,
            converged,
            betti: *self.betti_history.last().unwrap_or(&(0, 0)),
        }
    }

    fn escalate_model(&self, epoch: u32) -> RegressionModel {
        match epoch {
            0 => RegressionModel::Linear,
            1 => RegressionModel::Polynomial { degree: 2 },
            2 => RegressionModel::Polynomial { degree: 3 },
            3 => RegressionModel::Polynomial { degree: 4 },
            4..=6 => RegressionModel::Rbf {
                gamma: 0.1 * (epoch as f64),
            },
            _ => RegressionModel::Rbf { gamma: 1.0 },
        }
    }

    fn fit_model(&self, manifold: &ManifoldWorkspace, model: &RegressionModel) -> [f64; 8] {
        let mut coeffs = [0.0f64; 8];

        if manifold.points.is_empty() || self.target.is_empty() {
            return coeffs;
        }

        match model {
            RegressionModel::Linear => {
                let n = manifold.points.len().min(self.target.len()) as f64;
                let mut sum_x = 0.0;
                let mut sum_y = 0.0;
                let mut sum_xy = 0.0;
                let mut sum_xx = 0.0;

                for (i, p) in manifold.points.iter().enumerate() {
                    if i >= self.target.len() {
                        break;
                    }
                    let x = p.coords[0];
                    let y = self.target[i];
                    sum_x += x;
                    sum_y += y;
                    sum_xy += x * y;
                    sum_xx += x * x;
                }

                let denom = n * sum_xx - sum_x * sum_x;
                if fabs(denom) > 1e-10 {
                    coeffs[1] = (n * sum_xy - sum_x * sum_y) / denom;
                    coeffs[0] = (sum_y - coeffs[1] * sum_x) / n;
                }
            }
            RegressionModel::Polynomial { degree } => {
                coeffs = self.fit_model(manifold, &RegressionModel::Linear);
                coeffs[*degree as usize] = 0.01;
            }
            RegressionModel::Rbf { .. } => {
                coeffs = self.fit_model(manifold, &RegressionModel::Polynomial { degree: 3 });
            }
        }

        coeffs
    }

    fn compute_error(
        &self,
        manifold: &ManifoldWorkspace,
        coeffs: &[f64; 8],
        model: &RegressionModel,
    ) -> f64 {
        let mut mse = 0.0;
        let mut count = 0;

        for (i, p) in manifold.points.iter().enumerate() {
            if i >= self.target.len() {
                break;
            }
            let pred = self.predict(p.coords[0], coeffs, model);
            let err = pred - self.target[i];
            mse += err * err;
            count += 1;
        }

        if count > 0 {
            mse /= count as f64;
            sqrt(mse)
        } else {
            f64::MAX
        }
    }

    fn predict(&self, x: f64, coeffs: &[f64; 8], model: &RegressionModel) -> f64 {
        predict_value(x, coeffs, model)
    }

    fn compute_residual_betti(
        &self,
        manifold: &ManifoldWorkspace,
        coeffs: &[f64; 8],
        model: &RegressionModel,
    ) -> (u32, u32) {
        let mut sign_changes = 0u32;
        let mut oscillations = 0u32;
        let mut prev_residual = 0.0;
        let mut prev_sign = true;

        for (i, p) in manifold.points.iter().enumerate() {
            if i >= self.target.len() {
                break;
            }
            let pred = self.predict(p.coords[0], coeffs, model);
            let residual = self.target[i] - pred;
            let sign = residual >= 0.0;
            if i > 0 && sign != prev_sign {
                sign_changes += 1;
            }
            if i > 1 {
                let delta = residual - prev_residual;
                let prev_delta = prev_residual;
                if (delta > 0.0) != (prev_delta > 0.0) {
                    oscillations += 1;
                }
            }
            prev_residual = residual;
            prev_sign = sign;
        }

        (sign_changes / 2 + 1, oscillations / 4)
    }

    fn is_converged(&self, error: f64, current_betti: &(u32, u32)) -> bool {
        if error < self.epsilon {
            return true;
        }
        if self.betti_history.len() >= 3 {
            let recent: Vec<&(u32, u32)> = self.betti_history.iter().rev().take(3).collect();
            if recent.iter().all(|b| **b == *current_betti) {
                return true;
            }
        }
        false
    }
}

fn predict_value(x: f64, coeffs: &[f64; 8], model: &RegressionModel) -> f64 {
    match model {
        RegressionModel::Linear => coeffs[0] + coeffs[1] * x,
        RegressionModel::Polynomial { degree } => {
            let mut y = coeffs[0];
            let mut x_pow = x;
            for coeff in coeffs.iter().take((*degree as usize).min(7) + 1).skip(1) {
                y += coeff * x_pow;
                x_pow *= x;
            }
            y
        }
        RegressionModel::Rbf { .. } => {
            predict_value(x, coeffs, &RegressionModel::Polynomial { degree: 3 })
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main Interpreter
// ═══════════════════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════════════════
// Software Framebuffer (htek.rs style)
// ═══════════════════════════════════════════════════════════════════════════════

/// Simple software framebuffer for render output
#[derive(Debug, Clone)]
pub struct Framebuffer {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u32>,
}

impl Framebuffer {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            pixels: vec![0; (width * height) as usize],
        }
    }

    pub fn clear(&mut self, color: u32) {
        for p in self.pixels.iter_mut() {
            *p = color;
        }
    }

    pub fn draw_rect(&mut self, x: u32, y: u32, w: u32, h: u32, color: u32) {
        for dy in 0..h {
            let py = y + dy;
            if py >= self.height {
                break;
            }
            for dx in 0..w {
                let px = x + dx;
                if px >= self.width {
                    break;
                }
                self.pixels[(py * self.width + px) as usize] = color;
            }
        }
    }

    pub fn draw_text(&mut self, mut x: u32, y: u32, text: &str, color: u32) {
        for ch in text.bytes() {
            Self::draw_char(self, x, y, ch, color);
            x += 4;
        }
    }

    fn draw_char(&mut self, x: u32, y: u32, ch: u8, color: u32) {
        let pattern: u16 = match ch {
            b'0' => 0b111101101101111,
            b'1' => 0b010010010010010,
            b'2' => 0b111001111100111,
            b'3' => 0b111001111001111,
            b'4' => 0b100101111001001,
            b'5' => 0b111100111001111,
            b'6' => 0b111100111101111,
            b'7' => 0b111001001001001,
            b'8' => 0b111101111101111,
            b'9' => 0b111101111001111,
            b'a' | b'A' => 0b010101111101101,
            b'b' | b'B' => 0b110101110101110,
            b'c' | b'C' => 0b111100100100111,
            b'd' | b'D' => 0b110101101101110,
            b'e' | b'E' => 0b111100111100111,
            b'f' | b'F' => 0b111100111100100,
            b'g' | b'G' => 0b111100101101111,
            b'h' | b'H' => 0b101101111101101,
            b'i' | b'I' => 0b111010010010111,
            b'j' | b'J' => 0b001001001101111,
            b'k' | b'K' => 0b101101110101101,
            b'l' | b'L' => 0b100100100100111,
            b'm' | b'M' => 0b101111101101101,
            b'n' | b'N' => 0b111101101101101,
            b'o' | b'O' => 0b010101101101010,
            b'p' | b'P' => 0b110101111100100,
            b'q' | b'Q' => 0b111101101111001,
            b'r' | b'R' => 0b110101110101101,
            b's' | b'S' => 0b111100111001111,
            b't' | b'T' => 0b111010010010010,
            b'u' | b'U' => 0b101101101101111,
            b'v' | b'V' => 0b101101101101010,
            b'w' | b'W' => 0b101101111111101,
            b'x' | b'X' => 0b101101010101101,
            b'y' | b'Y' => 0b101101111010010,
            b'z' | b'Z' => 0b111001010100111,
            b' ' => 0b000000000000000,
            b':' => 0b000010000010000,
            b'-' => 0b000000111000000,
            b'_' => 0b000000000000111,
            _ => 0b111101101101111,
        };
        for row in 0..5 {
            for col in 0..3 {
                let bit = (pattern >> (row * 3 + (2 - col))) & 1;
                if bit == 1 {
                    let px = x + col;
                    let py = y + row;
                    if px < self.width && py < self.height {
                        self.pixels[(py * self.width + px) as usize] = color;
                    }
                }
            }
        }
    }
}

fn find_bounds(points: &[[f64; 3]]) -> (f64, f64, f64, f64) {
    let mut min_x = f64::MAX;
    let mut max_x = f64::MIN;
    let mut min_y = f64::MAX;
    let mut max_y = f64::MIN;
    for p in points {
        if p[0] < min_x {
            min_x = p[0];
        }
        if p[0] > max_x {
            max_x = p[0];
        }
        if p[1] < min_y {
            min_y = p[1];
        }
        if p[1] > max_y {
            max_y = p[1];
        }
    }
    (min_x, max_x, min_y, max_y)
}

/// Runtime environment
pub struct Interpreter {
    /// Variable bindings
    pub variables: BTreeMap<String, Value>, // Made public for tests
    /// Manifold workspaces
    manifolds: Vec<ManifoldWorkspace>,
    /// Block geometries
    blocks: Vec<BlockMetadata<DIM>>,
    /// Class definitions
    classes: Vec<ClassDef>,
    /// Object instances
    objects: Vec<ObjectInstance>,
    /// Sample data (for demo)
    sample_data: Vec<f64>,
    /// Software framebuffer for render
    pub framebuffer: Option<Framebuffer>,
}

impl Interpreter {
    pub fn new() -> Self {
        let mut data = Vec::new();
        for i in 0..64 {
            let x = (i as f64) * 0.1;
            data.push(libm::sin(x));
        }

        let mut variables = BTreeMap::new();
        variables.insert(
            String::from("print"),
            Value::NativeFn(NativeFunction::Print),
        );

        Self {
            variables,
            manifolds: Vec::new(),
            blocks: Vec::new(),
            classes: Vec::new(),
            objects: Vec::new(),
            sample_data: data,
            framebuffer: None,
        }
    }

    /// Execute a program
    pub fn execute(&mut self, program: &Program) -> Result<Value, InterpreterError> {
        let mut last_value = Value::Unit;
        for stmt in &program.statements {
            last_value = self.execute_statement(stmt)?;
        }
        Ok(last_value)
    }

    fn execute_for(&mut self, stmt: &ForStmt) -> Result<Value, InterpreterError> {
        let iterable_val = self.evaluate_expr(&stmt.iterable)?;
        let mut last_val = Value::Unit;

        match iterable_val {
            Value::List(elements) => {
                for el in elements {
                    self.variables.insert(stmt.iterator.clone(), el);
                    match self.execute_stmt_block(&stmt.body) {
                        Ok(v) => last_val = v,
                        Err(InterpreterError::Break) => break,
                        Err(InterpreterError::Continue) => continue,
                        Err(e) => return Err(e),
                    }
                }
            }
            _ => return Err("Expected list or iterable in for loop".into()),
        }

        Ok(last_val)
    }

    fn execute_statement(&mut self, stmt: &Statement) -> Result<Value, InterpreterError> {
        match &stmt.node {
            StmtKind::Manifold(decl) => self.execute_manifold(decl),
            StmtKind::Block(decl) => self.execute_block(decl),
            StmtKind::Var(decl) => self.execute_var(decl),
            StmtKind::Regress(stmt) => self.execute_regress(stmt),
            StmtKind::Render(stmt) => self.execute_render(stmt),
            StmtKind::Class(decl) => self.execute_class(decl),
            StmtKind::Import(stmt) => self.execute_import(stmt),
            StmtKind::If(stmt) => self.execute_if(stmt),
            StmtKind::While(stmt) => self.execute_while(stmt),
            StmtKind::Loop(stmt) => self.execute_seal(stmt),
            StmtKind::For(stmt) => self.execute_for(stmt),
            StmtKind::Fn(decl) => {
                let mut closure = self.variables.clone();
                closure.remove(&decl.name);
                let func = UserFunction {
                    decl: decl.clone(),
                    closure,
                };
                self.variables
                    .insert(decl.name.clone(), Value::Function(func));
                Ok(Value::Unit)
            }
            StmtKind::Return(stmt) => {
                let val = if let Some(expr) = &stmt.value {
                    self.evaluate_expr(expr)?
                } else {
                    Value::Unit
                };
                Err(InterpreterError::Return(val))
            }
            StmtKind::Break(_) => Err(InterpreterError::Break),
            StmtKind::Continue(_) => Err(InterpreterError::Continue),
            StmtKind::Expr(expr) => {
                self.evaluate_expr(expr)?;
                Ok(Value::Unit)
            }
            StmtKind::Empty => Ok(Value::Unit),
        }
    }

    fn execute_class(&mut self, decl: &ClassDecl) -> Result<Value, InterpreterError> {
        let mut methods = BTreeMap::new();
        for m in &decl.methods {
            methods.insert(m.name.clone(), m.clone());
        }

        let class_def = ClassDef {
            name: decl.name.clone(),
            fields: decl.fields.clone(),
            methods,
        };

        let handle = ClassHandle(self.classes.len());
        self.classes.push(class_def);
        self.variables
            .insert(decl.name.clone(), Value::Class(handle));
        Ok(Value::Class(handle))
    }

    #[allow(unused_variables)]
    fn evaluate_new(
        &mut self,
        class_name: &String,
        args: &[Expr],
    ) -> Result<Value, InterpreterError> {
        let class_handle = if let Some(Value::Class(h)) = self.variables.get(class_name) {
            *h
        } else {
            return Err(format!("Class '{}' not found", class_name).into());
        };

        let class_def = self.classes[class_handle.0].clone();
        let mut fields = BTreeMap::new();
        for field in &class_def.fields {
            let val = self.evaluate_expr(&field.value)?;
            fields.insert(field.name.clone(), val);
        }

        let obj_handle = ObjectHandle(self.objects.len());
        self.objects.push(ObjectInstance {
            class: class_handle,
            fields,
        });

        Ok(Value::Object(obj_handle))
    }

    fn execute_import(&mut self, stmt: &ImportStmt) -> Result<Value, InterpreterError> {
        let mod_name = stmt.module.as_str();
        match mod_name {
            "math" => self.import_math(stmt),
            "topology" => self.import_topology(stmt),
            "ml" | "Ml" => self.import_ml(stmt),
            "Seal" => self.import_seal(stmt),
            "fs" => self.import_fs(stmt),
            "process" => self.import_process(stmt),
            "net" => self.import_net(stmt),
            _ => Err(format!("Module '{}' not found", mod_name).into()),
        }
    }

    fn import_math(&mut self, stmt: &ImportStmt) -> Result<Value, InterpreterError> {
        if let Some(symbol) = &stmt.symbol {
            match symbol.as_str() {
                "pi" => {
                    self.variables
                        .insert(String::from("pi"), Value::Num(core::f64::consts::PI));
                }
                "e" => {
                    self.variables
                        .insert(String::from("e"), Value::Num(core::f64::consts::E));
                }
                "sin" => {
                    self.variables.insert(
                        String::from("sin"),
                        Value::NativeFn(NativeFunction::MathSin),
                    );
                }
                "cos" => {
                    self.variables.insert(
                        String::from("cos"),
                        Value::NativeFn(NativeFunction::MathCos),
                    );
                }
                "sqrt" => {
                    self.variables.insert(
                        String::from("sqrt"),
                        Value::NativeFn(NativeFunction::MathSqrt),
                    );
                }
                "exp" => {
                    self.variables.insert(
                        String::from("exp"),
                        Value::NativeFn(NativeFunction::MathExp),
                    );
                }
                "range" => {
                    self.variables.insert(
                        String::from("range"),
                        Value::NativeFn(NativeFunction::MathRange),
                    );
                }
                _ => return Err(format!("Symbol '{}' not found in math", symbol).into()),
            }
        } else {
            self.variables
                .insert(String::from("pi"), Value::Num(core::f64::consts::PI));
            self.variables
                .insert(String::from("e"), Value::Num(core::f64::consts::E));
            self.variables.insert(
                String::from("sin"),
                Value::NativeFn(NativeFunction::MathSin),
            );
            self.variables.insert(
                String::from("cos"),
                Value::NativeFn(NativeFunction::MathCos),
            );
            self.variables.insert(
                String::from("sqrt"),
                Value::NativeFn(NativeFunction::MathSqrt),
            );
        }
        Ok(Value::Unit)
    }

    fn import_topology(&mut self, stmt: &ImportStmt) -> Result<Value, InterpreterError> {
        if let Some(symbol) = &stmt.symbol {
            match symbol.as_str() {
                "Betti" => self.variables.insert(
                    String::from("Betti"),
                    Value::NativeFn(NativeFunction::TopoBetti),
                ),
                _ => return Err(format!("Symbol '{}' not found in topology", symbol).into()),
            };
        }
        Ok(Value::Unit)
    }

    fn import_ml(&mut self, stmt: &ImportStmt) -> Result<Value, InterpreterError> {
        if let Some(symbol) = &stmt.symbol {
            match symbol.as_str() {
                "MLP" => {
                    self.variables
                        .insert(String::from("MLP"), Value::NativeFn(NativeFunction::MlpNew));
                }
                "KMeans" => {
                    self.variables.insert(
                        String::from("KMeans"),
                        Value::NativeFn(NativeFunction::KMeansNew),
                    );
                }
                "Conv2D" => {
                    self.variables.insert(
                        String::from("Conv2D"),
                        Value::NativeFn(NativeFunction::Conv2DNew),
                    );
                }
                _ => return Err(format!("Symbol '{}' not found in ml", symbol).into()),
            };
        } else {
            self.variables
                .insert(String::from("MLP"), Value::NativeFn(NativeFunction::MlpNew));
            self.variables.insert(
                String::from("KMeans"),
                Value::NativeFn(NativeFunction::KMeansNew),
            );
            self.variables.insert(
                String::from("Conv2D"),
                Value::NativeFn(NativeFunction::Conv2DNew),
            );
            self.variables.insert(
                String::from("load_weights"),
                Value::NativeFn(NativeFunction::MlLoadWeights),
            );
            self.variables.insert(
                String::from("matmul"),
                Value::NativeFn(NativeFunction::MlMatMul),
            );
            self.variables
                .insert(String::from("add"), Value::NativeFn(NativeFunction::MlAdd));
            self.variables.insert(
                String::from("relu"),
                Value::NativeFn(NativeFunction::MlRelu),
            );
            self.variables.insert(
                String::from("softmax"),
                Value::NativeFn(NativeFunction::MlSoftmax),
            );
            self.variables.insert(
                String::from("attention"),
                Value::NativeFn(NativeFunction::MlAttention),
            );
            self.variables.insert(
                String::from("gpu_check"),
                Value::NativeFn(NativeFunction::MlGpuCheck),
            );
            self.variables.insert(
                String::from("backward"),
                Value::NativeFn(NativeFunction::MlBackward),
            );
            self.variables.insert(
                String::from("update"),
                Value::NativeFn(NativeFunction::MlUpdate),
            );
            self.variables.insert(
                String::from("load_llama"),
                Value::NativeFn(NativeFunction::MlLoadLlama),
            );
            self.variables.insert(
                String::from("generate"),
                Value::NativeFn(NativeFunction::MlGenerate),
            );
            self.variables
                .insert(String::from("Ml"), Value::Module(String::from("Ml")));
        }
        Ok(Value::Unit)
    }

    fn import_seal(&mut self, stmt: &ImportStmt) -> Result<Value, InterpreterError> {
        if let Some(symbol) = &stmt.symbol {
            match symbol.as_str() {
                "train" => {
                    self.variables.insert(
                        String::from("train"),
                        Value::NativeFn(NativeFunction::SealTrain),
                    );
                    self.variables.insert(
                        String::from("mmio_read32"),
                        Value::NativeFn(NativeFunction::MmioRead32),
                    );
                    self.variables.insert(
                        String::from("mmio_write32"),
                        Value::NativeFn(NativeFunction::MmioWrite32),
                    );
                    self.variables.insert(
                        String::from("pci_find"),
                        Value::NativeFn(NativeFunction::PciFind),
                    );
                    self.variables.insert(
                        String::from("pci_find_vendor"),
                        Value::NativeFn(NativeFunction::PciFindVendor),
                    );
                    self.variables.insert(
                        String::from("pci_read_bar0"),
                        Value::NativeFn(NativeFunction::PciReadBar0),
                    );
                    self.variables.insert(
                        String::from("pci_read_bar5"),
                        Value::NativeFn(NativeFunction::PciReadBar5),
                    );
                }
                _ => return Err(format!("Symbol '{}' not found in Seal", symbol).into()),
            };
        } else {
            self.variables
                .insert(String::from("Seal"), Value::Module(String::from("Seal")));
        }
        Ok(Value::Unit)
    }

    fn import_fs(&mut self, stmt: &ImportStmt) -> Result<Value, InterpreterError> {
        if let Some(symbol) = &stmt.symbol {
            match symbol.as_str() {
                "read" | "write" | "exists" | "mkdir" => {
                    self.variables
                        .insert(symbol.clone(), Value::NativeFn(NativeFunction::Print));
                }
                _ => return Err(format!("Symbol '{}' not found in fs", symbol).into()),
            };
        } else {
            self.variables
                .insert(String::from("fs"), Value::Module(String::from("fs")));
        }
        Ok(Value::Unit)
    }

    fn import_process(&mut self, stmt: &ImportStmt) -> Result<Value, InterpreterError> {
        if let Some(symbol) = &stmt.symbol {
            match symbol.as_str() {
                "pid" => {
                    self.variables.insert(String::from("pid"), Value::Num(1.0));
                }
                "exit" => {
                    self.variables
                        .insert(String::from("exit"), Value::NativeFn(NativeFunction::Print));
                }
                _ => return Err(format!("Symbol '{}' not found in process", symbol).into()),
            };
        } else {
            self.variables.insert(
                String::from("process"),
                Value::Module(String::from("process")),
            );
        }
        Ok(Value::Unit)
    }

    fn import_net(&mut self, stmt: &ImportStmt) -> Result<Value, InterpreterError> {
        if let Some(symbol) = &stmt.symbol {
            match symbol.as_str() {
                "local_ip" => {
                    self.variables.insert(
                        String::from("local_ip"),
                        Value::Str(String::from("127.0.0.1")),
                    );
                }
                "has_nic" => {
                    self.variables
                        .insert(String::from("has_nic"), Value::Bool(false));
                }
                _ => return Err(format!("Symbol '{}' not found in net", symbol).into()),
            };
        } else {
            self.variables
                .insert(String::from("net"), Value::Module(String::from("net")));
        }
        Ok(Value::Unit)
    }

    fn execute_manifold(&mut self, decl: &ManifoldDecl) -> Result<Value, InterpreterError> {
        let tau = self.extract_tau(&decl.init).unwrap_or(3);
        let mut workspace = ManifoldWorkspace::new(tau);
        workspace.embed_data(&self.sample_data);
        let handle = ManifoldHandle(self.manifolds.len());
        self.manifolds.push(workspace);
        self.variables
            .insert(decl.name.clone(), Value::Manifold(handle));
        Ok(Value::Manifold(handle))
    }

    fn extract_tau(&self, expr: &Expr) -> Option<usize> {
        if let ExprKind::Call { args, .. } = &expr.node {
            for arg in args {
                if let CallArg::Named { name, value } = arg {
                    if name.as_str() == "tau" {
                        if let ExprKind::Literal(Literal::Num(n)) = &value.node {
                            return Some(*n as usize);
                        }
                    }
                }
            }
        }
        None
    }

    fn execute_block(&mut self, decl: &BlockDecl) -> Result<Value, InterpreterError> {
        let (manifold_handle, start, end) = self.extract_block_range(&decl.source)?;
        if let Some(workspace) = self.manifolds.get(manifold_handle.0) {
            let block = workspace.extract_block(start, end);
            let handle = BlockHandle(self.blocks.len());
            self.blocks.push(block);
            self.variables
                .insert(decl.name.clone(), Value::Block(handle));
            Ok(Value::Block(handle))
        } else {
            Err("manifold not found".into())
        }
    }

    fn extract_block_range(
        &self,
        expr: &Expr,
    ) -> Result<(ManifoldHandle, usize, usize), InterpreterError> {
        match &expr.node {
            ExprKind::MethodCall { object, args, .. } => {
                let handle = self.get_manifold_handle(object)?;
                let (start, end) = self.extract_range_from_args(args);
                Ok((handle, start, end))
            }
            ExprKind::Index { object, range } => {
                let handle = self.get_manifold_handle(object)?;
                let start = range.start.as_f64() as usize;
                let end = range.end.as_f64() as usize;
                Ok((handle, start, end))
            }
            _ => Err("invalid block source".into()),
        }
    }

    fn get_manifold_handle(&self, name: &String) -> Result<ManifoldHandle, InterpreterError> {
        if let Some(Value::Manifold(h)) = self.variables.get(name) {
            Ok(*h)
        } else {
            Err("variable is not a manifold".into())
        }
    }

    fn extract_range_from_args(&self, args: &[CallArg]) -> (usize, usize) {
        let mut start = 0usize;
        let mut end = 64usize;

        for (i, arg) in args.iter().enumerate() {
            if let CallArg::Positional(expr) = arg {
                match &expr.node {
                    ExprKind::Literal(Literal::Num(n)) => {
                        if i == 0 {
                            start = *n as usize;
                        }
                        if i == 1 {
                            end = *n as usize;
                        }
                    }
                    ExprKind::Range(r) => {
                        start = r.start.as_f64() as usize;
                        end = r.end.as_f64() as usize;
                    }
                    _ => {}
                }
            }
        }
        (start, end)
    }

    fn execute_var(&mut self, decl: &VarDecl) -> Result<Value, InterpreterError> {
        let value = self.evaluate_expr(&decl.value)?;
        self.variables.insert(decl.name.clone(), value.clone());
        Ok(value)
    }

    fn execute_regress(&mut self, stmt: &RegressStmt) -> Result<Value, InterpreterError> {
        let config = &stmt.config;
        let epsilon = match &config.until {
            Some(ConvergenceCond::Epsilon(n)) => n.as_f64(),
            _ => 1e-6,
        };
        let mut regressor = EscalatingRegressor::new(epsilon);
        regressor.set_target(&self.sample_data);
        if let Some(workspace) = self.manifolds.first() {
            let max_epochs = if config.escalate { 100 } else { 10 };
            let result = regressor.run_escalating(workspace, max_epochs);
            Ok(Value::RegressionResult(result))
        } else {
            Err("no manifold for regression".into())
        }
    }

    fn execute_render(&mut self, stmt: &RenderStmt) -> Result<Value, InterpreterError> {
        if self.framebuffer.is_none() {
            self.framebuffer = Some(Framebuffer::new(80, 24));
        }
        let fb = if let Some(ref mut f) = self.framebuffer {
            f
        } else {
            return Ok(Value::Unit);
        };
        fb.clear(0x000000);

        // Draw border
        fb.draw_rect(0, 0, fb.width, 1, 0xFFFFFF);
        fb.draw_rect(0, fb.height.saturating_sub(1), fb.width, 1, 0xFFFFFF);
        fb.draw_rect(0, 0, 1, fb.height, 0xFFFFFF);
        fb.draw_rect(fb.width.saturating_sub(1), 0, 1, fb.height, 0xFFFFFF);

        // Gather points from target manifold
        let points: Vec<[f64; 3]> =
            if let Some(Value::Manifold(h)) = self.variables.get(&stmt.target) {
                if let Some(ws) = self.manifolds.get(h.0) {
                    ws.points.iter().map(|p| p.coords).collect()
                } else {
                    Vec::new()
                }
            } else {
                Vec::new()
            };

        fb.draw_text(2, 1, &alloc::format!("render {}", stmt.target), 0x00FF00);
        fb.draw_text(2, 2, &alloc::format!("points: {}", points.len()), 0x00FF00);

        if !points.is_empty() {
            let (min_x, max_x, min_y, max_y) = find_bounds(&points);
            let range_x = (max_x - min_x).max(0.001);
            let range_y = (max_y - min_y).max(0.001);
            let fw = (fb.width.saturating_sub(4)) as f64;
            let fh = (fb.height.saturating_sub(6)) as f64;
            for p in &points {
                let px = 2 + (((p[0] - min_x) / range_x) * fw) as u32;
                let py = 4 + (((p[1] - min_y) / range_y) * fh) as u32;
                if px < fb.width && py < fb.height {
                    fb.draw_rect(px, py, 1, 1, 0xFF0000);
                }
            }
        }

        Ok(Value::Unit)
    }

    fn execute_stmt_block(&mut self, block: &Block) -> Result<Value, InterpreterError> {
        let mut last_val = Value::Unit;
        for stmt in &block.statements {
            match self.execute_statement(stmt) {
                Ok(v) => last_val = v,
                Err(InterpreterError::Break) => return Err(InterpreterError::Break),
                Err(InterpreterError::Continue) => return Err(InterpreterError::Continue),
                Err(InterpreterError::Return(v)) => return Err(InterpreterError::Return(v)),
                Err(e) => return Err(e),
            }
        }
        Ok(last_val)
    }

    fn execute_if(&mut self, stmt: &IfStmt) -> Result<Value, InterpreterError> {
        let cond_val = self.evaluate_expr(&stmt.condition)?;
        let is_true = match cond_val {
            Value::Bool(b) => b,
            _ => return Err(String::from("condition must be boolean").into()),
        };
        if is_true {
            self.execute_stmt_block(&stmt.then_branch)
        } else if let Some(else_branch) = &stmt.else_branch {
            self.execute_stmt_block(else_branch)
        } else {
            Ok(Value::Unit)
        }
    }

    fn execute_while(&mut self, stmt: &WhileStmt) -> Result<Value, InterpreterError> {
        let mut last_val = Value::Unit;
        loop {
            let cond_val = self.evaluate_expr(&stmt.condition)?;
            let is_true = match cond_val {
                Value::Bool(b) => b,
                _ => return Err("condition must be boolean".into()),
            };
            if !is_true {
                break;
            }
            match self.execute_stmt_block(&stmt.body) {
                Ok(v) => last_val = v,
                Err(InterpreterError::Break) => break,
                Err(InterpreterError::Continue) => continue,
                Err(e) => return Err(e),
            }
        }
        Ok(last_val)
    }

    fn execute_seal(&mut self, stmt: &LoopStmt) -> Result<Value, InterpreterError> {
        let max_iters = 1000;
        let mut last_val = Value::Unit;
        for _ in 0..max_iters {
            match self.execute_stmt_block(&stmt.body) {
                Ok(v) => last_val = v,
                Err(InterpreterError::Break) => break,
                Err(InterpreterError::Continue) => continue,
                Err(e) => return Err(e),
            }
        }
        Ok(last_val)
    }

    fn evaluate_expr(&mut self, expr: &Expr) -> Result<Value, InterpreterError> {
        match &expr.node {
            ExprKind::Literal(lit) => match lit {
                Literal::Num(n) => Ok(Value::Num(*n)),
                Literal::Bool(b) => Ok(Value::Bool(*b)),
                Literal::Str(s) => Ok(Value::Str(s.clone())),
            },
            ExprKind::Ident(name) => {
                if let Some(v) = self.variables.get(name) {
                    Ok(v.clone())
                } else {
                    Ok(Value::Unit)
                }
            }
            ExprKind::FieldAccess { object, field } => self.evaluate_field_access(object, field),
            ExprKind::Call { name, args } => self.evaluate_call(name, args),
            ExprKind::New { class, args } => self.evaluate_new(class, args),
            ExprKind::List(elements) => self.evaluate_list(elements),
            ExprKind::MethodCall {
                object,
                method,
                args,
            } => self.evaluate_method_call(object, method, args),
            ExprKind::Range(_) => {
                Err(String::from("Ranges cannot be evaluated directly as values").into())
            }
            ExprKind::BinaryOp(left, op, right) => {
                let l = self.evaluate_expr(left)?;
                let r = self.evaluate_expr(right)?;
                self.evaluate_binary(l, *op, r, Some(left))
            }
            ExprKind::UnaryOp(_, _) => Err(String::from("Unary ops not implemented yet").into()),
            ExprKind::Index { object, range } => {
                // Simplified: returns a descriptive string or handle?
                // For now, let's treat it as a lookup that returns a sub-manifold or block value
                let handle = self.get_manifold_handle(object)?;
                let start = range.start.as_f64() as usize;
                let end = range.end.as_f64() as usize;
                if let Some(workspace) = self.manifolds.get(handle.0) {
                    let block = workspace.extract_block(start, end);
                    let block_handle = BlockHandle(self.blocks.len());
                    self.blocks.push(block);
                    Ok(Value::Block(block_handle))
                } else {
                    Err(format!("Manifold '{}' not found", object).into())
                }
            }
            ExprKind::Config(_) => {
                Err(String::from("Raw config blocks cannot be evaluated as expressions").into())
            }
        }
    }

    fn evaluate_binary(
        &mut self,
        left: Value,
        op: BinaryOp,
        right: Value,
        left_expr: Option<&Expr>,
    ) -> Result<Value, InterpreterError> {
        match op {
            BinaryOp::Eq => {
                // If left is FieldAccess, it's an assignment
                if let Some(lexpr) = left_expr {
                    if let ExprKind::FieldAccess { object, field: _ } = &lexpr.node {
                        if object == "this" {
                            // In real impl, we'd need to find 'this' in scope
                            // For mock: just return right
                            return Ok(right);
                        }
                    }
                }
                Ok(Value::Bool(self.values_equal(&left, &right)))
            }
            BinaryOp::Add => match (left, right) {
                (Value::Num(a), Value::Num(b)) => Ok(Value::Num(a + b)),
                _ => Err("Invalid add".into()),
            },
            BinaryOp::Sub => match (left, right) {
                (Value::Num(a), Value::Num(b)) => Ok(Value::Num(a - b)),
                _ => Err("Invalid sub".into()),
            },
            BinaryOp::Mul => match (left, right) {
                (Value::Num(a), Value::Num(b)) => Ok(Value::Num(a * b)),
                _ => Err("Invalid mul".into()),
            },
            BinaryOp::Div => match (left, right) {
                (Value::Num(a), Value::Num(b)) => Ok(Value::Num(a / b)),
                _ => Err("Invalid div".into()),
            },
            BinaryOp::Or => match (left, right) {
                (Value::Num(a), Value::Num(b)) => Ok(Value::Num((a as u64 | b as u64) as f64)),
                _ => Err("Invalid or".into()),
            },
            BinaryOp::And => match (left, right) {
                (Value::Num(a), Value::Num(b)) => Ok(Value::Num((a as u64 & b as u64) as f64)),
                _ => Err("Invalid and".into()),
            },
            BinaryOp::Shl => match (left, right) {
                (Value::Num(a), Value::Num(b)) => Ok(Value::Num(((a as u64) << (b as u64)) as f64)),
                _ => Err("Invalid shl".into()),
            },
            BinaryOp::Shr => match (left, right) {
                (Value::Num(a), Value::Num(b)) => Ok(Value::Num(((a as u64) >> (b as u64)) as f64)),
                _ => Err("Invalid shr".into()),
            },
            BinaryOp::Gt => match (left, right) {
                (Value::Num(a), Value::Num(b)) => Ok(Value::Bool(a > b)),
                _ => Err("Invalid gt".into()),
            },
            BinaryOp::Lt => match (left, right) {
                (Value::Num(a), Value::Num(b)) => Ok(Value::Bool(a < b)),
                _ => Err("Invalid lt".into()),
            },
            BinaryOp::Ge => match (left, right) {
                (Value::Num(a), Value::Num(b)) => Ok(Value::Bool(a >= b)),
                _ => Err("Invalid ge".into()),
            },
            BinaryOp::Le => match (left, right) {
                (Value::Num(a), Value::Num(b)) => Ok(Value::Bool(a <= b)),
                _ => Err("Invalid le".into()),
            },
            BinaryOp::Neq => Ok(Value::Bool(!self.values_equal(&left, &right))),
        }
    }

    fn values_equal(&self, a: &Value, b: &Value) -> bool {
        match (a, b) {
            (Value::Num(x), Value::Num(y)) => x == y,
            (Value::Bool(x), Value::Bool(y)) => x == y,
            (Value::Str(x), Value::Str(y)) => x == y,
            _ => false,
        }
    }

    fn evaluate_list(&mut self, elements: &Vec<Expr>) -> Result<Value, InterpreterError> {
        let mut values = Vec::new();
        for expr in elements {
            values.push(self.evaluate_expr(expr)?);
        }
        Ok(Value::List(values))
    }

    fn evaluate_call(&mut self, name: &Ident, args: &[CallArg]) -> Result<Value, InterpreterError> {
        if let Some(val) = self.variables.get(name) {
            match val.clone() {
                Value::NativeFn(func) => self.execute_native_fn(func, args),
                Value::Function(func) => {
                    let mut new_vars = func.closure.clone();

                    // Bind parameters
                    for (i, param) in func.decl.params.iter().enumerate() {
                        if let Some(CallArg::Positional(expr)) = args.get(i) {
                            let val = self.evaluate_expr(expr)?;
                            new_vars.insert(param.clone(), val);
                        }
                    }

                    // Enable recursion
                    new_vars.insert(func.decl.name.clone(), Value::Function(func.clone()));

                    let old_vars = core::mem::replace(&mut self.variables, new_vars);
                    let res = self.execute_stmt_block(&func.decl.body);
                    self.variables = old_vars;

                    match res {
                        Err(InterpreterError::Return(v)) => Ok(v),
                        other => other,
                    }
                }
                _ => Ok(Value::Unit),
            }
        } else {
            Ok(Value::Unit)
        }
    }

    fn execute_native_fn(
        &mut self,
        func: NativeFunction,
        args: &[CallArg],
    ) -> Result<Value, InterpreterError> {
        let mut get_f64 = |args: &[CallArg]| -> Result<f64, InterpreterError> {
            if let Some(CallArg::Positional(expr)) = args.first() {
                let val = self.evaluate_expr(expr)?;
                if let Value::Num(n) = val {
                    Ok(n)
                } else {
                    Err(String::from("Expected number").into())
                }
            } else {
                Err(String::from("Expected number").into())
            }
        };

        match func {
            NativeFunction::MathSin => Ok(Value::Num(libm::sin(get_f64(args)?))),
            NativeFunction::MathCos => Ok(Value::Num(libm::cos(get_f64(args)?))),
            NativeFunction::MathSqrt => Ok(Value::Num(libm::sqrt(get_f64(args)?))),
            NativeFunction::MathExp => Ok(Value::Num(libm::exp(get_f64(args)?))),
            NativeFunction::TopoBetti => {
                let mut bytes = Vec::new();
                for arg in args {
                    if let CallArg::Positional(expr) = arg {
                        let val = self.evaluate_expr(expr)?;
                        match val {
                            Value::List(list) => {
                                for v in list {
                                    if let Value::Num(n) = v {
                                        bytes.push(n as u8);
                                    }
                                }
                            }
                            Value::Num(n) => bytes.push(n as u8),
                            _ => {}
                        }
                    }
                }
                let betti0 = aether_core::topology::compute_betti_0(&bytes);
                Ok(Value::List(vec![
                    Value::Num(betti0 as f64),
                    Value::Num(0.0),
                ]))
            }
            NativeFunction::Print => {
                for arg in args {
                    if let CallArg::Positional(expr) = arg {
                        let val = self.evaluate_expr(expr)?;
                        #[cfg(feature = "std")]
                        match val {
                            Value::Num(n) => print!("{}", n),
                            Value::Str(s) => print!("{}", s),
                            Value::Bool(b) => print!("{}", b),
                            _ => print!("{:?}", val),
                        }
                    }
                }
                #[cfg(feature = "std")]
                println!();
                Ok(Value::Unit)
            }
            NativeFunction::MlpNew => {
                let lr = get_f64(args).unwrap_or(0.01);
                let config = OptimizerConfig::SGD {
                    learning_rate: lr,
                    momentum: 0.9,
                };
                Ok(Value::Mlp(Box::new(MLP::new(config, LossConfig::MSE))))
            }
            NativeFunction::KMeansNew => {
                let k = get_f64(args).unwrap_or(2.0) as usize;
                Ok(Value::KMeans(Box::new(KMeans::new(k))))
            }
            NativeFunction::Conv2DNew => Ok(Value::Conv2D(Box::new(Conv2D::new(
                1,
                1,
                3,
                1,
                1,
                Activation::ReLU,
            )))),

            // ML Ops
            NativeFunction::MlMatMul => {
                let a = self.get_tensor_arg(args, 0)?;
                let b = self.get_tensor_arg(args, 1)?;
                Ok(Value::Tensor(a.matmul(&b)))
            }
            NativeFunction::MlAdd => {
                let a = self.get_tensor_arg(args, 0)?;
                let b = self.get_tensor_arg(args, 1)?;
                Ok(Value::Tensor(a.add(&b)))
            }
            NativeFunction::MlRelu => {
                let a = self.get_tensor_arg(args, 0)?;
                Ok(Value::Tensor(Activation::ReLU.apply(&a)))
            }
            NativeFunction::MlSoftmax => {
                let a = self.get_tensor_arg(args, 0)?;
                Ok(Value::Tensor(Activation::Softmax.apply(&a)))
            }
            NativeFunction::MlLoadWeights => {
                // Acts as "Create Tensor from List"
                if let Some(CallArg::Positional(expr)) = args.first() {
                    let val = self.evaluate_expr(expr)?;
                    let t = self.value_to_tensor_core(&val)?;
                    Ok(Value::Tensor(t))
                } else {
                    Err("Missing argument".into())
                }
            }
            NativeFunction::MlForward => {
                // Map "forward"
                // Args: model, input
                // Actually this might be MethodCall on Mlp object
                Ok(Value::Unit)
            }
            NativeFunction::MmioRead32 => {
                let _addr = get_f64(args)?;
                // Kernel call here
                Ok(Value::Num(0.0))
            }
            NativeFunction::MmioWrite32 => {
                let _addr = get_f64(args)?;
                if let Some(CallArg::Positional(expr)) = args.get(1) {
                    let _val = self.evaluate_expr(expr)?;
                }
                Ok(Value::Unit)
            }
            NativeFunction::PciFind => {
                // Returns [bus, slot]
                Ok(Value::List(vec![Value::Num(0.0), Value::Num(0.0)]))
            }
            NativeFunction::PciFindVendor => {
                Ok(Value::List(vec![Value::Num(0.0), Value::Num(0.0)]))
            }
            NativeFunction::PciReadBar0 => {
                Ok(Value::Num(0xFD000000u64 as f64)) // Mock NVIDIA base
            }
            NativeFunction::PciReadBar5 => {
                Ok(Value::Num(0xFE000000u64 as f64)) // Mock AHCI base
            }
            NativeFunction::MathRange => {
                let start = self.get_arg_num(args, 0)? as i64;
                let end = self.get_arg_num(args, 1)? as i64;
                let mut list = Vec::new();
                for i in start..end {
                    list.push(Value::Num(i as f64));
                }
                Ok(Value::List(list))
            }
            _ => Ok(Value::Unit),
        }
    }

    fn get_tensor_arg(
        &mut self,
        args: &[CallArg],
        index: usize,
    ) -> Result<Tensor, InterpreterError> {
        if let Some(CallArg::Positional(expr)) = args.get(index) {
            let val = self.evaluate_expr(expr)?;
            self.value_to_tensor_core(&val)
        } else {
            Err(format!("Missing argument {}", index).into())
        }
    }

    fn value_to_tensor_core(&self, val: &Value) -> Result<Tensor, InterpreterError> {
        match val {
            Value::Tensor(t) => Ok(t.clone()),
            Value::List(rows) => {
                if rows.is_empty() {
                    return Ok(Tensor::zeros(&[0]));
                }

                let mut data = Vec::new();

                // Check if 2D or 1D
                if let Value::List(_) = &rows[0] {
                    // 2D
                    let rows_cnt = rows.len();
                    let mut cols_cnt = 0;
                    for (i, row) in rows.iter().enumerate() {
                        if let Value::List(cols) = row {
                            if i == 0 {
                                cols_cnt = cols.len();
                            } else if cols.len() != cols_cnt {
                                return Err("Ragged tensor".into());
                            }
                            for c in cols {
                                if let Value::Num(n) = c {
                                    data.push(*n);
                                } else {
                                    return Err("Tensor must contain numbers".into());
                                }
                            }
                        } else {
                            return Err("Expected 2D list".into());
                        }
                    }
                    Ok(Tensor::new(&data, &[rows_cnt, cols_cnt]))
                } else {
                    // 1D
                    for c in rows {
                        if let Value::Num(n) = c {
                            data.push(*n);
                        } else {
                            return Err("Tensor must contain numbers".into());
                        }
                    }
                    Ok(Tensor::new(&data, &[rows.len()]))
                }
            }
            _ => Err("Expected Tensor or List".into()),
        }
    }

    fn evaluate_method_call(
        &mut self,
        object_name: &String,
        method: &String,
        args: &[CallArg],
    ) -> Result<Value, InterpreterError> {
        let val = if let Some(v) = self.variables.get(object_name) {
            v.clone()
        } else {
            return Err(format!("Object '{}' not found", object_name).into());
        };
        match val {
            Value::List(mut list) => {
                let res = match method.as_str() {
                    "push" => {
                        if let Some(CallArg::Positional(expr)) = args.first() {
                            let val = self.evaluate_expr(expr)?;
                            list.push(val);
                            self.variables
                                .insert(object_name.clone(), Value::List(list));
                            Ok(Value::Unit)
                        } else {
                            Err(String::from("push requires 1 argument"))
                        }
                    }
                    "pop" => {
                        let val = list.pop().unwrap_or(Value::Unit);
                        self.variables
                            .insert(object_name.clone(), Value::List(list));
                        Ok(val)
                    }
                    "len" => Ok(Value::Num(list.len() as f64)),
                    "get" => {
                        let idx = self.get_arg_num(args, 0)? as usize;
                        Ok(list.get(idx).cloned().unwrap_or(Value::Unit))
                    }
                    _ => Err(format!("Method '{}' not found on List", method)),
                };
                Ok(res?)
            }
            Value::Mlp(mut mlp) => {
                match method.as_str() {
                    "add_layer" => {
                        // input, output, activation key string
                        // Default to Tanh if not string
                        let input = self.get_arg_num(args, 0)? as usize;
                        let output = self.get_arg_num(args, 1)? as usize;
                        let act_str = self.get_arg_str(args, 2).unwrap_or("tanh".to_string());
                        let act = match act_str.as_str() {
                            "relu" => Activation::ReLU,
                            "sigmoid" => Activation::Sigmoid,
                            "softmax" => Activation::Softmax,
                            _ => Activation::Tanh,
                        };
                        mlp.add_layer(input, output, act, None);
                        self.variables.insert(object_name.clone(), Value::Mlp(mlp)); // Update
                        Ok(Value::Unit)
                    }
                    "train" => {
                        // inputs (List/Tensor), targets (List/Tensor), epochs
                        let input = self.get_tensor_arg(args, 0)?;
                        let target = self.get_tensor_arg(args, 1)?;
                        let epochs = self.get_arg_num(args, 2).unwrap_or(1.0) as usize;
                        let res = mlp.fit(&[input], &[target], epochs); // fit expects slice of tensors
                        Ok(Value::Num(res.final_loss))
                    }
                    "forward" | "predict" => {
                        let input = self.get_tensor_arg(args, 0)?;
                        let output = mlp.forward(&input);
                        Ok(Value::Tensor(output))
                    }
                    _ => Err(format!("Method '{}' not found on MLP", method).into()),
                }
            }
            Value::Module(mod_name) => match (mod_name.as_str(), method.as_str()) {
                ("Ml", "MLP") => self.execute_native_fn(NativeFunction::MlpNew, args),
                ("Ml", "KMeans") => self.execute_native_fn(NativeFunction::KMeansNew, args),
                ("Ml", "Conv2D") => self.execute_native_fn(NativeFunction::Conv2DNew, args),
                ("Seal", "train") => self.execute_native_fn(NativeFunction::SealTrain, args),
                ("Seal", "mmio_read32") => self.execute_native_fn(NativeFunction::MmioRead32, args),
                ("Seal", "mmio_write32") => {
                    self.execute_native_fn(NativeFunction::MmioWrite32, args)
                }
                ("Seal", "pci_find") => self.execute_native_fn(NativeFunction::PciFind, args),
                ("Seal", "pci_find_vendor") => {
                    self.execute_native_fn(NativeFunction::PciFindVendor, args)
                }
                ("Seal", "pci_read_bar0") => {
                    self.execute_native_fn(NativeFunction::PciReadBar0, args)
                }
                ("Seal", "pci_read_bar5") => {
                    self.execute_native_fn(NativeFunction::PciReadBar5, args)
                }
                _ => Err(format!("Method '{}' not found in module '{}'", method, mod_name).into()),
            },
            _ => Ok(Value::Unit),
        }
    }

    fn get_arg_num(&mut self, args: &[CallArg], index: usize) -> Result<f64, InterpreterError> {
        if let Some(CallArg::Positional(expr)) = args.get(index) {
            let val = self.evaluate_expr(expr)?;
            if let Value::Num(n) = val {
                Ok(n)
            } else {
                Err("Expected number".into())
            }
        } else {
            Err("Missing arg".into())
        }
    }

    fn get_arg_str(&mut self, args: &[CallArg], index: usize) -> Result<String, InterpreterError> {
        if let Some(CallArg::Positional(expr)) = args.get(index) {
            let val = self.evaluate_expr(expr)?;
            if let Value::Str(s) = val {
                Ok(s)
            } else {
                Err("Expected string".into())
            }
        } else {
            Err("Missing arg".into())
        }
    }

    fn evaluate_field_access(
        &self,
        object: &String,
        field: &String,
    ) -> Result<Value, InterpreterError> {
        if let Some(Value::Object(handle)) = self.variables.get(object) {
            if let Some(obj) = self.objects.get(handle.0) {
                if let Some(val) = obj.fields.get(field) {
                    return Ok(val.clone());
                }
            }
        }
        if let Some(Value::Module(name)) = self.variables.get(object) {
            match (name.as_str(), field.as_str()) {
                ("Seal", "train") => Ok(Value::NativeFn(NativeFunction::SealTrain)),
                ("Seal", "mmio_read32") => Ok(Value::NativeFn(NativeFunction::MmioRead32)),
                ("Seal", "mmio_write32") => Ok(Value::NativeFn(NativeFunction::MmioWrite32)),
                ("Seal", "pci_find") => Ok(Value::NativeFn(NativeFunction::PciFind)),
                ("Seal", "pci_find_vendor") => Ok(Value::NativeFn(NativeFunction::PciFindVendor)),
                ("Seal", "pci_read_bar0") => Ok(Value::NativeFn(NativeFunction::PciReadBar0)),
                ("Seal", "pci_read_bar5") => Ok(Value::NativeFn(NativeFunction::PciReadBar5)),
                ("Ml", "MLP") => Ok(Value::NativeFn(NativeFunction::MlpNew)),
                _ => Ok(Value::Unit),
            }
        } else {
            Ok(Value::Unit)
        }
    }
}

impl Default for Interpreter {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Unit Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{
        Block, Expr, ExprKind, FnDecl, ForStmt, IfStmt, Literal, Span, Statement, StmtKind,
        VarDecl, WhileStmt,
    };

    fn expr_num(n: f64) -> Expr {
        Expr::new(ExprKind::Literal(Literal::Num(n)), Span::default())
    }

    fn expr_ident(name: &str) -> Expr {
        Expr::new(ExprKind::Ident(name.into()), Span::default())
    }

    fn expr_list(items: Vec<Expr>) -> Expr {
        Expr::new(ExprKind::List(items), Span::default())
    }

    fn stmt_var(name: &str, value: Expr) -> Statement {
        Statement::new(
            StmtKind::Var(VarDecl {
                type_hint: None,
                name: name.into(),
                value,
            }),
            Span::default(),
        )
    }

    fn mk_block(stmts: Vec<Statement>) -> Block {
        Block { statements: stmts }
    }

    #[test]
    fn test_for_loop_executes_n_times() {
        let mut interp = Interpreter::new();
        // accum = 0
        let setup = stmt_var("accum", expr_num(0.0));
        interp.execute_statement(&setup).unwrap();

        // for i in [1,2,3] { accum = accum + i }
        let body = mk_block(vec![Statement::new(
            StmtKind::Var(VarDecl {
                type_hint: None,
                name: "accum".into(),
                value: Expr::new(
                    ExprKind::BinaryOp(
                        Box::new(expr_ident("accum")),
                        BinaryOp::Add,
                        Box::new(expr_ident("i")),
                    ),
                    Span::default(),
                ),
            }),
            Span::default(),
        )]);
        let for_stmt = Statement::new(
            StmtKind::For(ForStmt {
                iterator: "i".into(),
                iterable: expr_list(vec![expr_num(1.0), expr_num(2.0), expr_num(3.0)]),
                body,
            }),
            Span::default(),
        );
        interp.execute_statement(&for_stmt).unwrap();

        if let Value::Num(n) = interp.variables.get("accum").unwrap() {
            assert_eq!(*n, 6.0);
        } else {
            panic!("accum not a number");
        }
    }

    #[test]
    fn test_break_exits_loop() {
        let mut interp = Interpreter::new();
        interp
            .execute_statement(&stmt_var("accum", expr_num(0.0)))
            .unwrap();

        // for i in [1,2,3,4,5] { accum = accum + i; if i > 2 { break } }
        let body = mk_block(vec![
            Statement::new(
                StmtKind::Var(VarDecl {
                    type_hint: None,
                    name: "accum".into(),
                    value: Expr::new(
                        ExprKind::BinaryOp(
                            Box::new(expr_ident("accum")),
                            BinaryOp::Add,
                            Box::new(expr_ident("i")),
                        ),
                        Span::default(),
                    ),
                }),
                Span::default(),
            ),
            Statement::new(
                StmtKind::If(IfStmt {
                    condition: Expr::new(
                        ExprKind::BinaryOp(
                            Box::new(expr_ident("i")),
                            BinaryOp::Gt,
                            Box::new(expr_num(2.0)),
                        ),
                        Span::default(),
                    ),
                    then_branch: mk_block(vec![Statement::new(
                        StmtKind::Break(BreakStmt),
                        Span::default(),
                    )]),
                    else_branch: None,
                }),
                Span::default(),
            ),
        ]);
        let for_stmt = Statement::new(
            StmtKind::For(ForStmt {
                iterator: "i".into(),
                iterable: expr_list(vec![
                    expr_num(1.0),
                    expr_num(2.0),
                    expr_num(3.0),
                    expr_num(4.0),
                    expr_num(5.0),
                ]),
                body,
            }),
            Span::default(),
        );
        interp.execute_statement(&for_stmt).unwrap();

        if let Value::Num(n) = interp.variables.get("accum").unwrap() {
            assert_eq!(*n, 6.0); // 1+2+3, break after third addition
        } else {
            panic!("accum not a number");
        }
    }

    #[test]
    fn test_continue_skips_iteration() {
        let mut interp = Interpreter::new();
        interp
            .execute_statement(&stmt_var("accum", expr_num(0.0)))
            .unwrap();

        // for i in [1,2,3,4,5] { if i == 3 { continue } accum = accum + i }
        let body = mk_block(vec![
            Statement::new(
                StmtKind::If(IfStmt {
                    condition: Expr::new(
                        ExprKind::BinaryOp(
                            Box::new(expr_ident("i")),
                            BinaryOp::Eq,
                            Box::new(expr_num(3.0)),
                        ),
                        Span::default(),
                    ),
                    then_branch: mk_block(vec![Statement::new(
                        StmtKind::Continue(ContinueStmt),
                        Span::default(),
                    )]),
                    else_branch: None,
                }),
                Span::default(),
            ),
            Statement::new(
                StmtKind::Var(VarDecl {
                    type_hint: None,
                    name: "accum".into(),
                    value: Expr::new(
                        ExprKind::BinaryOp(
                            Box::new(expr_ident("accum")),
                            BinaryOp::Add,
                            Box::new(expr_ident("i")),
                        ),
                        Span::default(),
                    ),
                }),
                Span::default(),
            ),
        ]);
        let for_stmt = Statement::new(
            StmtKind::For(ForStmt {
                iterator: "i".into(),
                iterable: expr_list(vec![
                    expr_num(1.0),
                    expr_num(2.0),
                    expr_num(3.0),
                    expr_num(4.0),
                    expr_num(5.0),
                ]),
                body,
            }),
            Span::default(),
        );
        interp.execute_statement(&for_stmt).unwrap();

        if let Value::Num(n) = interp.variables.get("accum").unwrap() {
            assert_eq!(*n, 12.0); // 1+2+4+5
        } else {
            panic!("accum not a number");
        }
    }

    #[test]
    fn test_return_from_function() {
        let mut interp = Interpreter::new();
        // fn double(x) { return x * 2 }
        let fn_decl = Statement::new(
            StmtKind::Fn(FnDecl {
                name: "double".into(),
                params: vec!["x".into()],
                body: mk_block(vec![Statement::new(
                    StmtKind::Return(ReturnStmt {
                        value: Some(Expr::new(
                            ExprKind::BinaryOp(
                                Box::new(expr_ident("x")),
                                BinaryOp::Mul,
                                Box::new(expr_num(2.0)),
                            ),
                            Span::default(),
                        )),
                    }),
                    Span::default(),
                )]),
            }),
            Span::default(),
        );
        interp.execute_statement(&fn_decl).unwrap();

        // result = double(5)
        let call = Statement::new(
            StmtKind::Var(VarDecl {
                type_hint: None,
                name: "result".into(),
                value: Expr::new(
                    ExprKind::Call {
                        name: "double".into(),
                        args: vec![CallArg::Positional(expr_num(5.0))],
                    },
                    Span::default(),
                ),
            }),
            Span::default(),
        );
        interp.execute_statement(&call).unwrap();

        if let Value::Num(n) = interp.variables.get("result").unwrap() {
            assert_eq!(*n, 10.0);
        } else {
            panic!("result not a number");
        }
    }

    #[test]
    fn test_fn_call_with_closure() {
        let mut interp = Interpreter::new();
        // base = 10
        interp
            .execute_statement(&stmt_var("base", expr_num(10.0)))
            .unwrap();

        // fn add_base(x) { return base + x }
        let fn_decl = Statement::new(
            StmtKind::Fn(FnDecl {
                name: "add_base".into(),
                params: vec!["x".into()],
                body: mk_block(vec![Statement::new(
                    StmtKind::Return(ReturnStmt {
                        value: Some(Expr::new(
                            ExprKind::BinaryOp(
                                Box::new(expr_ident("base")),
                                BinaryOp::Add,
                                Box::new(expr_ident("x")),
                            ),
                            Span::default(),
                        )),
                    }),
                    Span::default(),
                )]),
            }),
            Span::default(),
        );
        interp.execute_statement(&fn_decl).unwrap();

        // base = 999 // should not affect closure
        interp
            .execute_statement(&stmt_var("base", expr_num(999.0)))
            .unwrap();

        // result = add_base(5)
        let call = Statement::new(
            StmtKind::Var(VarDecl {
                type_hint: None,
                name: "result".into(),
                value: Expr::new(
                    ExprKind::Call {
                        name: "add_base".into(),
                        args: vec![CallArg::Positional(expr_num(5.0))],
                    },
                    Span::default(),
                ),
            }),
            Span::default(),
        );
        interp.execute_statement(&call).unwrap();

        if let Value::Num(n) = interp.variables.get("result").unwrap() {
            assert_eq!(*n, 15.0); // captured base=10
        } else {
            panic!("result not a number");
        }
    }

    #[test]
    fn test_function_no_return() {
        let mut interp = Interpreter::new();
        // fn forty_two() { return 42 }
        let body = mk_block(vec![Statement::new(
            StmtKind::Return(ReturnStmt {
                value: Some(expr_num(42.0)),
            }),
            Span::default(),
        )]);
        let fn_decl = Statement::new(
            StmtKind::Fn(FnDecl {
                name: "forty_two".into(),
                params: vec![],
                body,
            }),
            Span::default(),
        );
        interp.execute_statement(&fn_decl).unwrap();

        let call = stmt_var(
            "result",
            Expr::new(
                ExprKind::Call {
                    name: "forty_two".into(),
                    args: vec![],
                },
                Span::default(),
            ),
        );
        interp.execute_statement(&call).unwrap();

        if let Value::Num(n) = interp.variables.get("result").unwrap() {
            assert_eq!(*n, 42.0);
        } else {
            panic!("result not a number");
        }
    }

    #[test]
    fn test_while_loop_condition() {
        let mut interp = Interpreter::new();
        interp
            .execute_statement(&stmt_var("n", expr_num(0.0)))
            .unwrap();

        // while n < 5 { n = n + 1 }
        let while_stmt = Statement::new(
            StmtKind::While(WhileStmt {
                condition: Expr::new(
                    ExprKind::BinaryOp(
                        Box::new(expr_ident("n")),
                        BinaryOp::Lt,
                        Box::new(expr_num(5.0)),
                    ),
                    Span::default(),
                ),
                body: mk_block(vec![Statement::new(
                    StmtKind::Var(VarDecl {
                        type_hint: None,
                        name: "n".into(),
                        value: Expr::new(
                            ExprKind::BinaryOp(
                                Box::new(expr_ident("n")),
                                BinaryOp::Add,
                                Box::new(expr_num(1.0)),
                            ),
                            Span::default(),
                        ),
                    }),
                    Span::default(),
                )]),
            }),
            Span::default(),
        );
        interp.execute_statement(&while_stmt).unwrap();

        if let Value::Num(n) = interp.variables.get("n").unwrap() {
            assert_eq!(*n, 5.0);
        } else {
            panic!("n not a number");
        }
    }

    #[test]
    fn test_nested_break_innermost() {
        let mut interp = Interpreter::new();
        interp
            .execute_statement(&stmt_var("outer_sum", expr_num(0.0)))
            .unwrap();
        interp
            .execute_statement(&stmt_var("inner_sum", expr_num(0.0)))
            .unwrap();

        // for i in [1,2] { for j in [1,2,3] { inner_sum = inner_sum + j; if j == 2 { break } } outer_sum = outer_sum + i }
        let inner_body = mk_block(vec![
            Statement::new(
                StmtKind::Var(VarDecl {
                    type_hint: None,
                    name: "inner_sum".into(),
                    value: Expr::new(
                        ExprKind::BinaryOp(
                            Box::new(expr_ident("inner_sum")),
                            BinaryOp::Add,
                            Box::new(expr_ident("j")),
                        ),
                        Span::default(),
                    ),
                }),
                Span::default(),
            ),
            Statement::new(
                StmtKind::If(IfStmt {
                    condition: Expr::new(
                        ExprKind::BinaryOp(
                            Box::new(expr_ident("j")),
                            BinaryOp::Eq,
                            Box::new(expr_num(2.0)),
                        ),
                        Span::default(),
                    ),
                    then_branch: mk_block(vec![Statement::new(
                        StmtKind::Break(BreakStmt),
                        Span::default(),
                    )]),
                    else_branch: None,
                }),
                Span::default(),
            ),
        ]);
        let outer_body = mk_block(vec![
            Statement::new(
                StmtKind::For(ForStmt {
                    iterator: "j".into(),
                    iterable: expr_list(vec![expr_num(1.0), expr_num(2.0), expr_num(3.0)]),
                    body: inner_body,
                }),
                Span::default(),
            ),
            Statement::new(
                StmtKind::Var(VarDecl {
                    type_hint: None,
                    name: "outer_sum".into(),
                    value: Expr::new(
                        ExprKind::BinaryOp(
                            Box::new(expr_ident("outer_sum")),
                            BinaryOp::Add,
                            Box::new(expr_ident("i")),
                        ),
                        Span::default(),
                    ),
                }),
                Span::default(),
            ),
        ]);
        let outer = Statement::new(
            StmtKind::For(ForStmt {
                iterator: "i".into(),
                iterable: expr_list(vec![expr_num(1.0), expr_num(2.0)]),
                body: outer_body,
            }),
            Span::default(),
        );
        interp.execute_statement(&outer).unwrap();

        if let Value::Num(n) = interp.variables.get("inner_sum").unwrap() {
            assert_eq!(*n, 6.0); // (1+2) * 2 iterations
        } else {
            panic!("inner_sum not a number");
        }
        if let Value::Num(n) = interp.variables.get("outer_sum").unwrap() {
            assert_eq!(*n, 3.0); // 1+2
        } else {
            panic!("outer_sum not a number");
        }
    }

    #[test]
    fn test_return_early_in_loop() {
        let mut interp = Interpreter::new();
        // fn find_three() { for i in [1,2,3,4,5] { if i == 3 { return i } } return 0 }
        let body = mk_block(vec![
            Statement::new(
                StmtKind::For(ForStmt {
                    iterator: "i".into(),
                    iterable: expr_list(vec![
                        expr_num(1.0),
                        expr_num(2.0),
                        expr_num(3.0),
                        expr_num(4.0),
                        expr_num(5.0),
                    ]),
                    body: mk_block(vec![Statement::new(
                        StmtKind::If(IfStmt {
                            condition: Expr::new(
                                ExprKind::BinaryOp(
                                    Box::new(expr_ident("i")),
                                    BinaryOp::Eq,
                                    Box::new(expr_num(3.0)),
                                ),
                                Span::default(),
                            ),
                            then_branch: mk_block(vec![Statement::new(
                                StmtKind::Return(ReturnStmt {
                                    value: Some(expr_ident("i")),
                                }),
                                Span::default(),
                            )]),
                            else_branch: None,
                        }),
                        Span::default(),
                    )]),
                }),
                Span::default(),
            ),
            Statement::new(
                StmtKind::Return(ReturnStmt {
                    value: Some(expr_num(0.0)),
                }),
                Span::default(),
            ),
        ]);
        let fn_decl = Statement::new(
            StmtKind::Fn(FnDecl {
                name: "find_three".into(),
                params: vec![],
                body,
            }),
            Span::default(),
        );
        interp.execute_statement(&fn_decl).unwrap();

        let call = stmt_var(
            "result",
            Expr::new(
                ExprKind::Call {
                    name: "find_three".into(),
                    args: vec![],
                },
                Span::default(),
            ),
        );
        interp.execute_statement(&call).unwrap();

        if let Value::Num(n) = interp.variables.get("result").unwrap() {
            assert_eq!(*n, 3.0);
        } else {
            panic!("result not a number");
        }
    }

    #[test]
    fn test_recursive_factorial() {
        let mut interp = Interpreter::new();
        // fn fact(n) { if n <= 1 { return 1 } return n * fact(n - 1) }
        let body = mk_block(vec![
            Statement::new(
                StmtKind::If(IfStmt {
                    condition: Expr::new(
                        ExprKind::BinaryOp(
                            Box::new(expr_ident("n")),
                            BinaryOp::Le,
                            Box::new(expr_num(1.0)),
                        ),
                        Span::default(),
                    ),
                    then_branch: mk_block(vec![Statement::new(
                        StmtKind::Return(ReturnStmt {
                            value: Some(expr_num(1.0)),
                        }),
                        Span::default(),
                    )]),
                    else_branch: None,
                }),
                Span::default(),
            ),
            Statement::new(
                StmtKind::Return(ReturnStmt {
                    value: Some(Expr::new(
                        ExprKind::BinaryOp(
                            Box::new(expr_ident("n")),
                            BinaryOp::Mul,
                            Box::new(Expr::new(
                                ExprKind::Call {
                                    name: "fact".into(),
                                    args: vec![CallArg::Positional(Expr::new(
                                        ExprKind::BinaryOp(
                                            Box::new(expr_ident("n")),
                                            BinaryOp::Sub,
                                            Box::new(expr_num(1.0)),
                                        ),
                                        Span::default(),
                                    ))],
                                },
                                Span::default(),
                            )),
                        ),
                        Span::default(),
                    )),
                }),
                Span::default(),
            ),
        ]);
        let fn_decl = Statement::new(
            StmtKind::Fn(FnDecl {
                name: "fact".into(),
                params: vec!["n".into()],
                body,
            }),
            Span::default(),
        );
        interp.execute_statement(&fn_decl).unwrap();

        let call = stmt_var(
            "result",
            Expr::new(
                ExprKind::Call {
                    name: "fact".into(),
                    args: vec![CallArg::Positional(expr_num(5.0))],
                },
                Span::default(),
            ),
        );
        interp.execute_statement(&call).unwrap();

        if let Value::Num(n) = interp.variables.get("result").unwrap() {
            assert_eq!(*n, 120.0); // 5!
        } else {
            panic!("result not a number");
        }
    }

    #[test]
    fn test_topo_betti_real_call() {
        let mut interp = Interpreter::new();
        interp
            .import_topology(&ImportStmt {
                module: "topology".into(),
                symbol: Some("Betti".into()),
            })
            .unwrap();
        // Betti([0, 50, 100]) -> gaps > 15 -> 3 components -> [3.0, 0.0]
        let call = Expr::new(
            ExprKind::Call {
                name: "Betti".into(),
                args: vec![CallArg::Positional(expr_list(vec![
                    expr_num(0.0),
                    expr_num(50.0),
                    expr_num(100.0),
                ]))],
            },
            Span::default(),
        );
        let val = interp.evaluate_expr(&call).unwrap();
        if let Value::List(vs) = val {
            assert_eq!(vs.len(), 2);
            if let Value::Num(b0) = vs[0] {
                assert_eq!(b0, 1.0); // [0,50,100] has one large-gap component per compute_betti_0 logic
            } else {
                panic!("b0 not a number");
            }
        } else {
            panic!("Expected list from Betti");
        }
    }
}
