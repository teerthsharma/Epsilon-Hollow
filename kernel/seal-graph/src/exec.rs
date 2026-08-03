// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Validate-then-run interpreter for a loaded [`Artifact`].
//!
//! [`Graph::from_artifact`] walks the ops once, statically infers every value's
//! shape, and refuses the whole graph on the first disagreement. Only a graph
//! that passed that walk can be run, so [`Graph::run`] never stops halfway
//! through with a partially written value store.
//!
//! Tensors are [`aether_core::ml::Tensor`] — this crate defines no tensor type
//! of its own. `MatMul`, `Add`, `Mul`, `Transpose` and the element-wise
//! activations delegate to `aether-core`; the shape checks in front of them
//! exist because those routines assert rather than return errors.
//!
//! # Determinism
//!
//! Every kernel is a fixed-order scalar loop over `f64`. There is no threading,
//! no reduction reordering, and no compiler fast-math. The same artifact and
//! the same inputs therefore produce bit-identical outputs, which
//! [`tests::run_is_bit_deterministic`] checks by comparing `to_bits`.
//!
//! # What this executor does not do
//!
//! * **No autograd, no training, no optimizer.** Forward evaluation only.
//! * **No fusion, no constant folding, no memory reuse.** Every op allocates
//!   its output. A graph's peak memory is the sum of all its intermediates.
//! * **No batching semantics beyond what the shapes say.** `Softmax` runs over
//!   the trailing axis; `Conv2d` and `MaxPool2d` assume NCHW.
//! * **No `Conv2d` groups, dilation, or asymmetric padding**, and no padding at
//!   all on `MaxPool2d`.
//! * **No NaN or overflow policy.** Arithmetic follows IEEE-754 and a NaN in
//!   the input propagates to the output silently.
//!
//! `aether_core::ml::convolution::Conv2D` was considered for `Conv2d` and not
//! used: its weights live in fixed `[[[[f64; 5]; 5]; 3]; 8]` arrays, which caps
//! a graph at 3 input channels, 8 output channels, a 5x5 kernel and a 32x32
//! image, and it generates its own random weights rather than accepting the
//! artifact's. The loop nest here has no such caps.

use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

use aether_core::ml::neural::Activation;
use aether_core::ml::tensor::Tensor;
use libm::exp;

use crate::artifact::{elem_count, Artifact, ArtifactError, Op, OpCode, MAX_TENSOR_ELEMS};

/// Everything that can stop a graph from being built or run.
#[derive(Debug, Clone, PartialEq)]
pub enum ExecError {
    /// The bytes did not parse as an artifact.
    Artifact(ArtifactError),
    /// An op consumed a value nothing had defined yet. Also what a graph whose
    /// ops are not topologically sorted fails with.
    UndefinedValue(u32),
    /// Two producers claim the same value id, or a producer claims an input.
    DuplicateDefinition(u32),
    /// An op received the wrong number of inputs or attributes.
    BadArity(OpCode),
    /// An op received an operand of the wrong rank.
    BadRank {
        /// Op that rejected the operand.
        op: OpCode,
        /// Rank the op requires.
        want: usize,
        /// Rank it was given.
        got: usize,
    },
    /// Operand shapes are individually well-formed but do not fit together.
    ShapeMismatch(OpCode),
    /// An attribute was negative, out of range, or not a valid permutation.
    BadAttribute(OpCode),
    /// A size computation overflowed `usize`.
    Overflow,
    /// A single tensor, declared or inferred, exceeds [`MAX_TENSOR_ELEMS`].
    ///
    /// The parser applies that cap to declared initializer data, where the
    /// bytes must physically be present. This variant covers the shapes the
    /// validator *infers*, which no file size bounds.
    TensorTooLarge {
        /// Elements the offending shape implies.
        elems: usize,
        /// The per-tensor ceiling.
        limit: usize,
    },
    /// The sum over every value in the graph exceeds [`MAX_GRAPH_ELEMS`].
    ///
    /// No buffer is reused, so this sum is the graph's peak residency.
    GraphTooLarge {
        /// Elements the whole graph implies.
        elems: usize,
        /// The whole-graph ceiling.
        limit: usize,
    },
    /// `run` was not given a tensor for a declared graph input.
    MissingInput(String),
    /// `run` was given a name the graph does not declare as an input.
    UnknownInput(String),
    /// A supplied input tensor does not have the declared shape.
    InputShapeMismatch {
        /// Input name.
        name: String,
        /// Shape the artifact declares.
        want: Vec<usize>,
        /// Shape actually supplied.
        got: Vec<usize>,
    },
}

impl From<ArtifactError> for ExecError {
    fn from(e: ArtifactError) -> Self {
        ExecError::Artifact(e)
    }
}

/// Peak residency a single graph may commit to, in `f64` elements.
///
/// 2^26 elements is 512 MiB at 8 bytes per element. That is deliberately
/// generous for a research kernel and deliberately finite: the point is that a
/// bound exists and a refusal is explainable, not that this particular number
/// is right for every machine. A caller that knows its own RAM should lower it.
///
/// Sized against [`MAX_TENSOR_ELEMS`] (2^24, 128 MiB): a graph may hold a few
/// tensors at the per-tensor ceiling, not sixty-five thousand of them.
pub const MAX_GRAPH_ELEMS: usize = 1 << 26;

/// Running total of every value a graph defines.
///
/// `exec` reuses no buffers — each op allocates its own output — so the sum
/// over all values *is* peak memory, not an over-estimate of it.
#[derive(Debug, Default)]
struct Budget {
    elems: usize,
}

impl Budget {
    /// Charge one shape against both ceilings.
    ///
    /// Per-tensor first, so a single absurd shape is reported as such rather
    /// than as a whole-graph overflow.
    fn charge(&mut self, shape: &[usize]) -> Result<(), ExecError> {
        let elems = elem_count(shape)?;
        if elems > MAX_TENSOR_ELEMS {
            return Err(ExecError::TensorTooLarge {
                elems,
                limit: MAX_TENSOR_ELEMS,
            });
        }
        self.elems = self.elems.checked_add(elems).ok_or(ExecError::Overflow)?;
        if self.elems > MAX_GRAPH_ELEMS {
            return Err(ExecError::GraphTooLarge {
                elems: self.elems,
                limit: MAX_GRAPH_ELEMS,
            });
        }
        Ok(())
    }
}

/// A validated graph, ready to run any number of times.
#[derive(Debug, Clone, PartialEq)]
pub struct Graph {
    artifact: Artifact,
    /// Static shape of every value id. Every entry is `Some` after validation.
    shapes: Vec<Vec<usize>>,
    /// Materialised initializers, indexed by value id.
    consts: Vec<Option<Tensor>>,
}

impl Graph {
    /// Parse `.sealg` bytes and validate the graph they describe.
    pub fn load(bytes: &[u8]) -> Result<Self, ExecError> {
        Self::from_artifact(Artifact::load(bytes)?)
    }

    /// Validate an already-parsed artifact.
    ///
    /// Checks single assignment, topological order, op arity, attribute
    /// ranges, and every shape relation. Returns on the first failure without
    /// materialising anything beyond the initializers already walked.
    pub fn from_artifact(artifact: Artifact) -> Result<Self, ExecError> {
        let n = artifact.value_names.len();
        let mut shapes: Vec<Option<Vec<usize>>> = vec![None; n];
        let mut consts: Vec<Option<Tensor>> = vec![None; n];
        // Peak residency, accumulated across every value the graph defines.
        // Checked as each shape becomes known so an over-budget graph is
        // refused during validation, before `run` allocates anything.
        let mut budget = Budget::default();

        for (id, shape) in &artifact.inputs {
            if shape.contains(&0) {
                return Err(ArtifactError::ZeroDim.into());
            }
            budget.charge(shape)?;
            define(&mut shapes, *id, shape.clone())?;
        }

        for (id, t) in &artifact.initializers {
            // Guard the assert inside `Tensor::from_vec`: an artifact built by
            // hand rather than parsed has not been through this check.
            if t.shape.contains(&0) {
                return Err(ArtifactError::ZeroDim.into());
            }
            if elem_count(&t.shape)? != t.data.len() {
                return Err(ArtifactError::ShapeElemMismatch.into());
            }
            budget.charge(&t.shape)?;
            define(&mut shapes, *id, t.shape.clone())?;
            let slot = consts
                .get_mut(*id as usize)
                .ok_or(ArtifactError::UnknownValue(*id))?;
            *slot = Some(Tensor::from_vec(t.data.clone(), t.shape.clone()));
        }

        for op in &artifact.ops {
            if op.inputs.len() != op.code.arity() {
                return Err(ExecError::BadArity(op.code));
            }
            if let Some(want) = op.code.attr_count() {
                if op.attrs.len() != want {
                    return Err(ExecError::BadArity(op.code));
                }
            } else if op.attrs.is_empty() {
                return Err(ExecError::BadArity(op.code));
            }
            let mut ins: Vec<&[usize]> = Vec::with_capacity(op.inputs.len());
            for id in &op.inputs {
                ins.push(shape_of(&shapes, *id)?);
            }
            let out = infer(op.code, &ins, &op.attrs)?;
            // The load-bearing charge. An outer product reads 2n elements and
            // writes n², so no bound on the artifact's size bounds this.
            budget.charge(&out)?;
            define(&mut shapes, op.output, out)?;
        }

        for id in &artifact.outputs {
            shape_of(&shapes, *id)?;
        }

        // Values never defined are unreachable dead names, not an error; give
        // them an empty shape so the table is total and indexing is infallible.
        let shapes = shapes.into_iter().map(Option::unwrap_or_default).collect();
        Ok(Graph {
            artifact,
            shapes,
            consts,
        })
    }

    /// Names and declared shapes of the graph's inputs, in artifact order.
    pub fn input_signature(&self) -> Vec<(&str, &[usize])> {
        self.artifact
            .inputs
            .iter()
            .map(|(id, shape)| (self.name_of(*id), shape.as_slice()))
            .collect()
    }

    /// Names of the graph's outputs, in artifact order.
    pub fn output_names(&self) -> Vec<&str> {
        self.artifact
            .outputs
            .iter()
            .map(|id| self.name_of(*id))
            .collect()
    }

    fn name_of(&self, id: u32) -> &str {
        self.artifact
            .value_names
            .get(id as usize)
            .map_or("", String::as_str)
    }

    /// Evaluate the graph.
    ///
    /// Every declared input must appear in `inputs` with exactly its declared
    /// shape; extra names are refused rather than ignored.
    pub fn run(&self, inputs: &[(&str, Tensor)]) -> Result<Vec<(String, Tensor)>, ExecError> {
        let mut vals = self.consts.clone();

        for (id, want) in &self.artifact.inputs {
            let name = self.name_of(*id);
            let supplied = inputs
                .iter()
                .find(|(n, _)| *n == name)
                .ok_or_else(|| ExecError::MissingInput(name.to_string()))?;
            if supplied.1.shape != *want {
                return Err(ExecError::InputShapeMismatch {
                    name: name.to_string(),
                    want: want.clone(),
                    got: supplied.1.shape.clone(),
                });
            }
            let slot = vals
                .get_mut(*id as usize)
                .ok_or(ArtifactError::UnknownValue(*id))?;
            *slot = Some(supplied.1.clone());
        }

        for (name, _) in inputs {
            if !self
                .artifact
                .inputs
                .iter()
                .any(|(id, _)| self.name_of(*id) == *name)
            {
                return Err(ExecError::UnknownInput((*name).to_string()));
            }
        }

        for op in &self.artifact.ops {
            let mut ins: Vec<Tensor> = Vec::with_capacity(op.inputs.len());
            for id in &op.inputs {
                let t = vals
                    .get(*id as usize)
                    .and_then(Option::as_ref)
                    .ok_or(ExecError::UndefinedValue(*id))?;
                ins.push(t.clone());
            }
            let out_shape = self
                .shapes
                .get(op.output as usize)
                .ok_or(ArtifactError::UnknownValue(op.output))?;
            let produced = eval(op, &ins, out_shape)?;
            let slot = vals
                .get_mut(op.output as usize)
                .ok_or(ArtifactError::UnknownValue(op.output))?;
            *slot = Some(produced);
        }

        let mut out = Vec::with_capacity(self.artifact.outputs.len());
        for id in &self.artifact.outputs {
            let t = vals
                .get(*id as usize)
                .and_then(Option::as_ref)
                .ok_or(ExecError::UndefinedValue(*id))?;
            out.push((self.name_of(*id).to_string(), t.clone()));
        }
        Ok(out)
    }
}

// ── Validation helpers ──────────────────────────────────────────────────────

fn define(shapes: &mut [Option<Vec<usize>>], id: u32, shape: Vec<usize>) -> Result<(), ExecError> {
    let slot = shapes
        .get_mut(id as usize)
        .ok_or(ArtifactError::UnknownValue(id))?;
    if slot.is_some() {
        return Err(ExecError::DuplicateDefinition(id));
    }
    *slot = Some(shape);
    Ok(())
}

fn shape_of(shapes: &[Option<Vec<usize>>], id: u32) -> Result<&[usize], ExecError> {
    shapes
        .get(id as usize)
        .ok_or(ArtifactError::UnknownValue(id))?
        .as_deref()
        .ok_or(ExecError::UndefinedValue(id))
}

fn want_rank(op: OpCode, s: &[usize], want: usize) -> Result<(), ExecError> {
    if s.len() == want {
        Ok(())
    } else {
        Err(ExecError::BadRank {
            op,
            want,
            got: s.len(),
        })
    }
}

/// Read a non-negative attribute as a `usize`.
fn attr_usize(op: OpCode, attrs: &[i64], i: usize) -> Result<usize, ExecError> {
    match attrs.get(i) {
        Some(&v) if v >= 0 => usize::try_from(v).map_err(|_| ExecError::Overflow),
        _ => Err(ExecError::BadAttribute(op)),
    }
}

fn attr_flag(op: OpCode, attrs: &[i64], i: usize) -> Result<bool, ExecError> {
    match attrs.get(i) {
        Some(0) => Ok(false),
        Some(1) => Ok(true),
        _ => Err(ExecError::BadAttribute(op)),
    }
}

/// Row-broadcast rule: equal shapes, or a rank-1 rhs matching the trailing dim.
fn elementwise_shape(op: OpCode, a: &[usize], b: &[usize]) -> Result<Vec<usize>, ExecError> {
    if a == b {
        return Ok(a.to_vec());
    }
    if b.len() == 1 && a.last() == b.first() {
        return Ok(a.to_vec());
    }
    Err(ExecError::ShapeMismatch(op))
}

fn matmul_shape(op: OpCode, a: &[usize], b: &[usize]) -> Result<Vec<usize>, ExecError> {
    want_rank(op, a, 2)?;
    want_rank(op, b, 2)?;
    if a[1] != b[0] {
        return Err(ExecError::ShapeMismatch(op));
    }
    Ok(vec![a[0], b[1]])
}

fn transposed(s: &[usize], flip: bool) -> Vec<usize> {
    if flip && s.len() == 2 {
        vec![s[1], s[0]]
    } else {
        s.to_vec()
    }
}

/// Output spatial extent of a sliding window. Refuses a window that does not
/// fit and a zero stride.
fn window_out(
    op: OpCode,
    input: usize,
    pad: usize,
    k: usize,
    stride: usize,
) -> Result<usize, ExecError> {
    if stride == 0 || k == 0 {
        return Err(ExecError::BadAttribute(op));
    }
    let padded = input
        .checked_add(pad.checked_mul(2).ok_or(ExecError::Overflow)?)
        .ok_or(ExecError::Overflow)?;
    if padded < k {
        return Err(ExecError::ShapeMismatch(op));
    }
    Ok((padded - k) / stride + 1)
}

/// Static shape inference for one op. The only place shape rules are written.
fn infer(code: OpCode, ins: &[&[usize]], attrs: &[i64]) -> Result<Vec<usize>, ExecError> {
    let a = ins.first().copied().ok_or(ExecError::BadArity(code))?;
    match code {
        OpCode::MatMul => {
            let b = ins.get(1).copied().ok_or(ExecError::BadArity(code))?;
            matmul_shape(code, a, b)
        }
        OpCode::Gemm => {
            let b = ins.get(1).copied().ok_or(ExecError::BadArity(code))?;
            let c = ins.get(2).copied().ok_or(ExecError::BadArity(code))?;
            want_rank(code, a, 2)?;
            want_rank(code, b, 2)?;
            let ta = attr_flag(code, attrs, 0)?;
            let tb = attr_flag(code, attrs, 1)?;
            let y = matmul_shape(code, &transposed(a, ta), &transposed(b, tb))?;
            elementwise_shape(code, &y, c)
        }
        OpCode::Add | OpCode::Mul => {
            let b = ins.get(1).copied().ok_or(ExecError::BadArity(code))?;
            elementwise_shape(code, a, b)
        }
        OpCode::Relu | OpCode::Sigmoid | OpCode::Tanh => Ok(a.to_vec()),
        OpCode::Softmax => {
            if a.is_empty() {
                return Err(ExecError::BadRank {
                    op: code,
                    want: 1,
                    got: 0,
                });
            }
            Ok(a.to_vec())
        }
        OpCode::Reshape => {
            let total = elem_count(a)?;
            let mut dims = Vec::with_capacity(attrs.len());
            let mut wild = None;
            let mut known: usize = 1;
            for (i, &v) in attrs.iter().enumerate() {
                if v == -1 {
                    if wild.is_some() {
                        return Err(ExecError::BadAttribute(code));
                    }
                    wild = Some(i);
                    dims.push(1);
                } else if v > 0 {
                    let d = usize::try_from(v).map_err(|_| ExecError::Overflow)?;
                    known = known.checked_mul(d).ok_or(ExecError::Overflow)?;
                    dims.push(d);
                } else {
                    return Err(ExecError::BadAttribute(code));
                }
            }
            match wild {
                Some(i) => {
                    if known == 0 || total % known != 0 {
                        return Err(ExecError::ShapeMismatch(code));
                    }
                    let d = total / known;
                    if d == 0 {
                        return Err(ExecError::ShapeMismatch(code));
                    }
                    dims[i] = d;
                }
                None => {
                    if known != total {
                        return Err(ExecError::ShapeMismatch(code));
                    }
                }
            }
            Ok(dims)
        }
        OpCode::Transpose => {
            if attrs.len() != a.len() {
                return Err(ExecError::BadAttribute(code));
            }
            let mut seen = vec![false; a.len()];
            let mut out = Vec::with_capacity(a.len());
            for i in 0..attrs.len() {
                let p = attr_usize(code, attrs, i)?;
                match seen.get_mut(p) {
                    Some(slot) if !*slot => *slot = true,
                    _ => return Err(ExecError::BadAttribute(code)),
                }
                out.push(a[p]);
            }
            Ok(out)
        }
        OpCode::Conv2d => {
            let w = ins.get(1).copied().ok_or(ExecError::BadArity(code))?;
            let b = ins.get(2).copied().ok_or(ExecError::BadArity(code))?;
            want_rank(code, a, 4)?;
            want_rank(code, w, 4)?;
            want_rank(code, b, 1)?;
            if a[1] != w[1] || b[0] != w[0] {
                return Err(ExecError::ShapeMismatch(code));
            }
            let sh = attr_usize(code, attrs, 0)?;
            let sw = attr_usize(code, attrs, 1)?;
            let ph = attr_usize(code, attrs, 2)?;
            let pw = attr_usize(code, attrs, 3)?;
            let oh = window_out(code, a[2], ph, w[2], sh)?;
            let ow = window_out(code, a[3], pw, w[3], sw)?;
            Ok(vec![a[0], w[0], oh, ow])
        }
        OpCode::MaxPool2d => {
            want_rank(code, a, 4)?;
            let kh = attr_usize(code, attrs, 0)?;
            let kw = attr_usize(code, attrs, 1)?;
            let sh = attr_usize(code, attrs, 2)?;
            let sw = attr_usize(code, attrs, 3)?;
            let oh = window_out(code, a[2], 0, kh, sh)?;
            let ow = window_out(code, a[3], 0, kw, sw)?;
            Ok(vec![a[0], a[1], oh, ow])
        }
    }
}

// ── Kernels ─────────────────────────────────────────────────────────────────

/// Copy a tensor's elements out from under its lock.
///
/// ponytail: one copy per operand. It keeps every kernel below a plain slice
/// loop and sidesteps deadlocking on a graph that feeds the same value into two
/// operands of one op (`spin::Mutex` is not reentrant). Operate on the guards
/// directly if a profile ever shows the copies mattering.
fn flat(t: &Tensor) -> Vec<f64> {
    t.data.lock().clone()
}

fn row_strides(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * shape[i + 1];
    }
    s
}

fn permute(x: &Tensor, perm: &[usize], out_shape: &[usize]) -> Tensor {
    let xd = flat(x);
    let in_strides = row_strides(&x.shape);
    let rank = out_shape.len();
    let mut out = vec![0.0f64; xd.len()];
    for (o, slot) in out.iter_mut().enumerate() {
        let mut rem = o;
        let mut src = 0usize;
        for d in (0..rank).rev() {
            let dim = out_shape.get(d).copied().unwrap_or(1).max(1);
            let i = rem % dim;
            rem /= dim;
            let axis = perm.get(d).copied().unwrap_or(0);
            src += i * in_strides.get(axis).copied().unwrap_or(0);
        }
        *slot = xd.get(src).copied().unwrap_or(0.0);
    }
    Tensor::from_vec(out, out_shape.to_vec())
}

fn broadcast_row(x: &Tensor, b: &Tensor, f: fn(f64, f64) -> f64) -> Tensor {
    let xd = flat(x);
    let bd = flat(b);
    let w = bd.len().max(1);
    let out: Vec<f64> = xd
        .iter()
        .enumerate()
        .map(|(i, &v)| f(v, bd.get(i % w).copied().unwrap_or(0.0)))
        .collect();
    Tensor::from_vec(out, x.shape.clone())
}

fn softmax_last(x: &Tensor) -> Tensor {
    let mut d = flat(x);
    let last = x.shape.last().copied().unwrap_or(0);
    if last == 0 {
        return Tensor::from_vec(d, x.shape.clone());
    }
    for row in d.chunks_mut(last) {
        let mut max = f64::NEG_INFINITY;
        for &v in row.iter() {
            if v > max {
                max = v;
            }
        }
        let mut sum = 0.0;
        for v in row.iter_mut() {
            *v = exp(*v - max);
            sum += *v;
        }
        if sum > 0.0 {
            for v in row.iter_mut() {
                *v /= sum;
            }
        }
    }
    Tensor::from_vec(d, x.shape.clone())
}

/// Read a rank-4 NCHW shape without indexing.
fn nchw(op: OpCode, shape: &[usize]) -> Result<(usize, usize, usize, usize), ExecError> {
    match shape {
        [n, c, h, w] => Ok((*n, *c, *h, *w)),
        _ => Err(ExecError::BadRank {
            op,
            want: 4,
            got: shape.len(),
        }),
    }
}

/// Read the four non-negative attributes every spatial op carries.
fn four_attrs(op: OpCode, attrs: &[i64]) -> Result<(usize, usize, usize, usize), ExecError> {
    Ok((
        attr_usize(op, attrs, 0)?,
        attr_usize(op, attrs, 1)?,
        attr_usize(op, attrs, 2)?,
        attr_usize(op, attrs, 3)?,
    ))
}

/// NCHW convolution, groups 1, dilation 1. Shapes are validated by [`infer`].
fn conv2d(
    x: &Tensor,
    w: &Tensor,
    b: &Tensor,
    attrs: &[i64],
    out_shape: &[usize],
) -> Result<Tensor, ExecError> {
    let (xd, wd, bd) = (flat(x), flat(w), flat(b));
    let (batch, chan, h, width) = nchw(OpCode::Conv2d, &x.shape)?;
    let (filters, _, kh, kw) = nchw(OpCode::Conv2d, &w.shape)?;
    let (sh, sw, ph, pw) = four_attrs(OpCode::Conv2d, attrs)?;
    let (_, _, oh, ow) = nchw(OpCode::Conv2d, out_shape)?;

    let mut out = vec![0.0f64; batch * filters * oh * ow];
    for n in 0..batch {
        for m in 0..filters {
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut acc = bd.get(m).copied().unwrap_or(0.0);
                    for c in 0..chan {
                        for ky in 0..kh {
                            let py = oy * sh + ky;
                            if py < ph {
                                continue;
                            }
                            let iy = py - ph;
                            if iy >= h {
                                continue;
                            }
                            for kx in 0..kw {
                                let px = ox * sw + kx;
                                if px < pw {
                                    continue;
                                }
                                let ix = px - pw;
                                if ix >= width {
                                    continue;
                                }
                                let xi = ((n * chan + c) * h + iy) * width + ix;
                                let wi = ((m * chan + c) * kh + ky) * kw + kx;
                                acc += xd.get(xi).copied().unwrap_or(0.0)
                                    * wd.get(wi).copied().unwrap_or(0.0);
                            }
                        }
                    }
                    if let Some(slot) = out.get_mut(((n * filters + m) * oh + oy) * ow + ox) {
                        *slot = acc;
                    }
                }
            }
        }
    }
    Ok(Tensor::from_vec(out, out_shape.to_vec()))
}

/// NCHW max pool, no padding. Shapes are validated by [`infer`].
fn maxpool2d(x: &Tensor, attrs: &[i64], out_shape: &[usize]) -> Result<Tensor, ExecError> {
    let xd = flat(x);
    let (batch, chan, h, width) = nchw(OpCode::MaxPool2d, &x.shape)?;
    let (kh, kw, sh, sw) = four_attrs(OpCode::MaxPool2d, attrs)?;
    let (_, _, oh, ow) = nchw(OpCode::MaxPool2d, out_shape)?;

    let mut out = vec![0.0f64; batch * chan * oh * ow];
    for n in 0..batch {
        for c in 0..chan {
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut best = f64::NEG_INFINITY;
                    for ky in 0..kh {
                        let iy = oy * sh + ky;
                        if iy >= h {
                            continue;
                        }
                        for kx in 0..kw {
                            let ix = ox * sw + kx;
                            if ix >= width {
                                continue;
                            }
                            let v = xd
                                .get(((n * chan + c) * h + iy) * width + ix)
                                .copied()
                                .unwrap_or(f64::NEG_INFINITY);
                            if v > best {
                                best = v;
                            }
                        }
                    }
                    if let Some(slot) = out.get_mut(((n * chan + c) * oh + oy) * ow + ox) {
                        *slot = best;
                    }
                }
            }
        }
    }
    Ok(Tensor::from_vec(out, out_shape.to_vec()))
}

fn act(x: &Tensor, a: Activation) -> Tensor {
    x.map(|v| a.apply_scalar(v))
}

/// Run one op. Reached only after [`infer`] accepted the same shapes, so every
/// `aether-core` call below has its preconditions met.
fn eval(op: &Op, ins: &[Tensor], out_shape: &[usize]) -> Result<Tensor, ExecError> {
    let a = ins.first().ok_or(ExecError::BadArity(op.code))?;
    Ok(match op.code {
        OpCode::MatMul => {
            let b = ins.get(1).ok_or(ExecError::BadArity(op.code))?;
            a.matmul(b)
        }
        OpCode::Gemm => {
            let b = ins.get(1).ok_or(ExecError::BadArity(op.code))?;
            let c = ins.get(2).ok_or(ExecError::BadArity(op.code))?;
            let at = if attr_flag(op.code, &op.attrs, 0)? {
                a.transpose()
            } else {
                a.clone()
            };
            let bt = if attr_flag(op.code, &op.attrs, 1)? {
                b.transpose()
            } else {
                b.clone()
            };
            let y = at.matmul(&bt);
            if y.shape == c.shape {
                y.add(c)
            } else {
                broadcast_row(&y, c, |p, q| p + q)
            }
        }
        OpCode::Add => {
            let b = ins.get(1).ok_or(ExecError::BadArity(op.code))?;
            if a.shape == b.shape {
                a.add(b)
            } else {
                broadcast_row(a, b, |p, q| p + q)
            }
        }
        OpCode::Mul => {
            let b = ins.get(1).ok_or(ExecError::BadArity(op.code))?;
            if a.shape == b.shape {
                a.mul(b)
            } else {
                broadcast_row(a, b, |p, q| p * q)
            }
        }
        OpCode::Relu => act(a, Activation::ReLU),
        OpCode::Sigmoid => act(a, Activation::Sigmoid),
        OpCode::Tanh => act(a, Activation::Tanh),
        OpCode::Softmax => softmax_last(a),
        OpCode::Reshape => Tensor::from_vec(flat(a), out_shape.to_vec()),
        OpCode::Transpose => {
            let mut perm = Vec::with_capacity(op.attrs.len());
            for i in 0..op.attrs.len() {
                perm.push(attr_usize(op.code, &op.attrs, i)?);
            }
            permute(a, &perm, out_shape)
        }
        OpCode::Conv2d => {
            let w = ins.get(1).ok_or(ExecError::BadArity(op.code))?;
            let b = ins.get(2).ok_or(ExecError::BadArity(op.code))?;
            conv2d(a, w, b, &op.attrs, out_shape)?
        }
        OpCode::MaxPool2d => maxpool2d(a, &op.attrs, out_shape)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifact::TensorDef;

    fn names(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| (*s).to_string()).collect()
    }

    /// x[1,2] -> Gemm(x, W1[2,2], b1[2], trans_b) -> Relu
    ///        -> Gemm(h, W2[1,2], b2[1], trans_b) = y[1,1]
    ///
    /// Weights are stored in PyTorch `nn.Linear` layout `[out, in]`, which is
    /// what `trans_b = 1` exists for.
    fn mlp() -> Artifact {
        Artifact {
            value_names: names(&["x", "w1", "b1", "pre", "h", "w2", "b2", "y"]),
            inputs: vec![(0, vec![1, 2])],
            outputs: vec![7],
            initializers: vec![
                (
                    1,
                    TensorDef::new(vec![2, 2], vec![1.0, -2.0, 0.5, 1.0]).expect("w1"),
                ),
                (2, TensorDef::new(vec![2], vec![0.5, -0.5]).expect("b1")),
                (5, TensorDef::new(vec![1, 2], vec![3.0, -1.0]).expect("w2")),
                (6, TensorDef::new(vec![1], vec![0.25]).expect("b2")),
            ],
            ops: vec![
                Op {
                    code: OpCode::Gemm,
                    inputs: vec![0, 1, 2],
                    output: 3,
                    attrs: vec![0, 1],
                },
                Op {
                    code: OpCode::Relu,
                    inputs: vec![3],
                    output: 4,
                    attrs: vec![],
                },
                Op {
                    code: OpCode::Gemm,
                    inputs: vec![4, 5, 6],
                    output: 7,
                    attrs: vec![0, 1],
                },
            ],
        }
    }

    fn run_mlp() -> Vec<(String, Tensor)> {
        let g = Graph::load(&mlp().encode()).expect("graph");
        g.run(&[("x", Tensor::from_vec(vec![1.0, 2.0], vec![1, 2]))])
            .expect("run")
    }

    #[test]
    fn mlp_matches_hand_computation() {
        // pre = x @ W1^T + b1
        //     = [1*1 + 2*(-2), 1*0.5 + 2*1] + [0.5, -0.5]
        //     = [-3, 2.5] + [0.5, -0.5] = [-2.5, 2.0]
        // h   = relu(pre) = [0, 2.0]
        // y   = h @ W2^T + b2 = [0*3 + 2*(-1)] + [0.25] = -1.75
        // Every intermediate is exact in binary floating point, so this is an
        // equality check, not a tolerance check.
        let out = run_mlp();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, "y");
        assert_eq!(out[0].1.shape, vec![1, 1]);
        assert_eq!(out[0].1.get(&[0, 0]), -1.75);
    }

    #[test]
    fn run_is_bit_deterministic() {
        let a = run_mlp();
        let b = run_mlp();
        assert_eq!(a[0].1.get(&[0, 0]).to_bits(), b[0].1.get(&[0, 0]).to_bits());
    }

    #[test]
    fn graph_is_reusable_across_runs() {
        let g = Graph::load(&mlp().encode()).expect("graph");
        let first = g
            .run(&[("x", Tensor::from_vec(vec![1.0, 2.0], vec![1, 2]))])
            .expect("run");
        let second = g
            .run(&[("x", Tensor::from_vec(vec![1.0, 2.0], vec![1, 2]))])
            .expect("run");
        assert_eq!(first[0].1.get(&[0, 0]), second[0].1.get(&[0, 0]));
    }

    #[test]
    fn signature_reports_declared_names_and_shapes() {
        let g = Graph::load(&mlp().encode()).expect("graph");
        assert_eq!(g.input_signature(), vec![("x", &[1usize, 2][..])]);
        assert_eq!(g.output_names(), vec!["y"]);
    }

    /// 3x3 image, one 2x2 filter of `[[1,0],[0,1]]`, stride 1, no padding,
    /// then a 2x2 max pool.
    fn cnn() -> Artifact {
        Artifact {
            value_names: names(&["img", "k", "kb", "feat", "out"]),
            inputs: vec![(0, vec![1, 1, 3, 3])],
            outputs: vec![4],
            initializers: vec![
                (
                    1,
                    TensorDef::new(vec![1, 1, 2, 2], vec![1.0, 0.0, 0.0, 1.0]).expect("k"),
                ),
                (2, TensorDef::new(vec![1], vec![0.0]).expect("kb")),
            ],
            ops: vec![
                Op {
                    code: OpCode::Conv2d,
                    inputs: vec![0, 1, 2],
                    output: 3,
                    attrs: vec![1, 1, 0, 0],
                },
                Op {
                    code: OpCode::MaxPool2d,
                    inputs: vec![3],
                    output: 4,
                    attrs: vec![2, 2, 2, 2],
                },
            ],
        }
    }

    #[test]
    fn cnn_matches_hand_computation() {
        // Convolving [[1,2,3],[4,5,6],[7,8,9]] with the 2x2 diagonal filter
        // sums each window's main diagonal:
        //   [[1+5, 2+6], [4+8, 5+9]] = [[6, 8], [12, 14]]
        // A 2x2 max pool over that yields 14.
        let g = Graph::load(&cnn().encode()).expect("graph");
        let img = Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            vec![1, 1, 3, 3],
        );
        let out = g.run(&[("img", img)]).expect("run");
        assert_eq!(out[0].1.shape, vec![1, 1, 1, 1]);
        assert_eq!(out[0].1.get(&[0, 0, 0, 0]), 14.0);
    }

    #[test]
    fn conv_padding_widens_the_output() {
        let mut a = cnn();
        a.ops[0].attrs = vec![1, 1, 1, 1]; // pad 1 -> 4x4 feature map
        a.ops[1].attrs = vec![4, 4, 4, 4]; // pool the whole thing
        let g = Graph::load(&a.encode()).expect("graph");
        let img = Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            vec![1, 1, 3, 3],
        );
        let out = g.run(&[("img", img)]).expect("run");
        // Padded windows still peak on the [5,9] diagonal window: 14.
        assert_eq!(out[0].1.get(&[0, 0, 0, 0]), 14.0);
    }

    /// Single-op graph over one input, for exercising individual ops.
    fn unary(code: OpCode, in_shape: Vec<usize>, attrs: Vec<i64>) -> Artifact {
        Artifact {
            value_names: names(&["a", "b"]),
            inputs: vec![(0, in_shape)],
            outputs: vec![1],
            initializers: vec![],
            ops: vec![Op {
                code,
                inputs: vec![0],
                output: 1,
                attrs,
            }],
        }
    }

    #[test]
    fn softmax_normalises_the_trailing_axis_per_row() {
        let g = Graph::load(&unary(OpCode::Softmax, vec![2, 2], vec![]).encode()).expect("graph");
        let x = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0], vec![2, 2]);
        let out = g.run(&[("a", x)]).expect("run");
        for r in 0..2 {
            let s = out[0].1.get(&[r, 0]) + out[0].1.get(&[r, 1]);
            assert!((s - 1.0).abs() < 1e-12, "row {r} sums to {s}");
            assert_eq!(out[0].1.get(&[r, 0]), 0.5);
        }
    }

    #[test]
    fn transpose_permutes_rank_three() {
        let a = unary(OpCode::Transpose, vec![2, 1, 3], vec![2, 0, 1]);
        let g = Graph::load(&a.encode()).expect("graph");
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 1, 3]);
        let out = g.run(&[("a", x)]).expect("run");
        assert_eq!(out[0].1.shape, vec![3, 2, 1]);
        // element (i, j, 0) of the result is element (j, 0, i) of the input
        assert_eq!(out[0].1.get(&[0, 0, 0]), 1.0);
        assert_eq!(out[0].1.get(&[0, 1, 0]), 4.0);
        assert_eq!(out[0].1.get(&[2, 1, 0]), 6.0);
    }

    #[test]
    fn reshape_resolves_the_inferred_dimension() {
        let g =
            Graph::load(&unary(OpCode::Reshape, vec![2, 3], vec![-1, 2]).encode()).expect("graph");
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let out = g.run(&[("a", x)]).expect("run");
        assert_eq!(out[0].1.shape, vec![3, 2]);
        assert_eq!(out[0].1.get(&[2, 1]), 6.0);
    }

    #[test]
    fn sigmoid_and_tanh_hit_their_fixed_points() {
        for (code, want) in [(OpCode::Sigmoid, 0.5), (OpCode::Tanh, 0.0)] {
            let g = Graph::load(&unary(code, vec![1], vec![]).encode()).expect("graph");
            let out = g
                .run(&[("a", Tensor::from_vec(vec![0.0], vec![1]))])
                .expect("run");
            assert_eq!(out[0].1.get(&[0]), want, "{code:?}");
        }
    }

    // ── refusal cases ───────────────────────────────────────────────────────

    #[test]
    fn refuses_matmul_with_mismatched_inner_dimension() {
        let a = Artifact {
            value_names: names(&["x", "w", "y"]),
            inputs: vec![(0, vec![2, 3])],
            outputs: vec![2],
            initializers: vec![(1, TensorDef::new(vec![4, 5], vec![0.0; 20]).expect("w"))],
            ops: vec![Op {
                code: OpCode::MatMul,
                inputs: vec![0, 1],
                output: 2,
                attrs: vec![],
            }],
        };
        assert_eq!(
            Graph::load(&a.encode()),
            Err(ExecError::ShapeMismatch(OpCode::MatMul))
        );
    }

    #[test]
    fn refuses_add_with_unbroadcastable_shapes() {
        let a = Artifact {
            value_names: names(&["x", "b", "y"]),
            inputs: vec![(0, vec![2, 3])],
            outputs: vec![2],
            initializers: vec![(1, TensorDef::new(vec![5], vec![0.0; 5]).expect("b"))],
            ops: vec![Op {
                code: OpCode::Add,
                inputs: vec![0, 1],
                output: 2,
                attrs: vec![],
            }],
        };
        assert_eq!(
            Graph::load(&a.encode()),
            Err(ExecError::ShapeMismatch(OpCode::Add))
        );
    }

    #[test]
    fn refuses_ops_that_are_not_topologically_sorted() {
        let mut a = mlp();
        a.ops.swap(0, 1); // Relu now reads value 3 before Gemm defines it
        assert_eq!(Graph::load(&a.encode()), Err(ExecError::UndefinedValue(3)));
    }

    #[test]
    fn refuses_two_producers_for_one_value() {
        let mut a = mlp();
        a.ops[2].output = 4; // already defined by the Relu
        assert_eq!(
            Graph::load(&a.encode()),
            Err(ExecError::DuplicateDefinition(4))
        );
    }

    #[test]
    fn refuses_an_op_that_overwrites_a_graph_input() {
        let mut a = mlp();
        a.ops[1].output = 0;
        assert_eq!(
            Graph::load(&a.encode()),
            Err(ExecError::DuplicateDefinition(0))
        );
    }

    #[test]
    fn refuses_an_output_nothing_produces() {
        let mut a = mlp();
        a.value_names.push("dangling".to_string());
        a.outputs = vec![8];
        assert_eq!(Graph::load(&a.encode()), Err(ExecError::UndefinedValue(8)));
    }

    #[test]
    fn refuses_a_kernel_larger_than_its_padded_input() {
        let mut a = cnn();
        a.initializers[0].1 = TensorDef::new(vec![1, 1, 5, 5], vec![0.0; 25]).expect("k");
        assert_eq!(
            Graph::load(&a.encode()),
            Err(ExecError::ShapeMismatch(OpCode::Conv2d))
        );
    }

    #[test]
    fn refuses_zero_stride() {
        let mut a = cnn();
        a.ops[0].attrs = vec![0, 1, 0, 0];
        assert_eq!(
            Graph::load(&a.encode()),
            Err(ExecError::BadAttribute(OpCode::Conv2d))
        );
    }

    #[test]
    fn refuses_conv_weights_whose_channels_disagree() {
        let mut a = cnn();
        a.initializers[0].1 = TensorDef::new(vec![1, 2, 2, 2], vec![0.0; 8]).expect("k");
        assert_eq!(
            Graph::load(&a.encode()),
            Err(ExecError::ShapeMismatch(OpCode::Conv2d))
        );
    }

    #[test]
    fn refuses_conv_on_the_wrong_rank() {
        let mut a = cnn();
        a.inputs[0].1 = vec![3, 3];
        assert_eq!(
            Graph::load(&a.encode()),
            Err(ExecError::BadRank {
                op: OpCode::Conv2d,
                want: 4,
                got: 2
            })
        );
    }

    #[test]
    fn refuses_reshape_that_changes_the_element_count() {
        let a = unary(OpCode::Reshape, vec![2, 3], vec![4, 2]);
        assert_eq!(
            Graph::load(&a.encode()),
            Err(ExecError::ShapeMismatch(OpCode::Reshape))
        );
    }

    #[test]
    fn refuses_reshape_with_two_inferred_dimensions() {
        let a = unary(OpCode::Reshape, vec![2, 3], vec![-1, -1]);
        assert_eq!(
            Graph::load(&a.encode()),
            Err(ExecError::BadAttribute(OpCode::Reshape))
        );
    }

    #[test]
    fn refuses_a_permutation_with_a_repeated_axis() {
        let a = unary(OpCode::Transpose, vec![2, 3], vec![0, 0]);
        assert_eq!(
            Graph::load(&a.encode()),
            Err(ExecError::BadAttribute(OpCode::Transpose))
        );
    }

    #[test]
    fn refuses_a_permutation_of_the_wrong_length() {
        let a = unary(OpCode::Transpose, vec![2, 3], vec![0]);
        assert_eq!(
            Graph::load(&a.encode()),
            Err(ExecError::BadAttribute(OpCode::Transpose))
        );
    }

    #[test]
    fn refuses_a_gemm_transpose_flag_that_is_not_zero_or_one() {
        let mut a = mlp();
        a.ops[0].attrs = vec![0, 7];
        assert_eq!(
            Graph::load(&a.encode()),
            Err(ExecError::BadAttribute(OpCode::Gemm))
        );
    }

    #[test]
    fn refuses_a_missing_input_at_run_time() {
        let g = Graph::load(&mlp().encode()).expect("graph");
        assert_eq!(g.run(&[]), Err(ExecError::MissingInput("x".to_string())));
    }

    #[test]
    fn refuses_an_input_of_the_wrong_shape() {
        let g = Graph::load(&mlp().encode()).expect("graph");
        let bad = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        assert_eq!(
            g.run(&[("x", bad)]),
            Err(ExecError::InputShapeMismatch {
                name: "x".to_string(),
                want: vec![1, 2],
                got: vec![1, 3],
            })
        );
    }

    #[test]
    fn refuses_an_input_the_graph_does_not_declare() {
        let g = Graph::load(&mlp().encode()).expect("graph");
        let x = Tensor::from_vec(vec![1.0, 2.0], vec![1, 2]);
        let extra = Tensor::from_vec(vec![0.0], vec![1]);
        assert_eq!(
            g.run(&[("x", x), ("stowaway", extra)]),
            Err(ExecError::UnknownInput("stowaway".to_string()))
        );
    }

    #[test]
    fn refuses_a_hand_built_initializer_whose_data_does_not_match_its_shape() {
        // Bypasses the parser: `Artifact` fields are public, so `from_artifact`
        // has to repeat the check rather than trust it.
        let a = Artifact {
            value_names: names(&["w", "y"]),
            inputs: vec![],
            outputs: vec![0],
            initializers: vec![(
                0,
                TensorDef {
                    shape: vec![2, 2],
                    data: vec![1.0],
                },
            )],
            ops: vec![],
        };
        assert_eq!(
            Graph::from_artifact(a),
            Err(ExecError::Artifact(ArtifactError::ShapeElemMismatch))
        );
    }

    #[test]
    fn refuses_bad_bytes_before_touching_the_graph() {
        assert_eq!(
            Graph::load(b"too short"),
            Err(ExecError::Artifact(ArtifactError::TooShort))
        );
        assert_eq!(
            Graph::load(&[0x5Au8; 64]),
            Err(ExecError::Artifact(ArtifactError::BadMagic))
        );
    }

    #[test]
    fn artifact_survives_the_wire_and_still_runs() {
        // The three layers in sequence: encode, ship, load, run, ship back.
        use crate::wire::{read_message, write_message, MemTransport, Message, NamedTensor};

        let mut link = MemTransport::new();
        write_message(
            &mut link,
            &Message::PushArtifact {
                bytes: mlp().encode(),
            },
        )
        .expect("push");

        let g = match read_message(&mut link).expect("pull") {
            Message::PushArtifact { bytes } => Graph::load(&bytes).expect("graph"),
            other => panic!("wrong message: {other:?}"),
        };
        let out = g
            .run(&[("x", Tensor::from_vec(vec![1.0, 2.0], vec![1, 2]))])
            .expect("run");

        let reply = Message::Result {
            code: 0,
            values: out
                .iter()
                .map(|(name, t)| NamedTensor {
                    name: name.clone(),
                    value: TensorDef::new(t.shape.clone(), flat(t)).expect("tensor"),
                })
                .collect(),
        };
        write_message(&mut link, &reply).expect("reply");
        assert_eq!(read_message(&mut link).expect("read reply"), reply);
        match reply {
            Message::Result { values, .. } => assert_eq!(values[0].value.data, vec![-1.75]),
            other => panic!("wrong message: {other:?}"),
        }
    }

    #[test]
    fn feeding_one_value_to_both_operands_does_not_deadlock() {
        // Two operands backed by the same Arc: the kernels copy before locking,
        // so a non-reentrant spin lock is never taken twice.
        let a = Artifact {
            value_names: names(&["x", "y"]),
            inputs: vec![(0, vec![2, 2])],
            outputs: vec![1],
            initializers: vec![],
            ops: vec![Op {
                code: OpCode::Mul,
                inputs: vec![0, 0],
                output: 1,
                attrs: vec![],
            }],
        };
        let g = Graph::load(&a.encode()).expect("graph");
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let out = g.run(&[("x", x)]).expect("run");
        assert_eq!(out[0].1.get(&[1, 1]), 16.0);
    }
}
