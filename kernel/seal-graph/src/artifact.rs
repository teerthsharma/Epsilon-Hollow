// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! The `.sealg` artifact format.
//!
//! A `.sealg` file carries one topologically-sorted graph: a value-name table,
//! the declared input bindings with their fixed shapes, the output bindings,
//! the initializer tensors (weights), and the op list. It is emitted host-side
//! and consumed by [`crate::exec::Graph`].
//!
//! # Byte layout
//!
//! Everything is little-endian, matching the `push_u32` / `read_u32` idiom in
//! `kernel/seal-os/src/ml_engine.rs`.
//!
//! ```text
//! header (48 bytes)
//!   [0..8]    magic         b"SEALG001"
//!   [8..12]   u32           format version (currently 1)
//!   [12..16]  u32           payload length in bytes
//!   [16..48]  [u8; 32]      SHA-256 of the payload bytes, exactly
//!
//! payload
//!   u32 n_values
//!   n_values x { u32 name_len; name_len bytes UTF-8 }        -- value id = index
//!   u32 n_inputs
//!   n_inputs x { u32 value_id; u32 rank; rank x u32 dim }
//!   u32 n_outputs
//!   n_outputs x { u32 value_id }
//!   u32 n_initializers
//!   n_initializers x { u32 value_id; u32 rank; rank x u32 dim;
//!                      u32 n_elems; n_elems x f64 }
//!   u32 n_ops
//!   n_ops x { u8 opcode; u32 n_in; n_in x u32 value_id;
//!             u32 out_value_id; u32 n_attrs; n_attrs x i64 }
//! ```
//!
//! The digest covers the payload and nothing else, so a Python exporter writes
//! `hashlib.sha256(payload).digest()` with no framing subtleties. [`Artifact::load`]
//! refuses on mismatch before parsing a single payload field.
//!
//! # What the parser guarantees
//!
//! * No panic, no out-of-range index, and no allocation sized by an unvalidated
//!   length field: every count is checked against the minimum bytes its items
//!   would occupy before any `Vec` is reserved.
//! * Every value id referenced anywhere is `< n_values`.
//! * Every declared dimension is non-zero, `rank <= MAX_RANK`, and the element
//!   count equals the product of the dimensions.
//! * Op arity and attribute count match the opcode.
//!
//! # What the parser does not guarantee
//!
//! Shape agreement *between* ops, single-assignment of value ids, and
//! topological ordering are not checked here — they are checked in
//! [`crate::exec::Graph::from_artifact`], which needs the same walk anyway.
//! A successfully loaded [`Artifact`] is well-formed bytes, not a runnable
//! graph.

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use sha2::{Digest, Sha256};

/// Magic bytes at offset 0 of every `.sealg` artifact.
pub const MAGIC: &[u8; 8] = b"SEALG001";
/// Format version this build emits and accepts.
pub const FORMAT_VERSION: u32 = 1;
/// Fixed header size: magic + version + payload length + digest.
pub const HEADER_LEN: usize = 48;
/// Length of the SHA-256 digest in the header, and of a wire frame trailer.
pub const DIGEST_LEN: usize = 32;

/// Highest tensor rank the format admits. `Conv2d` needs 4.
pub const MAX_RANK: usize = 6;
/// Longest value name in bytes.
pub const MAX_NAME_LEN: usize = 256;
/// Most distinct values (edges) a graph may name.
pub const MAX_VALUES: usize = 65_536;
/// Most ops a graph may contain.
pub const MAX_OPS: usize = 65_536;
/// Most attributes a single op may carry.
pub const MAX_ATTRS: usize = 16;
/// Most inputs a single op may consume.
pub const MAX_OP_INPUTS: usize = 8;
/// Most `f64` elements in one tensor (16Mi elements = 128 MiB).
pub const MAX_TENSOR_ELEMS: usize = 1 << 24;

/// Everything that can go wrong turning untrusted bytes into an [`Artifact`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArtifactError {
    /// Buffer is smaller than the fixed 48-byte header.
    TooShort,
    /// First 8 bytes are not [`MAGIC`].
    BadMagic,
    /// Version field names a format this build cannot read.
    UnsupportedVersion(u32),
    /// Declared payload length does not match the bytes actually supplied.
    LengthMismatch,
    /// SHA-256 over the payload disagrees with the header digest.
    DigestMismatch,
    /// A field ran past the end of the payload.
    Truncated,
    /// Payload parsed successfully but bytes remain after the last op.
    TrailingBytes,
    /// A count or size exceeded its compile-time cap; the field is named.
    LimitExceeded(&'static str),
    /// A value name was not valid UTF-8.
    BadUtf8,
    /// Opcode byte does not name an op this build implements.
    BadOpcode(u8),
    /// Op carries the wrong number of inputs or attributes for its opcode.
    BadArity(OpCode),
    /// A value id referenced a slot outside the value-name table.
    UnknownValue(u32),
    /// A declared dimension was zero.
    ZeroDim,
    /// Declared element count is not the product of the declared dimensions.
    ShapeElemMismatch,
    /// A size computation overflowed `usize`.
    Overflow,
}

/// A dense `f64` tensor as it appears in an artifact or on the wire.
///
/// Deliberately independent of `aether_core::ml::Tensor`: parsing produces
/// plain data with no lock and no `Arc`, and [`crate::exec`] converts.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorDef {
    /// Dimensions, outermost first. Every entry is non-zero.
    pub shape: Vec<usize>,
    /// Row-major elements; `data.len() == shape.iter().product()`.
    pub data: Vec<f64>,
}

impl TensorDef {
    /// Build a tensor definition, checking that the data length matches the shape.
    pub fn new(shape: Vec<usize>, data: Vec<f64>) -> Result<Self, ArtifactError> {
        if shape.contains(&0) {
            return Err(ArtifactError::ZeroDim);
        }
        if elem_count(&shape)? != data.len() {
            return Err(ArtifactError::ShapeElemMismatch);
        }
        Ok(Self { shape, data })
    }
}

/// Product of a shape, refusing overflow instead of wrapping.
pub fn elem_count(shape: &[usize]) -> Result<usize, ArtifactError> {
    let mut n: usize = 1;
    for &d in shape {
        n = n.checked_mul(d).ok_or(ArtifactError::Overflow)?;
    }
    Ok(n)
}

/// The op set this build executes.
///
/// Chosen as the smallest ONNX subset that runs an MLP and a small CNN. The
/// discriminants are the on-disk opcode bytes and are stable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum OpCode {
    /// `C = A @ B`, both rank 2. No attributes.
    MatMul = 0x01,
    /// `Y = op(A) @ op(B) + C`. Attributes `[trans_a, trans_b]`, each 0 or 1.
    /// `C` is rank 1 matching the trailing dimension, or rank 2 matching `Y`.
    Gemm = 0x02,
    /// Element-wise add. Equal shapes, or rank-1 rhs matching the trailing dim.
    Add = 0x03,
    /// Element-wise multiply, same shape rule as [`OpCode::Add`].
    Mul = 0x04,
    /// `max(x, 0)`.
    Relu = 0x05,
    /// Logistic sigmoid.
    Sigmoid = 0x06,
    /// Hyperbolic tangent.
    Tanh = 0x07,
    /// Softmax over the trailing axis only. No axis attribute.
    Softmax = 0x08,
    /// Reshape. Attributes are the target dims; at most one may be `-1`.
    Reshape = 0x09,
    /// Transpose. Attributes are a permutation of `0..rank`.
    Transpose = 0x0A,
    /// 2-D convolution, NCHW, groups 1, dilation 1. Inputs `X[N,C,H,W]`,
    /// `W[M,C,kh,kw]`, `B[M]`. Attributes `[stride_h, stride_w, pad_h, pad_w]`.
    Conv2d = 0x0B,
    /// 2-D max pool, NCHW, no padding. Attributes `[kh, kw, stride_h, stride_w]`.
    MaxPool2d = 0x0C,
}

impl OpCode {
    /// Decode an opcode byte.
    pub fn from_u8(v: u8) -> Option<Self> {
        Some(match v {
            0x01 => OpCode::MatMul,
            0x02 => OpCode::Gemm,
            0x03 => OpCode::Add,
            0x04 => OpCode::Mul,
            0x05 => OpCode::Relu,
            0x06 => OpCode::Sigmoid,
            0x07 => OpCode::Tanh,
            0x08 => OpCode::Softmax,
            0x09 => OpCode::Reshape,
            0x0A => OpCode::Transpose,
            0x0B => OpCode::Conv2d,
            0x0C => OpCode::MaxPool2d,
            _ => return None,
        })
    }

    /// Exact number of inputs this op consumes.
    pub fn arity(self) -> usize {
        match self {
            OpCode::Relu
            | OpCode::Sigmoid
            | OpCode::Tanh
            | OpCode::Softmax
            | OpCode::Reshape
            | OpCode::Transpose
            | OpCode::MaxPool2d => 1,
            OpCode::MatMul | OpCode::Add | OpCode::Mul => 2,
            OpCode::Gemm | OpCode::Conv2d => 3,
        }
    }

    /// Exact attribute count, or `None` when the count is variable (`Reshape`
    /// and `Transpose` carry one attribute per output dimension).
    pub fn attr_count(self) -> Option<usize> {
        match self {
            OpCode::MatMul
            | OpCode::Add
            | OpCode::Mul
            | OpCode::Relu
            | OpCode::Sigmoid
            | OpCode::Tanh
            | OpCode::Softmax => Some(0),
            OpCode::Gemm => Some(2),
            OpCode::Conv2d | OpCode::MaxPool2d => Some(4),
            OpCode::Reshape | OpCode::Transpose => None,
        }
    }
}

/// One node of the graph. Single output; every op in the set produces one value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Op {
    /// Which op to run.
    pub code: OpCode,
    /// Input value ids, in the order the opcode documents.
    pub inputs: Vec<u32>,
    /// Value id this op defines.
    pub output: u32,
    /// Positional attributes; meaning is per-opcode, see [`OpCode`].
    pub attrs: Vec<i64>,
}

/// A parsed, structurally valid `.sealg` artifact.
#[derive(Debug, Clone, PartialEq)]
pub struct Artifact {
    /// Name of every value; the index is the value id.
    pub value_names: Vec<String>,
    /// Graph inputs as `(value_id, declared shape)`.
    pub inputs: Vec<(u32, Vec<usize>)>,
    /// Graph outputs as value ids.
    pub outputs: Vec<u32>,
    /// Weights as `(value_id, tensor)`.
    pub initializers: Vec<(u32, TensorDef)>,
    /// Ops in the order they must be executed.
    pub ops: Vec<Op>,
}

// ── Encoding ────────────────────────────────────────────────────────────────

fn push_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn push_i64(buf: &mut Vec<u8>, v: i64) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn push_f64(buf: &mut Vec<u8>, v: f64) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn push_name(buf: &mut Vec<u8>, s: &str) {
    push_u32(buf, s.len() as u32);
    buf.extend_from_slice(s.as_bytes());
}

fn push_dims(buf: &mut Vec<u8>, dims: &[usize]) {
    push_u32(buf, dims.len() as u32);
    for &d in dims {
        push_u32(buf, d as u32);
    }
}

/// Append a shape-and-data tensor body. Shared with the wire codec.
pub(crate) fn push_tensor_def(buf: &mut Vec<u8>, t: &TensorDef) {
    push_dims(buf, &t.shape);
    push_u32(buf, t.data.len() as u32);
    for &v in &t.data {
        push_f64(buf, v);
    }
}

/// SHA-256 of `bytes`. Used for both the artifact digest and the frame trailer.
pub(crate) fn digest32(bytes: &[u8]) -> [u8; DIGEST_LEN] {
    let mut h = Sha256::new();
    h.update(bytes);
    let out = h.finalize();
    let mut d = [0u8; DIGEST_LEN];
    d.copy_from_slice(&out);
    d
}

impl Artifact {
    /// Serialize to `.sealg` bytes, header and digest included.
    pub fn encode(&self) -> Vec<u8> {
        let mut p = Vec::new();

        push_u32(&mut p, self.value_names.len() as u32);
        for n in &self.value_names {
            push_name(&mut p, n);
        }

        push_u32(&mut p, self.inputs.len() as u32);
        for (id, shape) in &self.inputs {
            push_u32(&mut p, *id);
            push_dims(&mut p, shape);
        }

        push_u32(&mut p, self.outputs.len() as u32);
        for id in &self.outputs {
            push_u32(&mut p, *id);
        }

        push_u32(&mut p, self.initializers.len() as u32);
        for (id, t) in &self.initializers {
            push_u32(&mut p, *id);
            push_tensor_def(&mut p, t);
        }

        push_u32(&mut p, self.ops.len() as u32);
        for op in &self.ops {
            p.push(op.code as u8);
            push_u32(&mut p, op.inputs.len() as u32);
            for i in &op.inputs {
                push_u32(&mut p, *i);
            }
            push_u32(&mut p, op.output);
            push_u32(&mut p, op.attrs.len() as u32);
            for a in &op.attrs {
                push_i64(&mut p, *a);
            }
        }

        let mut buf = Vec::with_capacity(HEADER_LEN + p.len());
        buf.extend_from_slice(MAGIC);
        push_u32(&mut buf, FORMAT_VERSION);
        push_u32(&mut buf, p.len() as u32);
        buf.extend_from_slice(&digest32(&p));
        buf.extend_from_slice(&p);
        buf
    }

    /// Parse `.sealg` bytes. Refuses on bad magic, unknown version, length
    /// disagreement, digest mismatch, or any malformed payload field.
    pub fn load(bytes: &[u8]) -> Result<Self, ArtifactError> {
        let header = bytes.get(..HEADER_LEN).ok_or(ArtifactError::TooShort)?;
        if header.get(..8) != Some(&MAGIC[..]) {
            return Err(ArtifactError::BadMagic);
        }
        let mut hr = Reader::new(header);
        hr.skip(8).ok_or(ArtifactError::TooShort)?;
        let version = hr.u32().ok_or(ArtifactError::TooShort)?;
        if version != FORMAT_VERSION {
            return Err(ArtifactError::UnsupportedVersion(version));
        }
        let payload_len = hr.u32().ok_or(ArtifactError::TooShort)? as usize;
        let want = hr.take(DIGEST_LEN).ok_or(ArtifactError::TooShort)?;

        let payload = bytes.get(HEADER_LEN..).ok_or(ArtifactError::TooShort)?;
        if payload.len() != payload_len {
            return Err(ArtifactError::LengthMismatch);
        }
        if digest32(payload) != want {
            return Err(ArtifactError::DigestMismatch);
        }

        let mut r = Reader::new(payload);
        let artifact = Self::decode_payload(&mut r)?;
        if r.remaining() != 0 {
            return Err(ArtifactError::TrailingBytes);
        }
        Ok(artifact)
    }

    fn decode_payload(r: &mut Reader<'_>) -> Result<Self, ArtifactError> {
        // Value names. Each costs at least a 4-byte length prefix.
        let n_values = r.count(MAX_VALUES, "n_values", 4)?;
        let mut value_names = Vec::with_capacity(n_values);
        for _ in 0..n_values {
            value_names.push(r.name()?);
        }
        let n_values_u32 = n_values as u32;
        let check_id = |id: u32| -> Result<u32, ArtifactError> {
            if id < n_values_u32 {
                Ok(id)
            } else {
                Err(ArtifactError::UnknownValue(id))
            }
        };

        // Inputs: value id + rank + dims, so at least 8 bytes each.
        let n_inputs = r.count(MAX_VALUES, "n_inputs", 8)?;
        let mut inputs = Vec::with_capacity(n_inputs);
        for _ in 0..n_inputs {
            let id = check_id(r.u32().ok_or(ArtifactError::Truncated)?)?;
            inputs.push((id, r.dims()?));
        }

        let n_outputs = r.count(MAX_VALUES, "n_outputs", 4)?;
        let mut outputs = Vec::with_capacity(n_outputs);
        for _ in 0..n_outputs {
            outputs.push(check_id(r.u32().ok_or(ArtifactError::Truncated)?)?);
        }

        // Initializers: id + rank + n_elems, so at least 12 bytes each.
        let n_init = r.count(MAX_VALUES, "n_initializers", 12)?;
        let mut initializers = Vec::with_capacity(n_init);
        for _ in 0..n_init {
            let id = check_id(r.u32().ok_or(ArtifactError::Truncated)?)?;
            initializers.push((id, r.tensor_def()?));
        }

        // Ops: opcode + n_in + out + n_attrs, so at least 13 bytes each.
        let n_ops = r.count(MAX_OPS, "n_ops", 13)?;
        let mut ops = Vec::with_capacity(n_ops);
        for _ in 0..n_ops {
            let raw = r.u8().ok_or(ArtifactError::Truncated)?;
            let code = OpCode::from_u8(raw).ok_or(ArtifactError::BadOpcode(raw))?;
            let n_in = r.count(MAX_OP_INPUTS, "op inputs", 4)?;
            if n_in != code.arity() {
                return Err(ArtifactError::BadArity(code));
            }
            let mut op_inputs = Vec::with_capacity(n_in);
            for _ in 0..n_in {
                op_inputs.push(check_id(r.u32().ok_or(ArtifactError::Truncated)?)?);
            }
            let output = check_id(r.u32().ok_or(ArtifactError::Truncated)?)?;
            let n_attrs = r.count(MAX_ATTRS, "op attrs", 8)?;
            match code.attr_count() {
                Some(want) if want != n_attrs => return Err(ArtifactError::BadArity(code)),
                // Reshape and Transpose carry one attribute per output dim.
                None if n_attrs == 0 || n_attrs > MAX_RANK => {
                    return Err(ArtifactError::BadArity(code))
                }
                _ => {}
            }
            let mut attrs = Vec::with_capacity(n_attrs);
            for _ in 0..n_attrs {
                attrs.push(r.i64().ok_or(ArtifactError::Truncated)?);
            }
            ops.push(Op {
                code,
                inputs: op_inputs,
                output,
                attrs,
            });
        }

        Ok(Artifact {
            value_names,
            inputs,
            outputs,
            initializers,
            ops,
        })
    }
}

// ── Decoding primitives ─────────────────────────────────────────────────────

/// Bounds-checked cursor over a byte buffer.
///
/// Every accessor returns `Option`, mirroring `read_u32` in
/// `kernel/seal-os/src/ml_engine.rs`. Nothing here indexes a slice directly.
pub(crate) struct Reader<'a> {
    bytes: &'a [u8],
    off: usize,
}

impl<'a> Reader<'a> {
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, off: 0 }
    }

    pub(crate) fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.off)
    }

    pub(crate) fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let end = self.off.checked_add(n)?;
        let s = self.bytes.get(self.off..end)?;
        self.off = end;
        Some(s)
    }

    fn skip(&mut self, n: usize) -> Option<()> {
        self.take(n).map(|_| ())
    }

    pub(crate) fn u8(&mut self) -> Option<u8> {
        self.take(1).map(|s| s[0])
    }

    pub(crate) fn u32(&mut self) -> Option<u32> {
        let mut a = [0u8; 4];
        a.copy_from_slice(self.take(4)?);
        Some(u32::from_le_bytes(a))
    }

    pub(crate) fn u64(&mut self) -> Option<u64> {
        let mut a = [0u8; 8];
        a.copy_from_slice(self.take(8)?);
        Some(u64::from_le_bytes(a))
    }

    pub(crate) fn i64(&mut self) -> Option<i64> {
        self.u64().map(|v| v as i64)
    }

    pub(crate) fn f64(&mut self) -> Option<f64> {
        self.u64().map(f64::from_bits)
    }

    /// Read a count and refuse it if it exceeds `cap` or if the buffer cannot
    /// possibly hold `count * min_each` further bytes.
    ///
    /// This is the only place a count is allowed to reach `Vec::with_capacity`.
    pub(crate) fn count(
        &mut self,
        cap: usize,
        what: &'static str,
        min_each: usize,
    ) -> Result<usize, ArtifactError> {
        let n = self.u32().ok_or(ArtifactError::Truncated)? as usize;
        if n > cap {
            return Err(ArtifactError::LimitExceeded(what));
        }
        match n.checked_mul(min_each) {
            Some(need) if need <= self.remaining() => Ok(n),
            Some(_) => Err(ArtifactError::Truncated),
            None => Err(ArtifactError::Overflow),
        }
    }

    pub(crate) fn name(&mut self) -> Result<String, ArtifactError> {
        let len = self.u32().ok_or(ArtifactError::Truncated)? as usize;
        if len > MAX_NAME_LEN {
            return Err(ArtifactError::LimitExceeded("name_len"));
        }
        let raw = self.take(len).ok_or(ArtifactError::Truncated)?;
        core::str::from_utf8(raw)
            .map(ToString::to_string)
            .map_err(|_| ArtifactError::BadUtf8)
    }

    pub(crate) fn dims(&mut self) -> Result<Vec<usize>, ArtifactError> {
        let rank = self.u32().ok_or(ArtifactError::Truncated)? as usize;
        if rank > MAX_RANK {
            return Err(ArtifactError::LimitExceeded("rank"));
        }
        let mut dims = Vec::with_capacity(rank);
        for _ in 0..rank {
            let d = self.u32().ok_or(ArtifactError::Truncated)? as usize;
            if d == 0 {
                return Err(ArtifactError::ZeroDim);
            }
            dims.push(d);
        }
        Ok(dims)
    }

    pub(crate) fn tensor_def(&mut self) -> Result<TensorDef, ArtifactError> {
        let shape = self.dims()?;
        let n = self.count(MAX_TENSOR_ELEMS, "n_elems", 8)?;
        if elem_count(&shape)? != n {
            return Err(ArtifactError::ShapeElemMismatch);
        }
        let mut data = Vec::with_capacity(n);
        for _ in 0..n {
            data.push(self.f64().ok_or(ArtifactError::Truncated)?);
        }
        Ok(TensorDef { shape, data })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// Two-layer MLP artifact, also used by the executor tests.
    pub(crate) fn sample() -> Artifact {
        Artifact {
            value_names: vec![
                "x".to_string(),
                "w1".to_string(),
                "b1".to_string(),
                "h".to_string(),
                "y".to_string(),
            ],
            inputs: vec![(0, vec![1, 2])],
            outputs: vec![4],
            initializers: vec![
                (
                    1,
                    TensorDef::new(vec![2, 2], vec![1.0, -2.0, 0.5, 1.0]).expect("w1"),
                ),
                (2, TensorDef::new(vec![2], vec![0.5, -0.5]).expect("b1")),
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
            ],
        }
    }

    #[test]
    fn round_trip_preserves_everything() {
        let a = sample();
        let bytes = a.encode();
        assert_eq!(&bytes[..8], MAGIC);
        let back = Artifact::load(&bytes).expect("load");
        assert_eq!(a, back);
    }

    #[test]
    fn encode_is_byte_stable() {
        assert_eq!(sample().encode(), sample().encode());
    }

    #[test]
    fn rejects_bad_magic() {
        let mut b = sample().encode();
        b[0] = b'X';
        assert_eq!(Artifact::load(&b), Err(ArtifactError::BadMagic));
    }

    #[test]
    fn rejects_unknown_version() {
        let mut b = sample().encode();
        b[8] = 9;
        assert_eq!(
            Artifact::load(&b),
            Err(ArtifactError::UnsupportedVersion(9))
        );
    }

    #[test]
    fn rejects_flipped_payload_byte() {
        let mut b = sample().encode();
        let last = b.len() - 1;
        b[last] ^= 0x01;
        assert_eq!(Artifact::load(&b), Err(ArtifactError::DigestMismatch));
    }

    #[test]
    fn rejects_every_truncation() {
        // Every prefix of a valid artifact must be refused, never panic.
        let full = sample().encode();
        for cut in 0..full.len() {
            let err = Artifact::load(&full[..cut]).expect_err("prefix must be refused");
            assert!(
                matches!(err, ArtifactError::TooShort | ArtifactError::LengthMismatch),
                "cut {cut} gave {err:?}"
            );
        }
        assert!(Artifact::load(&full).is_ok());
    }

    #[test]
    fn rejects_extended_buffer() {
        let mut b = sample().encode();
        b.push(0);
        assert_eq!(Artifact::load(&b), Err(ArtifactError::LengthMismatch));
    }

    /// Re-digest a payload so a mutated payload reaches the field parser
    /// instead of stopping at the digest check.
    fn seal(payload: Vec<u8>) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(MAGIC);
        b.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        b.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        b.extend_from_slice(&digest32(&payload));
        b.extend_from_slice(&payload);
        b
    }

    #[test]
    fn rejects_huge_value_count_without_allocating() {
        // n_values = u32::MAX with nothing behind it.
        let payload = u32::MAX.to_le_bytes().to_vec();
        assert_eq!(
            Artifact::load(&seal(payload)),
            Err(ArtifactError::LimitExceeded("n_values"))
        );
    }

    #[test]
    fn rejects_plausible_but_unbacked_count() {
        // A count under the cap but with no bytes behind it.
        let payload = 1000u32.to_le_bytes().to_vec();
        assert_eq!(
            Artifact::load(&seal(payload)),
            Err(ArtifactError::Truncated)
        );
    }

    #[test]
    fn rejects_huge_tensor_element_count() {
        let mut p = Vec::new();
        p.extend_from_slice(&1u32.to_le_bytes()); // one value
        p.extend_from_slice(&1u32.to_le_bytes()); // name len 1
        p.push(b'w');
        p.extend_from_slice(&0u32.to_le_bytes()); // no inputs
        p.extend_from_slice(&0u32.to_le_bytes()); // no outputs
        p.extend_from_slice(&1u32.to_le_bytes()); // one initializer
        p.extend_from_slice(&0u32.to_le_bytes()); // value id 0
        p.extend_from_slice(&1u32.to_le_bytes()); // rank 1
        p.extend_from_slice(&4u32.to_le_bytes()); // dim 4
        p.extend_from_slice(&u32::MAX.to_le_bytes()); // n_elems = 4 Gi
        assert_eq!(
            Artifact::load(&seal(p)),
            Err(ArtifactError::LimitExceeded("n_elems"))
        );
    }

    #[test]
    fn rejects_shape_that_disagrees_with_element_count() {
        let mut a = sample();
        // 2x2 shape, 4 elements — claim 3 elements instead.
        a.initializers[0].1.shape = vec![2, 3];
        let bytes = a.encode();
        assert_eq!(
            Artifact::load(&bytes),
            Err(ArtifactError::ShapeElemMismatch)
        );
    }

    #[test]
    fn rejects_zero_dimension() {
        let mut a = sample();
        a.inputs[0].1 = vec![1, 0];
        assert_eq!(Artifact::load(&a.encode()), Err(ArtifactError::ZeroDim));
    }

    #[test]
    fn rejects_value_id_past_the_name_table() {
        let mut a = sample();
        a.outputs = vec![99];
        assert_eq!(
            Artifact::load(&a.encode()),
            Err(ArtifactError::UnknownValue(99))
        );
    }

    #[test]
    fn rejects_rank_beyond_cap() {
        let mut a = sample();
        a.inputs[0].1 = vec![1, 1, 1, 1, 1, 1, 1];
        assert_eq!(
            Artifact::load(&a.encode()),
            Err(ArtifactError::LimitExceeded("rank"))
        );
    }

    #[test]
    fn rejects_unknown_opcode() {
        // Minimal payload: one value, no bindings, one op with opcode 0xFE.
        let mut p = Vec::new();
        p.extend_from_slice(&1u32.to_le_bytes()); // n_values
        p.extend_from_slice(&1u32.to_le_bytes()); // name len
        p.push(b'v');
        p.extend_from_slice(&0u32.to_le_bytes()); // n_inputs
        p.extend_from_slice(&0u32.to_le_bytes()); // n_outputs
        p.extend_from_slice(&0u32.to_le_bytes()); // n_initializers
        p.extend_from_slice(&1u32.to_le_bytes()); // n_ops
        p.push(0xFE);
        p.extend_from_slice(&[0u8; 12]); // n_in, output, n_attrs
        assert_eq!(
            Artifact::load(&seal(p)),
            Err(ArtifactError::BadOpcode(0xFE))
        );
    }

    #[test]
    fn rejects_wrong_arity() {
        let mut a = sample();
        a.ops[0].inputs = vec![0, 1]; // Gemm needs 3
        assert_eq!(
            Artifact::load(&a.encode()),
            Err(ArtifactError::BadArity(OpCode::Gemm))
        );
    }

    #[test]
    fn rejects_wrong_attribute_count() {
        let mut a = sample();
        a.ops[0].attrs = vec![0]; // Gemm needs 2
        assert_eq!(
            Artifact::load(&a.encode()),
            Err(ArtifactError::BadArity(OpCode::Gemm))
        );
    }

    #[test]
    fn every_single_bit_mutation_is_refused() {
        // The digest makes every payload mutation detectable; header mutations
        // are caught by the magic, version, or length checks. Nothing panics.
        let full = sample().encode();
        for i in 0..full.len() {
            for bit in [0x01u8, 0x80] {
                let mut b = full.clone();
                b[i] ^= bit;
                assert!(Artifact::load(&b).is_err(), "mutation at {i} accepted");
            }
        }
    }

    #[test]
    fn opcode_byte_round_trip() {
        for raw in 0u8..=255 {
            match OpCode::from_u8(raw) {
                Some(c) => assert_eq!(c as u8, raw),
                None => assert!(!(0x01..=0x0C).contains(&raw)),
            }
        }
    }
}
