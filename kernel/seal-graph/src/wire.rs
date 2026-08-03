// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Length-framed message codec, generic over the link.
//!
//! The codec knows nothing about serial ports or sockets. It moves bytes
//! through a [`Transport`], so the same frames run over a UART today and a TCP
//! stream later with no change here — implementing [`Transport`] is the whole
//! porting cost.
//!
//! # Frame layout
//!
//! ```text
//!   u32       body_len          little-endian, 1 ..= MAX_FRAME_BODY
//!   u8        msg_type          first byte of the body
//!   ...       payload           body_len - 1 bytes
//!   [u8; 32]  SHA-256(body)     body = msg_type byte followed by payload
//! ```
//!
//! `body_len` counts the type byte and the payload, and excludes both the
//! length prefix and the digest trailer. A frame is therefore
//! `4 + body_len + 32` bytes on the link.
//!
//! # Message type codes
//!
//! | code | message | payload |
//! |------|---------|---------|
//! | 0x01 | `Hello` | `u32 version` |
//! | 0x02 | `PushArtifact` | `u32 len; len bytes` (a `.sealg` blob) |
//! | 0x03 | `RunGraph` | `u32 n; n x named tensor` |
//! | 0x04 | `FitObserve` | `u64 handle; f64 train; f64 val` |
//! | 0x05 | `FitRegime` | `u64 handle` |
//! | 0x06 | `Result` | `u64 code; u32 n; n x named tensor` |
//! | 0x07 | `Error` | `u32 code; u32 len; len bytes UTF-8` |
//!
//! A named tensor is `u32 name_len; name bytes; u32 rank; rank x u32 dim;
//! u32 n_elems; n_elems x f64` — the same tensor body the artifact format uses.
//!
//! `FitObserve` encodes `train` and `val` as IEEE-754 bit patterns, which is
//! exactly what `SYS_FIT_OBSERVE` (121) expects in `arg1` and `arg2`. The codec
//! does not issue the syscall; it only carries the arguments.
//!
//! # What this codec does not do
//!
//! No retransmission, no sequence numbers, no flow control, no encryption, and
//! no authentication. The digest catches a corrupted or truncated frame, not a
//! forged one. There is no session state: [`read_message`] returns one message
//! and takes no view of what came before it.

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use crate::artifact::{
    digest32, push_tensor_def, ArtifactError, Reader, TensorDef, DIGEST_LEN, MAX_NAME_LEN,
    MAX_VALUES,
};

/// Protocol version carried by [`Message::Hello`].
pub const PROTOCOL_VERSION: u32 = 1;

/// Largest frame body accepted, in bytes. Compile-time; nothing raises it at
/// run time. A frame declaring more is refused before any buffer is allocated.
pub const MAX_FRAME_BODY: usize = 8 * 1024 * 1024;

/// A byte link. Blocking, exact-length, no partial reads visible to the codec.
pub trait Transport {
    /// Link-specific failure.
    type Error;
    /// Fill `buf` completely or fail.
    fn read_exact(&mut self, buf: &mut [u8]) -> Result<(), Self::Error>;
    /// Write all of `buf` or fail.
    fn write_all(&mut self, buf: &[u8]) -> Result<(), Self::Error>;
}

/// A tensor with the name it binds to in the graph.
#[derive(Debug, Clone, PartialEq)]
pub struct NamedTensor {
    /// Value name, matching a name in the artifact's value table.
    pub name: String,
    /// Shape and data.
    pub value: TensorDef,
}

/// Every message the protocol defines.
#[derive(Debug, Clone, PartialEq)]
pub enum Message {
    /// Version handshake, sent first by the host.
    Hello {
        /// Protocol version the sender speaks.
        version: u32,
    },
    /// A `.sealg` artifact to load, verbatim.
    PushArtifact {
        /// Artifact bytes, parsed by `Artifact::load` on the far side.
        bytes: Vec<u8>,
    },
    /// Run the loaded graph on these inputs.
    RunGraph {
        /// One entry per declared graph input.
        inputs: Vec<NamedTensor>,
    },
    /// Report a training/validation loss pair — `SYS_FIT_OBSERVE` arguments.
    FitObserve {
        /// Fit-controller handle from `SYS_FIT_REGISTER`.
        handle: u64,
        /// Training loss.
        train: f64,
        /// Validation loss.
        val: f64,
    },
    /// Ask for the current fit regime — `SYS_FIT_REGIME` arguments.
    FitRegime {
        /// Fit-controller handle.
        handle: u64,
    },
    /// A successful reply. `code` carries the scalar answer (a regime code, a
    /// handle, or zero); `values` carries graph outputs when there are any.
    Result {
        /// Scalar result, zero when the reply is purely tensors.
        code: u64,
        /// Named output tensors, empty for scalar replies.
        values: Vec<NamedTensor>,
    },
    /// A failure reply.
    Error {
        /// Caller-defined error code.
        code: u32,
        /// Human-readable detail.
        msg: String,
    },
}

/// Codec failure. `E` is the transport's own error type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WireError<E> {
    /// The link failed.
    Transport(E),
    /// Declared body length exceeds [`MAX_FRAME_BODY`]. Nothing was allocated.
    FrameTooLarge(u32),
    /// SHA-256 trailer disagrees with the body.
    DigestMismatch,
    /// Type byte does not name a message.
    BadMessageType(u8),
    /// The body was malformed; the artifact decoder says how.
    Decode(ArtifactError),
}

impl<E> From<ArtifactError> for WireError<E> {
    fn from(e: ArtifactError) -> Self {
        WireError::Decode(e)
    }
}

fn push_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn push_u64(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn push_named(buf: &mut Vec<u8>, t: &NamedTensor) {
    push_u32(buf, t.name.len() as u32);
    buf.extend_from_slice(t.name.as_bytes());
    push_tensor_def(buf, &t.value);
}

fn read_named(r: &mut Reader<'_>) -> Result<NamedTensor, ArtifactError> {
    let name = r.name()?;
    let value = r.tensor_def()?;
    Ok(NamedTensor { name, value })
}

fn read_named_list(r: &mut Reader<'_>) -> Result<Vec<NamedTensor>, ArtifactError> {
    // A named tensor costs at least 16 bytes: name len, rank, one dim, n_elems.
    let n = r.count(MAX_VALUES, "named tensors", 16)?;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(read_named(r)?);
    }
    Ok(out)
}

impl Message {
    /// Type code as it appears on the link.
    pub fn type_code(&self) -> u8 {
        match self {
            Message::Hello { .. } => 0x01,
            Message::PushArtifact { .. } => 0x02,
            Message::RunGraph { .. } => 0x03,
            Message::FitObserve { .. } => 0x04,
            Message::FitRegime { .. } => 0x05,
            Message::Result { .. } => 0x06,
            Message::Error { .. } => 0x07,
        }
    }

    /// Serialize the frame body: type byte followed by payload.
    fn encode_body(&self) -> Vec<u8> {
        let mut b = vec![self.type_code()];
        match self {
            Message::Hello { version } => push_u32(&mut b, *version),
            Message::PushArtifact { bytes } => {
                push_u32(&mut b, bytes.len() as u32);
                b.extend_from_slice(bytes);
            }
            Message::RunGraph { inputs } => {
                push_u32(&mut b, inputs.len() as u32);
                for t in inputs {
                    push_named(&mut b, t);
                }
            }
            Message::FitObserve { handle, train, val } => {
                push_u64(&mut b, *handle);
                push_u64(&mut b, train.to_bits());
                push_u64(&mut b, val.to_bits());
            }
            Message::FitRegime { handle } => push_u64(&mut b, *handle),
            Message::Result { code, values } => {
                push_u64(&mut b, *code);
                push_u32(&mut b, values.len() as u32);
                for t in values {
                    push_named(&mut b, t);
                }
            }
            Message::Error { code, msg } => {
                push_u32(&mut b, *code);
                push_u32(&mut b, msg.len() as u32);
                b.extend_from_slice(msg.as_bytes());
            }
        }
        b
    }

    /// Parse a frame body whose digest has already been verified.
    fn decode_body(body: &[u8]) -> Result<Self, ArtifactError> {
        let mut r = Reader::new(body);
        let ty = r.u8().ok_or(ArtifactError::Truncated)?;
        let msg = match ty {
            0x01 => Message::Hello {
                version: r.u32().ok_or(ArtifactError::Truncated)?,
            },
            0x02 => {
                let n = r.count(MAX_FRAME_BODY, "artifact bytes", 1)?;
                let bytes = r.take(n).ok_or(ArtifactError::Truncated)?.to_vec();
                Message::PushArtifact { bytes }
            }
            0x03 => Message::RunGraph {
                inputs: read_named_list(&mut r)?,
            },
            0x04 => Message::FitObserve {
                handle: r.u64().ok_or(ArtifactError::Truncated)?,
                train: r.f64().ok_or(ArtifactError::Truncated)?,
                val: r.f64().ok_or(ArtifactError::Truncated)?,
            },
            0x05 => Message::FitRegime {
                handle: r.u64().ok_or(ArtifactError::Truncated)?,
            },
            0x06 => Message::Result {
                code: r.u64().ok_or(ArtifactError::Truncated)?,
                values: read_named_list(&mut r)?,
            },
            0x07 => {
                let code = r.u32().ok_or(ArtifactError::Truncated)?;
                let len = r.u32().ok_or(ArtifactError::Truncated)? as usize;
                if len > MAX_NAME_LEN * 8 {
                    return Err(ArtifactError::LimitExceeded("error msg"));
                }
                let raw = r.take(len).ok_or(ArtifactError::Truncated)?;
                let msg = core::str::from_utf8(raw)
                    .map_err(|_| ArtifactError::BadUtf8)?
                    .into();
                Message::Error { code, msg }
            }
            other => return Err(ArtifactError::BadOpcode(other)),
        };
        if r.remaining() != 0 {
            return Err(ArtifactError::TrailingBytes);
        }
        Ok(msg)
    }
}

/// Encode and send one message.
pub fn write_message<T: Transport>(t: &mut T, m: &Message) -> Result<(), WireError<T::Error>> {
    let body = m.encode_body();
    if body.len() > MAX_FRAME_BODY {
        return Err(WireError::FrameTooLarge(body.len() as u32));
    }
    let mut frame = Vec::with_capacity(4 + body.len() + DIGEST_LEN);
    push_u32(&mut frame, body.len() as u32);
    frame.extend_from_slice(&body);
    frame.extend_from_slice(&digest32(&body));
    t.write_all(&frame).map_err(WireError::Transport)
}

/// Receive and decode one message.
///
/// The declared length is checked against [`MAX_FRAME_BODY`] *before* the body
/// buffer is allocated, so a hostile length field cannot drive an allocation.
pub fn read_message<T: Transport>(t: &mut T) -> Result<Message, WireError<T::Error>> {
    let mut len_bytes = [0u8; 4];
    t.read_exact(&mut len_bytes).map_err(WireError::Transport)?;
    let len = u32::from_le_bytes(len_bytes);
    if len == 0 {
        return Err(WireError::Decode(ArtifactError::Truncated));
    }
    if len as usize > MAX_FRAME_BODY {
        return Err(WireError::FrameTooLarge(len));
    }

    let mut body = vec![0u8; len as usize];
    t.read_exact(&mut body).map_err(WireError::Transport)?;
    let mut trailer = [0u8; DIGEST_LEN];
    t.read_exact(&mut trailer).map_err(WireError::Transport)?;
    if digest32(&body) != trailer {
        return Err(WireError::DigestMismatch);
    }

    match Message::decode_body(&body) {
        Ok(m) => Ok(m),
        Err(ArtifactError::BadOpcode(b)) => Err(WireError::BadMessageType(b)),
        Err(e) => Err(WireError::Decode(e)),
    }
}

/// An in-memory loopback link: whatever is written can be read back.
///
/// Exists so the codec can be exercised without hardware. It is not a queue —
/// there is one buffer and one read cursor.
#[derive(Debug, Default)]
pub struct MemTransport {
    buf: Vec<u8>,
    pos: usize,
}

/// The only way a [`MemTransport`] fails: a read past what was written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Eof;

impl MemTransport {
    /// Empty loopback.
    pub fn new() -> Self {
        Self::default()
    }

    /// Loopback pre-loaded with bytes to be read.
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        Self { buf: bytes, pos: 0 }
    }

    /// Bytes written but not yet read back.
    pub fn pending(&self) -> usize {
        self.buf.len().saturating_sub(self.pos)
    }
}

impl Transport for MemTransport {
    type Error = Eof;

    fn read_exact(&mut self, out: &mut [u8]) -> Result<(), Eof> {
        let end = self.pos.checked_add(out.len()).ok_or(Eof)?;
        let src = self.buf.get(self.pos..end).ok_or(Eof)?;
        out.copy_from_slice(src);
        self.pos = end;
        Ok(())
    }

    fn write_all(&mut self, bytes: &[u8]) -> Result<(), Eof> {
        self.buf.extend_from_slice(bytes);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    fn named(name: &str) -> NamedTensor {
        NamedTensor {
            name: name.to_string(),
            value: TensorDef::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).expect("tensor"),
        }
    }

    fn all_messages() -> Vec<Message> {
        vec![
            Message::Hello {
                version: PROTOCOL_VERSION,
            },
            Message::PushArtifact {
                bytes: vec![0xDE, 0xAD, 0xBE, 0xEF],
            },
            Message::RunGraph {
                inputs: vec![named("x"), named("y")],
            },
            Message::FitObserve {
                handle: 7,
                train: 0.125,
                val: -2.5,
            },
            Message::FitRegime { handle: 7 },
            Message::Result {
                code: 3,
                values: vec![named("out")],
            },
            Message::Error {
                code: 22,
                msg: "shape mismatch".to_string(),
            },
        ]
    }

    #[test]
    fn round_trips_every_message_type() {
        let msgs = all_messages();
        let mut t = MemTransport::new();
        for m in &msgs {
            write_message(&mut t, m).expect("write");
        }
        for m in &msgs {
            let got = read_message(&mut t).expect("read");
            assert_eq!(&got, m);
        }
        assert_eq!(t.pending(), 0);
    }

    #[test]
    fn empty_link_reports_transport_error_not_panic() {
        let mut t = MemTransport::new();
        assert_eq!(
            read_message(&mut t).expect_err("must fail"),
            WireError::Transport(Eof)
        );
    }

    #[test]
    fn oversized_length_is_refused_before_allocating() {
        // 4 GiB declared, nothing behind it. Must not try to allocate.
        let mut t = MemTransport::from_bytes(u32::MAX.to_le_bytes().to_vec());
        assert_eq!(
            read_message(&mut t).expect_err("must fail"),
            WireError::FrameTooLarge(u32::MAX)
        );
    }

    #[test]
    fn zero_length_frame_is_refused() {
        let mut t = MemTransport::from_bytes(0u32.to_le_bytes().to_vec());
        assert!(matches!(
            read_message(&mut t),
            Err(WireError::Decode(ArtifactError::Truncated))
        ));
    }

    #[test]
    fn corrupted_body_fails_the_digest() {
        let mut t = MemTransport::new();
        write_message(&mut t, &Message::Hello { version: 1 }).expect("write");
        let mut raw = t.buf.clone();
        raw[5] ^= 0xFF; // first payload byte
        let mut t2 = MemTransport::from_bytes(raw);
        assert_eq!(
            read_message(&mut t2).expect_err("must fail"),
            WireError::DigestMismatch
        );
    }

    #[test]
    fn unknown_type_code_is_named() {
        let body = vec![0x7Fu8, 0, 0, 0, 0];
        let mut raw = (body.len() as u32).to_le_bytes().to_vec();
        raw.extend_from_slice(&body);
        raw.extend_from_slice(&digest32(&body));
        let mut t = MemTransport::from_bytes(raw);
        assert_eq!(
            read_message(&mut t).expect_err("must fail"),
            WireError::BadMessageType(0x7F)
        );
    }

    #[test]
    fn every_truncation_of_a_frame_is_refused() {
        let mut t = MemTransport::new();
        write_message(
            &mut t,
            &Message::RunGraph {
                inputs: vec![named("x")],
            },
        )
        .expect("write");
        let full = t.buf.clone();
        for cut in 0..full.len() {
            let mut t2 = MemTransport::from_bytes(full[..cut].to_vec());
            assert!(read_message(&mut t2).is_err(), "prefix {cut} accepted");
        }
        let mut t3 = MemTransport::from_bytes(full);
        assert!(read_message(&mut t3).is_ok());
    }

    #[test]
    fn trailing_payload_bytes_are_refused() {
        // A Hello body with an extra byte, digest recomputed so the length and
        // digest checks pass and only the strict-decode check can fire.
        let body = vec![0x01u8, 1, 0, 0, 0, 0xAA];
        let mut raw = (body.len() as u32).to_le_bytes().to_vec();
        raw.extend_from_slice(&body);
        raw.extend_from_slice(&digest32(&body));
        let mut t = MemTransport::from_bytes(raw);
        assert_eq!(
            read_message(&mut t).expect_err("must fail"),
            WireError::Decode(ArtifactError::TrailingBytes)
        );
    }

    #[test]
    fn fit_observe_carries_exact_bit_patterns() {
        // The values SYS_FIT_OBSERVE receives must survive the link unchanged.
        let m = Message::FitObserve {
            handle: u64::MAX,
            train: f64::MIN_POSITIVE,
            val: 1.0 / 3.0,
        };
        let mut t = MemTransport::new();
        write_message(&mut t, &m).expect("write");
        match read_message(&mut t).expect("read") {
            Message::FitObserve { handle, train, val } => {
                assert_eq!(handle, u64::MAX);
                assert_eq!(train.to_bits(), f64::MIN_POSITIVE.to_bits());
                assert_eq!(val.to_bits(), (1.0f64 / 3.0).to_bits());
            }
            other => panic!("wrong message: {other:?}"),
        }
    }
}
