// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! `seal-graph` — the `.sealg` artifact format, its transport codec, and a
//! forward-only graph executor for Seal OS.
//!
//! Three layers, each usable without the others:
//!
//! * [`artifact`] — the `.sealg` byte format. A host exporter (`torch.export`,
//!   a SavedModel walker) emits it; [`Artifact::load`] parses it. Every decode
//!   path is bounds-checked and returns [`ArtifactError`]; nothing in this
//!   module panics on adversarial bytes.
//! * [`wire`] — a length-framed message codec generic over a [`Transport`].
//!   The codec has no knowledge of serial ports or sockets; supplying a
//!   `Transport` impl is what picks the link.
//! * [`exec`] — validate-then-run interpreter over [`aether_core::ml::Tensor`].
//!
//! # What this crate does
//!
//! Loads a `.sealg` artifact, statically checks every shape in it, refuses the
//! whole graph if any check fails, and otherwise evaluates it in `f64` in the
//! artifact's declared op order. Same artifact plus same inputs yields
//! bit-identical outputs — there is no threading, no reduction reordering and
//! no fast-math in any of the kernels.
//!
//! # What this crate does not do
//!
//! * **No training and no autograd.** Forward evaluation only. Nothing here
//!   produces a gradient; `aether_core::ml::autograd` is not wired in.
//! * **No dtypes other than `f64`.** No quantization, no `f32`, no integers.
//!   A `.sealg` file is `f64` end to end, which makes it roughly 2x larger
//!   than the `f32` export it came from.
//! * **No general broadcasting.** `Add`/`Mul` accept equal shapes or a rank-1
//!   right-hand operand matching the trailing dimension. That covers a bias
//!   add and nothing else.
//! * **No dynamic shapes.** Input shapes are fixed by the artifact and
//!   checked exactly; a graph exported for batch 1 runs at batch 1.
//! * **No ONNX ingestion.** The op codes are named after their ONNX
//!   counterparts, but no `.onnx` protobuf is parsed here. Conversion is the
//!   exporter's job.
//! * **No authenticity.** The SHA-256 digest detects corruption and truncation.
//!   It is not a signature: anyone who can supply bytes can supply a matching
//!   digest. Artifact provenance is out of scope for this crate.
//! * **No syscall dispatch.** [`wire::Message::FitObserve`] and
//!   [`wire::Message::FitRegime`] carry the arguments of `SYS_FIT_OBSERVE`
//!   (121) and `SYS_FIT_REGIME` (122); issuing the syscalls is the caller's
//!   job. This crate never touches kernel state.
//! * **No performance work.** The kernels are naive loop nests that copy tensor
//!   data out from under its lock before operating. Correctness and the absence
//!   of panics were the goals; throughput was not measured.

#![cfg_attr(not(feature = "std"), no_std)]
#![deny(unsafe_code)]
#![warn(missing_docs)]

extern crate alloc;

pub mod artifact;
pub mod exec;
pub mod wire;

pub use artifact::{Artifact, ArtifactError, Op, OpCode, TensorDef, FORMAT_VERSION, MAGIC};
pub use exec::{ExecError, Graph};
pub use wire::{Message, NamedTensor, Transport, WireError};
