// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Memory amplification: a small artifact that commits the executor to a large
//! allocation.
//!
//! # The gap this documents
//!
//! `MAX_TENSOR_ELEMS` bounds any *single* tensor at 2^24 elements — 134 MB at
//! `f64`. Nothing bounds the *sum* across a graph's intermediates, and
//! `exec`'s own module docs state that "a graph's peak memory is the sum of all
//! its intermediates" because no buffer is reused.
//!
//! Those two facts compose badly. An outer product `[n,1] × [1,n]` reads `2n`
//! elements and writes `n²`, so the input is not a proxy for the output. At
//! `n = 4096` a **64 KB** pair of operands commits the executor to a **134 MB**
//! allocation: an amplification factor of roughly 2,000×. `MAX_OPS` permits
//! 65,536 ops, so the artifact can repeat the trick.
//!
//! The wire codec's `MAX_FRAME_BODY` (8 MiB) bounds what arrives over a
//! transport but does not help here, because the amplifying artifact is tiny.
//!
//! # Why this is a kernel problem specifically
//!
//! On a host, an over-large allocation fails and a process dies. In a `no_std`
//! UEFI kernel there is no process boundary to contain it. Once `seal-graph` is
//! reachable from a syscall or a serial listener — which is the next step — this
//! becomes an unauthenticated remote allocation primitive.
//!
//! # Status
//!
//! Closed. `Graph::from_artifact` now charges every shape — declared *and*
//! inferred — against a per-tensor ceiling (`MAX_TENSOR_ELEMS`) and a
//! whole-graph ceiling (`MAX_GRAPH_ELEMS`), refusing during validation before
//! `run` allocates anything. These tests assert the refusals and confirm that
//! an ordinary graph still loads.
//!
//! Nothing here calls `run`, so no test in this file allocates at scale.

use seal_graph::artifact::{Artifact, Op, OpCode, MAX_TENSOR_ELEMS};
use seal_graph::exec::{ExecError, Graph};

/// `[n,1] @ [1,n] -> [n,n]`. Reads 2n elements, writes n².
fn outer_product_artifact(n: usize) -> Artifact {
    Artifact {
        value_names: vec!["col".into(), "row".into(), "out".into()],
        inputs: vec![(0, vec![n, 1]), (1, vec![1, n])],
        outputs: vec![2],
        initializers: vec![],
        ops: vec![Op {
            code: OpCode::MatMul,
            inputs: vec![0, 1],
            output: 2,
            attrs: vec![],
        }],
    }
}

/// The amplification cases, spanning what used to be accepted. Each is a tiny
/// artifact whose *inferred* output is enormous; none is bounded by file size.
#[test]
fn amplifying_artifacts_are_refused_at_validation() {
    // 4096² = 16.7M elems, exactly at MAX_TENSOR_ELEMS: the largest legal
    // single tensor. With its two operands charged too the graph total is
    // 16.7M + 8K, still under MAX_GRAPH_ELEMS, so this one must still load.
    // It is the positive control — a budget that refuses everything is not a
    // budget, it is an outage.
    let ok = outer_product_artifact(4096).encode();
    assert!(
        Graph::load(&ok).is_ok(),
        "a graph at the per-tensor ceiling and under the graph ceiling must load"
    );

    println!("\n=== seal-graph amplification, post-fix ===");
    for n in [8192usize, 65_536, 1_000_000, 2_000_000_000] {
        let bytes = outer_product_artifact(n).encode();
        let elems = (n as u128) * (n as u128);
        let gib = (elems * 8) as f64 / 1_073_741_824.0;
        let verdict = Graph::load(&bytes);
        println!(
            "  n={n:<12} artifact {:>4} B  inferred {elems:>22} elems ({gib:>14.1} GiB) -> {}",
            bytes.len(),
            if verdict.is_err() {
                "refused"
            } else {
                "ACCEPTED"
            }
        );
        assert!(
            verdict.is_err(),
            "n={n} implies {gib:.1} GiB and must be refused during validation"
        );
    }
}

/// The refusal must be explainable, not an opaque failure. A caller needs to
/// know whether one tensor was absurd or the graph merely did not fit.
#[test]
fn refusal_names_which_ceiling_was_hit() {
    let bytes = outer_product_artifact(8192).encode();
    match Graph::load(&bytes) {
        Err(ExecError::TensorTooLarge { elems, limit }) => {
            assert_eq!(elems, 8192 * 8192);
            assert_eq!(limit, MAX_TENSOR_ELEMS);
        }
        other => panic!("expected TensorTooLarge, got {other:?}"),
    }
}

/// A whole-graph refusal is reachable without any single tensor being illegal.
/// Twelve tensors of 8.39M elements each are individually well under
/// `MAX_TENSOR_ELEMS` and collectively over `MAX_GRAPH_ELEMS` — which is the
/// case a per-tensor cap alone can never catch.
#[test]
fn many_legal_tensors_can_still_exceed_the_graph_budget() {
    const SIDE: usize = 2896; // 2896² = 8.39M elems, well under MAX_TENSOR_ELEMS
    const COPIES: usize = 12;

    let mut value_names = vec!["col".to_string(), "row".to_string()];
    let mut ops = Vec::new();
    for i in 0..COPIES {
        value_names.push(alloc_name(i));
        ops.push(Op {
            code: OpCode::MatMul,
            inputs: vec![0, 1],
            output: (2 + i) as u32,
            attrs: vec![],
        });
    }

    let artifact = Artifact {
        value_names,
        inputs: vec![(0, vec![SIDE, 1]), (1, vec![1, SIDE])],
        outputs: vec![2],
        initializers: vec![],
        ops,
    };

    match Graph::load(&artifact.encode()) {
        Err(ExecError::GraphTooLarge { elems, limit }) => {
            assert!(elems > limit, "{elems} should exceed {limit}");
            println!(
                "\n  {COPIES} tensors of {}² = {} elems each -> GraphTooLarge at {elems} (limit {limit})",
                SIDE,
                SIDE * SIDE
            );
        }
        other => panic!("expected GraphTooLarge, got {other:?}"),
    }
}

fn alloc_name(i: usize) -> String {
    let mut s = String::from("t");
    s.push_str(&i.to_string());
    s
}
