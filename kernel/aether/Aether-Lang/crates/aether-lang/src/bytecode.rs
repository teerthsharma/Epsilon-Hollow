// Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: Epsilon-Hollow

//! ═══════════════════════════════════════════════════════════════════════════════
//! Titan Bytecode: Instructions, Verification, Optimization, and Trace Caching
//! ═══════════════════════════════════════════════════════════════════════════════

#[cfg(not(feature = "std"))]
use alloc::string::{String, ToString};
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Titan Bytecode Instructions
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(non_camel_case_types)]
pub enum OpCode {
    /// No-operation (used by peephole optimizer to eliminate folded ops)
    NOP,
    /// Push constant value onto stack
    PUSH(f64),
    /// Push variable value
    LOAD(usize),
    /// Store top of stack to variable
    STORE(usize),

    /// Arithmetic
    ADD,
    SUB,
    MUL,
    DIV,

    /// Topology / Core Logic
    /// Embeds the top value into the manifold
    EMBED,
    /// Checks topological attention/neighbors
    ATTEND,
    /// Explicit entropy regulation point
    PRUNE,

    /// Control Flow
    JMP(isize),
    JMP_IF_FALSE(isize),

    /// Output
    PRINT,

    /// End of program
    HALT,
}

/// Bytecode Verifier: checks stack balance and local bounds before execution.
pub struct BytecodeVerifier;

impl BytecodeVerifier {
    /// Verify that `code` is well-formed:
    /// - Stack never underflows
    /// - Local variable indices are within `max_locals`
    /// - Jump targets are within the code bounds
    pub fn verify(code: &[OpCode], max_locals: usize) -> Result<(), String> {
        let mut stack: isize = 0;

        for (i, op) in code.iter().enumerate() {
            match op {
                OpCode::NOP | OpCode::HALT => {}

                OpCode::PUSH(_) | OpCode::LOAD(_) => {
                    stack += 1;
                }

                OpCode::ADD | OpCode::SUB | OpCode::MUL | OpCode::DIV => {
                    if stack < 2 {
                        return Err(format!(
                            "Verifier: stack underflow at {} ({:?}); stack={}",
                            i, op, stack
                        ));
                    }
                    stack -= 1;
                }

                OpCode::STORE(idx) => {
                    if *idx >= max_locals {
                        return Err(format!(
                            "Verifier: local index {} out of bounds (max {}) at {}",
                            idx, max_locals, i
                        ));
                    }
                    if stack < 1 {
                        return Err(format!(
                            "Verifier: stack underflow at {} ({:?}); stack={}",
                            i, op, stack
                        ));
                    }
                    stack -= 1;
                }

                OpCode::PRINT | OpCode::EMBED | OpCode::ATTEND | OpCode::PRUNE => {
                    if stack < 1 {
                        return Err(format!(
                            "Verifier: stack underflow at {} ({:?}); stack={}",
                            i, op, stack
                        ));
                    }
                    stack -= 1;
                }

                OpCode::JMP(offset) => {
                    let target = i as isize + 1 + offset;
                    if target < 0 || target as usize > code.len() {
                        return Err(format!(
                            "Verifier: jump out of bounds at {} (target {})",
                            i, target
                        ));
                    }
                }

                OpCode::JMP_IF_FALSE(offset) => {
                    if stack < 1 {
                        return Err(format!(
                            "Verifier: stack underflow at {} ({:?}); stack={}",
                            i, op, stack
                        ));
                    }
                    stack -= 1;
                    let target = i as isize + 1 + offset;
                    if target < 0 || target as usize > code.len() {
                        return Err(format!(
                            "Verifier: jump out of bounds at {} (target {})",
                            i, target
                        ));
                    }
                }
            }
        }

        Ok(())
    }
}

/// Peephole optimizer: folds constant arithmetic.
pub struct Peephole;

impl Peephole {
    /// Apply peephole optimizations in-place.
    /// Folded ops are replaced with `NOP`; call `compact` to remove them.
    pub fn run(ops: &mut [OpCode]) {
        let mut i = 0;
        while i + 2 < ops.len() {
            match (ops[i], ops[i + 1], ops[i + 2]) {
                (OpCode::PUSH(a), OpCode::PUSH(b), OpCode::ADD) => {
                    ops[i] = OpCode::PUSH(a + b);
                    ops[i + 1] = OpCode::NOP;
                    ops[i + 2] = OpCode::NOP;
                    i += 3;
                    continue;
                }
                (OpCode::PUSH(a), OpCode::PUSH(b), OpCode::SUB) => {
                    ops[i] = OpCode::PUSH(a - b);
                    ops[i + 1] = OpCode::NOP;
                    ops[i + 2] = OpCode::NOP;
                    i += 3;
                    continue;
                }
                (OpCode::PUSH(a), OpCode::PUSH(b), OpCode::MUL) => {
                    ops[i] = OpCode::PUSH(a * b);
                    ops[i + 1] = OpCode::NOP;
                    ops[i + 2] = OpCode::NOP;
                    i += 3;
                    continue;
                }
                (OpCode::PUSH(a), OpCode::PUSH(b), OpCode::DIV) => {
                    if b != 0.0 {
                        ops[i] = OpCode::PUSH(a / b);
                        ops[i + 1] = OpCode::NOP;
                        ops[i + 2] = OpCode::NOP;
                    }
                    i += 3;
                    continue;
                }
                _ => {}
            }
            i += 1;
        }
    }

    /// Remove all `NOP` instructions from a vector.
    ///
    /// **Warning:** this invalidates jump offsets. Only use when you know
    /// the bytecode has no internal jumps, or after fixing up offsets.
    pub fn compact(ops: &mut Vec<OpCode>) {
        ops.retain(|op| !matches!(op, OpCode::NOP));
    }
}

/// A cached trace: a contiguous slice of bytecode copied for fast execution.
#[derive(Debug, Clone)]
pub struct Trace {
    pub start_ip: usize,
    pub ops: Vec<OpCode>,
}

/// Trace cache for hot loops.
///
/// Detects backward jumps, counts loop-header hits, and compiles traces
/// of up to 256 ops when a threshold is crossed.
pub struct TraceCache {
    traces: Vec<(usize, Trace)>,
    counters: Vec<(usize, u64)>,
    threshold: u64,
    max_traces: usize,
    max_trace_len: usize,
}

impl TraceCache {
    pub fn new() -> Self {
        Self {
            traces: Vec::with_capacity(16),
            counters: Vec::with_capacity(16),
            threshold: 10,
            max_traces: 16,
            max_trace_len: 256,
        }
    }

    /// Record that a backward jump landed at `target_ip`.
    pub fn record_jump(&mut self, target_ip: usize) {
        for (ip, count) in self.counters.iter_mut() {
            if *ip == target_ip {
                *count += 1;
                return;
            }
        }
        if self.counters.len() < 32 {
            self.counters.push((target_ip, 1));
        }
    }

    /// Check whether `target_ip` has crossed the hotness threshold.
    pub fn is_hot(&self, target_ip: usize) -> bool {
        self.counters
            .iter()
            .any(|(ip, count)| *ip == target_ip && *count >= self.threshold)
    }

    /// Retrieve a compiled trace starting at `ip`.
    pub fn get_trace(&self, ip: usize) -> Option<&Trace> {
        self.traces
            .iter()
            .find(|(start, _)| *start == ip)
            .map(|(_, t)| t)
    }

    /// Compile a trace from `start_ip` (inclusive) to `end_ip` (exclusive).
    pub fn compile_trace(&mut self, code: &[OpCode], start_ip: usize, end_ip: usize) {
        if self.traces.len() >= self.max_traces {
            return;
        }
        let len = end_ip.saturating_sub(start_ip);
        if len == 0 || len > self.max_trace_len {
            return;
        }
        let ops = code[start_ip..end_ip].to_vec();
        self.traces.push((start_ip, Trace { start_ip, ops }));
    }
}
