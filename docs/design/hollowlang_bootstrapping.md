# HollowLang Compiler Bootstrap Design Document

> **Phase X Task**: CX-001
> **Priority**: CRITICAL (defines project identity)
> **Estimated Effort**: 12 months

---

## Overview

HollowLang is a systems programming language designed specifically for geometric/topological computing. It must replace Rust as the implementation language for Epsilon-Hollow kernel and userland.

## Design Principles

1. **Memory Safety via Linear Types**: Every resource (memory, file handle, IRQ) has a linear type. Use = move. No implicit copies.
2. **Geometric Primitives Native**: `SpherePoint`, `Manifold`, `VoronoiCell` are first-class types with compiler-known semantics.
3. **No Undefined Behavior**: The type system eliminates null pointers, use-after-free, and data races at compile time.
4. **Self-Hosting**: The compiler must be written in HollowLang.

## Bootstrap Path

### Stage 0: Specification (Month 1-2)
- Formal grammar (BNF)
- Type system rules (subtyping, variance)
- Memory model (linear/affine types, ownership)
- ABI specification (x86_64 System V compatible)

### Stage 1: Rust-Hosted Compiler (Month 3-6)
- Lexer: regex-free, hand-written for determinism
- Parser: recursive descent, zero ambiguity
- Type checker: Hindley-Milner + linear types
- LLVM IR codegen: via `llvm-sys` crate
- Standard library: `core` only (no allocator at first)

### Stage 2: HollowLang-to-Rust Transpiler (Month 7)
- Rapid prototyping: HollowLang syntax -> Rust code
- Allows testing language features without waiting for full codegen

### Stage 3: Self-Hosting (Month 8-10)
- Rewrite compiler in HollowLang
- Compile with Stage 1 compiler
- Recompile with output binary (Turing Tarpit test)
- Fix bootstrap differences

### Stage 4: Kernel Migration (Month 11-12)
- Translate `src/memory/` modules first (well-contained)
- Translate `src/fs/manifold_fs.rs` (the crown jewel)
- Translate `src/process/scheduler.rs`
- Keep assembly blocks (context switch) as `unsafe asm` blocks

## Verification

- **Type Safety Proof**: Coq proof that well-typed HollowLang programs cannot have UAF/data races.
- **Bootstrap Equality**: `compiler_v1(binary) == compiler_v2(binary)` bit-for-bit.
- **Kernel Regression**: All existing tests must pass after migration.

## Risks

| Risk | Mitigation |
|------|------------|
| LLVM dependency too heavy | Maintain own minimal x86_64 backend as fallback |
| Self-hosting takes too long | Keep Rust compiler as bootstrap path indefinitely |
| Type system too restrictive | Add `unsafe` escape hatch with audit requirements |

---

*Design document produced by Phase X Planning*
*2026-05-18*
