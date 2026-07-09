# API Reference

This page summarizes public surfaces that are visible from source. The Rust
source remains authoritative.

## Language Front End

| Surface | Source | Purpose |
| --- | --- | --- |
| `Lexer` | `crates/aether-lang/src/lexer.rs` | Token stream construction |
| `Parser` | `crates/aether-lang/src/parser.rs` | AST construction and recoverable parsing |
| `Program`, `StmtKind`, `ExprKind` | `crates/aether-lang/src/ast.rs` | Syntax tree representation |
| `typecheck_program` | `crates/aether-lang/src/typecheck.rs` | Type inference/checking entrypoint |

## Runtime

| Surface | Source | Purpose |
| --- | --- | --- |
| `Interpreter` | `crates/aether-lang/src/interpreter.rs` | Tree-walking execution |
| `Value` | `crates/aether-lang/src/interpreter.rs` | Runtime value representation |
| `ManifoldWorkspace` | `crates/aether-lang/src/interpreter.rs` | Embedded point workspace |
| `EscalatingRegressor` | `crates/aether-lang/src/interpreter.rs` | Regression workflow helper |
| `FsCallbacks`, `ProcessCallbacks`, `NetCallbacks`, `HwCallbacks` | `crates/aether-lang/src/interpreter.rs` | Host/runtime boundary |

## VM

| Surface | Source | Purpose |
| --- | --- | --- |
| `OpCode` | `crates/aether-lang/src/bytecode.rs` | Bytecode instruction enum |
| `BytecodeVerifier` | `crates/aether-lang/src/bytecode.rs` | Bytecode validation |
| `Peephole` | `crates/aether-lang/src/bytecode.rs` | Local bytecode compaction |
| `TitanVM` | `crates/aether-lang/src/vm.rs` | Bytecode execution prototype |
| `Compiler` | `crates/aether-lang/src/vm.rs` | AST-to-bytecode prototype |

## Kernel-Adjacent Surfaces

| Surface | Source | Purpose |
| --- | --- | --- |
| `SparseScheduler` | `crates/aether-kernel/src/scheduler.rs` | Deviation-gated scheduling model |
| `HardwareTopology` | `crates/aether-kernel/src/boot/topology.rs` | Boot topology summary |
| `verify_elf` | `crates/aether-kernel/src/loader.rs` | ELF header validation |
| `verify_binary_topology` | `crates/aether-kernel/src/loader.rs` | Binary topology predicate |

## Stability

The current API should be treated as research-stage. Pages should cite concrete
types and commands, but avoid promising long-term compatibility until the crate
declares a stable public API policy.
