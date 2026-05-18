# 05 — Aether-Lang Titan VM: Complete Bytecode Opcodes

## Goal

Complete the Titan VM so every Aether-Lang construct compiles to working bytecode and executes correctly — no silent drops, no no-op opcodes, no `_ => {}` catch-alls.

## Current State

`kernel/aether/Aether-Lang/crates/aether-lang/src/vm.rs` (~563 lines):

### Working opcodes
- `PUSH` — push constant onto stack
- `LOAD` / `STORE` — variable access by slot index
- `ADD` / `SUB` / `MUL` / `DIV` — arithmetic
- `JMP` / `JZ` — unconditional and conditional jump
- `HALT` — stop execution
- `PRINT` — debug output

### Broken / incomplete opcodes
- **`EMBED`** — supposed to embed a value into the manifold. Currently: `let _val = self.stack.pop()` — value is popped and discarded. No-op.
- **`PRUNE`** — supposed to prune low-weight connections. Currently: `let _threshold = self.stack.pop()` — popped and discarded. Closure `f` is empty.
- **`ATTEND`** — not implemented at all, not even defined as an opcode constant

### Compiler gaps (`compile_stmt`)
- `_ => {}` catch-all silently drops any statement type the compiler doesn't handle
- `Lt` (less-than) comparison uses a `SUB` hack — works for some cases but not for boolean comparisons or NaN
- No compilation for: `For`, `Fn`, `Return`, `Break`, `Continue`, `Class`, `Import`, `Render`
- No function call compilation

## Implementation Steps

1. **Remove `_ => {}` catch-all in `compile_stmt`**
   - Replace with explicit `panic!("unimplemented: {:?}", stmt)` or `Err(CompileError::Unimplemented(...))`
   - This surfaces which statements are actually reached during compilation

2. **Implement `EMBED` opcode**
   - Pop value from stack
   - Pop manifold reference from stack (or use a register)
   - Insert value as a new point in the manifold's point cloud on S²
   - Use JL projection (from epsilon bridge) to convert the value's embedding to spherical coordinates
   - Update the manifold's Voronoi index

3. **Implement `PRUNE` opcode**
   - Pop threshold from stack
   - Pop manifold reference from stack
   - Remove all points with weight below threshold
   - Recompute Voronoi index
   - Push count of pruned points onto stack

4. **Define and implement `ATTEND` opcode**
   - Add `ATTEND` constant (e.g., `pub const ATTEND: u8 = 13`)
   - Pop query point from stack
   - Pop manifold from stack
   - Compute attention weights: softmax of negative distances from query to all manifold points
   - Push weighted centroid (attended value) onto stack

5. **Implement comparison opcodes properly**
   - Add `EQ`, `LT`, `GT`, `LE`, `GE`, `NEQ` opcodes
   - Each pops two values, pushes boolean (1.0 or 0.0)
   - Remove the `SUB` hack for `Lt`

6. **Compile `For` loops to bytecode**
   - Emit: evaluate iterable → STORE iterator_slot
   - Loop start label
   - Emit: check iterator exhausted → JZ to loop_end
   - Emit: LOAD next element → STORE loop_var_slot
   - Compile body
   - JMP to loop start
   - Loop end label

7. **Compile `Fn` definitions and calls**
   - Function body compiled to separate bytecode chunk
   - `CALL` opcode: push return address, jump to function bytecode
   - `RET` opcode: pop return address, jump back, push return value

8. **Compile `Return` / `Break` / `Continue`**
   - `Return`: evaluate expression → `RET` opcode
   - `Break`: `JMP` to enclosing loop's end label (track loop labels in compiler state)
   - `Continue`: `JMP` to enclosing loop's start label

9. **Add `CALL` and `RET` opcodes**
   - `CALL`: push IP+1 onto call stack, set IP to function start
   - `RET`: pop from call stack, set IP, push return value

10. **Add VM tests**
    - Compile and run: `let x = 5; let y = x + 3;` → verify y == 8
    - Compile and run: `for i in range(0, 5) { embed(m, i); }` → verify 5 points embedded
    - Compile and run: `fn double(x) { return x * 2; }; double(21)` → verify 42
    - EMBED actually modifies manifold state
    - PRUNE actually removes points

## Dependencies

- **04-aether-lang-interpreter** (share the control flow error model for Break/Continue/Return)

## Acceptance Criteria

- [ ] No `_ => {}` catch-all in compiler
- [ ] `EMBED` modifies manifold state (not a no-op)
- [ ] `PRUNE` removes points below threshold
- [ ] `ATTEND` computes weighted attention over manifold points
- [ ] Comparison opcodes produce correct boolean results
- [ ] `For`, `Fn`, `Return`, `Break`, `Continue` all compile and execute
- [ ] `CALL` / `RET` opcodes work for function calls
- [ ] `cargo test -p aether-lang` passes with new VM tests

## Files to Modify

- `kernel/aether/Aether-Lang/crates/aether-lang/src/vm.rs` (major rewrite)

## Files to Reference

- `kernel/epsilon/epsilon/crates/epsilon/src/bridge.rs` (JL projection for EMBED)
- `kernel/aether/Aether-Lang/crates/aether-lang/src/ast.rs` (statement types)
- `kernel/aether/Aether-Lang/crates/aether-lang/src/parser.rs` (what constructs exist)
