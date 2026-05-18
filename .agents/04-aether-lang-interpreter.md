# 04 — Aether-Lang Interpreter: Implement All Stubs

## Goal

Complete the Bio-mode tree-walking interpreter so that every Aether-Lang statement actually executes — no more `Ok(Value::Unit)` stubs for control flow, functions, or rendering.

## Current State

`kernel/aether/Aether-Lang/crates/aether-lang/src/interpreter.rs` (~1348 lines):

### Working
- Variable declaration (`let x = 5`)
- Arithmetic expressions, comparisons, boolean logic
- `if` / `while` statements
- Module system: `math`, `topology`, `ml`, `Seal` native functions
- ML constructors: MLP, KMeans, Conv2D, tensor ops
- `manifold_decl`, `block_decl` creation
- Print / debug output

### Stubbed (return `Ok(Value::Unit)` — silently do nothing)
- **`For` loops** — parsed correctly but execute nothing
- **`Fn` (function definitions)** — parsed but never stored; cannot define user functions
- **`Return`** — parsed but returns Unit, doesn't unwind
- **`Break`** — parsed but doesn't break loops
- **`Continue`** — parsed but doesn't skip to next iteration
- **`execute_render`** — returns Unit, never renders anything
- **`TopoBetti`** — hardcoded to always return `[1.0, 0.0]`

### AST types (from `ast.rs` / `parser.rs`)
- Parser correctly produces AST nodes for all these constructs
- The gap is purely in `interpreter.rs` execution

## Implementation Steps

1. **Implement `For` loop execution**
   - Parse already produces `For { var, iterable, body }`
   - `iterable` evaluates to a `Value::List` or `Value::Range`
   - For each element: bind `var` in a new scope, execute `body`
   - Respect `Break` and `Continue` signals

2. **Implement `Fn` (function definitions)**
   - Store function in the environment: `env.insert(name, Value::Function { params, body, closure_env })`
   - Capture the current environment for closures (or use lexical scoping — simpler)
   - On call: create new scope with params bound to arguments, execute body

3. **Implement `Return` statement**
   - Use a special error variant `Err(InterpreterError::Return(value))` to unwind
   - Function call catches `Return` and extracts the value
   - If `Return` is hit outside a function, report error

4. **Implement `Break` / `Continue`**
   - Similar to Return: `Err(InterpreterError::Break)` and `Err(InterpreterError::Continue)`
   - `While` and `For` loops catch these:
     - `Break` → exit loop, return Unit
     - `Continue` → skip rest of body, continue iteration

5. **Implement `execute_render`**
   - Evaluate the render expression to a `Value::Manifold` or `Value::Block`
   - If running in graphical mode: convert point cloud to pixel coordinates, draw to framebuffer
   - If running in serial/test mode: output ASCII representation via `ascii_render.rs`
   - The `ascii_render.rs` module already exists — wire it up

6. **Fix `TopoBetti` computation**
   - Currently hardcoded: `[1.0, 0.0]`
   - Implement real Betti number computation for point clouds:
     - Build Vietoris-Rips complex from point cloud
     - Compute β₀ (connected components) and β₁ (holes) via boundary matrix reduction
     - Or use the simpler approach: β₀ = number of connected components in ε-neighborhood graph

7. **Add function call support for user-defined functions**
   - When evaluating a call expression: check if callee is `Value::Function`
   - Bind arguments to parameter names
   - Execute body in new scope
   - Catch `Return` to get return value

8. **Add tests**
   - `for i in range(0, 5) { ... }` — verify loop executes 5 times
   - `fn add(a, b) { return a + b; }; add(2, 3)` — verify returns 5
   - `for i in range(0, 10) { if i == 5 { break; } }` — verify breaks at 5
   - Nested functions and closures
   - Render a simple manifold in ASCII mode

## Dependencies

- None (standalone crate, `std` tests)

## Acceptance Criteria

- [ ] `for` loops execute their bodies
- [ ] User-defined functions work with parameters and return values
- [ ] `break` exits loops, `continue` skips iterations
- [ ] `return` unwinds from functions correctly
- [ ] `render` produces output (ASCII or graphical)
- [ ] `TopoBetti` returns non-hardcoded values based on actual point cloud topology
- [ ] All existing tests still pass
- [ ] New tests cover every previously-stubbed construct
- [ ] `cargo test -p aether-lang` passes

## Files to Modify

- `kernel/aether/Aether-Lang/crates/aether-lang/src/interpreter.rs` (primary)
- `kernel/aether/Aether-Lang/crates/aether-lang/src/ast.rs` (may need error variants)

## Files to Reference

- `kernel/aether/Aether-Lang/crates/aether-lang/src/parser.rs` (AST structure)
- `kernel/aether/Aether-Lang/crates/aether-lang/src/ascii_render.rs` (rendering backend)
