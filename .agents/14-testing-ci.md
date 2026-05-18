# 14 — Testing & CI: Integration Tests, Fuzzing, Benchmark Gates

## Goal

Expand test coverage from unit tests to integration tests, add fuzzing harnesses for all security-sensitive code, strengthen CI gates, and add benchmark regression detection.

## Current State

CI (`.github/workflows/ci.yml`, 369 lines) — 16 jobs:
- `fmt`, `clippy`, `test`, `miri`, `bench-regression`, `audit`, `deny`, `docs`, `bom`, `python`, `kernel-build`, `kernel-image`, `kernel-qemu-smoke`, `kernel-clippy`, `lean`
- QEMU smoke test checks 20 boot milestones via serial output
- Kernel clippy suppresses `unused`, `dead_code`, `approx_constant` (overly broad)
- `deny.toml`: 7 suppressed advisories

Existing tests:
- `aether-core`: Voronoi, governor, spectral, geodesic tests
- `epsilon`: 17 tests (teleport, bridge, governor)
- `aether-lang`: 3 lexer tests
- `aether-link`: algorithm tests
- Lean 4: 10 formal proofs

## Implementation Steps

### Phase A: Integration Tests

1. **ManifoldFS integration tests**
   - Create 100 files → verify listing → teleport 50 → verify topology preserved
   - Concurrent access patterns (simulate multi-process with threads in test harness)
   - Edge cases: max filename length, deeply nested directories, empty files, large files (1MB+)

2. **Aether-Lang end-to-end tests**
   - Parse → interpret → verify output for 20+ programs
   - Parse → compile → VM execute → verify output matches interpreter
   - Programs: fibonacci, sort, manifold operations, topology queries

3. **Epsilon teleport integration**
   - Bridge → teleport → verify point cloud identical
   - Chain: teleport A→B→C→A → verify round-trip topology preservation
   - Stress: 1000 sequential teleports → verify no state corruption

4. **Boot integration test (QEMU)**
   - Extend smoke test from 20 to 40 milestones
   - Add: file operations via shell, app window rendering, event loop responsiveness
   - Timeout: 60 seconds (up from current)

### Phase B: Fuzzing

5. **Create fuzzing targets** (`cargo-fuzz`)
   - `fuzz_manifold_ops`: random create/read/write/delete/teleport sequences
   - `fuzz_aether_lexer`: random byte strings → lexer should never panic
   - `fuzz_aether_parser`: random token sequences → parser should return error, not panic
   - `fuzz_epsilon_bridge`: random f64 arrays → bridge should handle gracefully
   - `fuzz_teleport_payload`: random bytes → deserialize should return error, not panic

6. **Add fuzzing to CI**
   - Run each fuzz target for 60 seconds per CI run
   - Store corpus in repo for reproducibility
   - Crash = CI failure

### Phase C: Benchmark Regression

7. **Strengthen benchmark gates**
   - ManifoldFS: `teleport` latency (must be <100μs)
   - Aether-Link: `decision_cycle` latency (must be <50ns)
   - Epsilon bridge: `project` latency (must be <1ms for 768-dim input)
   - Governor: `update_step` latency (must be <1μs)

8. **Add new benchmarks**
   - ManifoldFS: `create_file`, `read_file`, `similarity_search`
   - Aether-Lang: `parse_program`, `interpret_program`, `vm_execute`
   - Memory allocator: `alloc_dealloc_cycle` (after plan 02)

### Phase D: CI Improvements

9. **Narrow clippy suppressions**
   - Remove blanket `unused` and `dead_code` suppression from kernel clippy
   - Add targeted `#[allow(dead_code)]` only on specific items that are intentionally unused (future API surface)

10. **Review deny.toml suppressions**
    - Check if `RUSTSEC-2025-0020` (pyo3) affects any enabled features
    - Check if webpki advisories have upstream fixes available
    - Remove suppressions for fixed advisories

11. **Add QEMU timeout to CI**
    - Currently smoke test may hang on deadlock — add 120-second hard timeout
    - On timeout: capture serial log and fail with "boot deadlock detected"

12. **Coverage reporting**
    - Add `cargo-llvm-cov` for workspace crates
    - Report coverage percentage in CI summary
    - Gate: coverage must not decrease on PR

## Dependencies

- All other plans should complete first (tests verify completed work)
- Fuzzing depends on: 01 (no panic discipline), 04/05 (interpreter/VM robustness)

## Acceptance Criteria

- [ ] ≥50 integration tests across all subsystems
- [ ] 5 fuzzing targets running in CI
- [ ] Benchmark regression gate catches ≥10% slowdown
- [ ] No blanket clippy suppressions in kernel build
- [ ] QEMU smoke test covers 40+ milestones
- [ ] Coverage report generated on each CI run
- [ ] All CI jobs pass on clean main branch

## Files to Create

- `kernel/seal-os/tests/` (integration test directory)
- `kernel/epsilon/epsilon/fuzz/` (fuzz targets)
- `kernel/aether/Aether-Lang/crates/aether-lang/fuzz/` (fuzz targets)
- `kernel/seal-os/benches/` (new benchmarks)

## Files to Modify

- `.github/workflows/ci.yml` (new jobs, stronger gates)
- `deny.toml` (review suppressions)
