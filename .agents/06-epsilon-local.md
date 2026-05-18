# 06 — Epsilon Local Teleportation: Harden, Edge Cases, Fuzzing

## Goal

Harden the local void teleportation pipeline to be production-grade: handle all edge cases, add fuzzing, improve error reporting, and validate the mathematical invariants at each step.

## Current State

`kernel/epsilon/epsilon/crates/epsilon/src/teleport.rs` (~395 lines):
- **Working**: 4-step protocol (clutch → inject → assimilate → restore)
- **Working**: `SurgeryPermit` one-shot token (non-Clone, non-Copy)
- **Working**: Governor clutch (zeroes derivative gain during surgery)
- **Working**: 4 unit tests (success, topology rejection, void busy, remote unimplemented)
- `RemoteVoid` explicitly returns `RemoteUnimplemented` (planned for plan 07)

`kernel/epsilon/epsilon/crates/epsilon/src/governor.rs` (~299 lines):
- **Working**: PD controller with bounded ε ∈ [0.001, 10.0]
- **Working**: Surgery permit issuance and consumption
- 5 unit tests

`kernel/epsilon/epsilon/crates/epsilon/src/bridge.rs` (~686 lines):
- **Working**: JL projection ℝ^E → S² via Xoshiro256** + Box-Muller + L2 normalize
- **Working**: `EmbeddingBridge` with graph building, payload construction, retry on disconnect
- 8 unit tests including cross-agent determinism, semantic proximity

## Gap Analysis

| Issue | Severity |
|-------|----------|
| No validation of point cloud dimensionality consistency | Medium |
| No handling of degenerate inputs (zero vectors, all-same points) | Medium |
| No concurrent teleport protection (two teleports racing) | High |
| Governor ε bounds not tested at extremes | Low |
| No fuzzing harness | Medium |
| Bridge retry limit is hardcoded, no backoff | Low |
| No telemetry / metrics on teleport success rate | Low |

## Implementation Steps

1. **Add input validation to teleport pipeline**
   - Validate point cloud has exactly 64 points (or configurable N)
   - Validate no NaN/Inf in coordinates
   - Validate point cloud lies on S² (|p| ≈ 1.0 within tolerance)
   - Return `TeleportError::InvalidPointCloud` on failure

2. **Handle degenerate inputs in bridge**
   - Zero embedding vector → return error (can't normalize to S²)
   - All-identical embeddings → return error (degenerate Voronoi → no topology)
   - Single-point payload → skip graph building, inject directly

3. **Add concurrent teleport guard**
   - Use `AtomicBool` flag on the void: `is_teleporting`
   - `try_lock` at start of teleport, release at end
   - Return `TeleportError::VoidBusy` on contention (already exists, but enforce it atomically)

4. **Extreme ε testing**
   - Test governor at ε = 0.001 (minimum bound): verify permit still issues
   - Test governor at ε = 10.0 (maximum bound): verify clamping works
   - Test rapid permit-issue-consume cycles (100 iterations)
   - Test derivative gain zeroing: verify β stays 0 during surgery, restores after

5. **Create fuzzing harness**
   - `cargo-fuzz` target: random embedding vectors → bridge → teleport
   - Fuzz inputs: random f64 arrays (including NaN, Inf, subnormals, ±0)
   - Fuzz point cloud sizes: 0, 1, 63, 64, 65, 1000
   - Assert: no panic, no UB, teleport either succeeds or returns typed error

6. **Add backoff to bridge retry**
   - Current: retry up to N times on disconnected graph
   - Add: exponential backoff with jitter on retry delay
   - Add: configurable max retries via `BridgeConfig`

7. **Add teleport metrics**
   - `TeleportMetrics { attempts: u64, successes: u64, topology_rejects: u64, void_busy: u64, avg_latency_ns: u64 }`
   - Updated atomically on each teleport attempt
   - Exposed via `pub fn metrics() -> TeleportMetrics`

8. **Property-based tests**
   - Teleport preserves topology: Betti numbers of point cloud unchanged
   - Teleport is idempotent: teleport(teleport(x)) == teleport(x) (for same void)
   - Bridge determinism: same seed → same projection matrix → same S² coordinates
   - Governor monotonicity: if error decreases, ε should converge

## Dependencies

- **01-kernel-safety** (atomic patterns, no-panic discipline)

## Acceptance Criteria

- [ ] No panic on any degenerate input (NaN, zero vector, empty cloud)
- [ ] Concurrent teleport attempts return `VoidBusy` (not corrupt state)
- [ ] Fuzzing harness runs 100k iterations with zero crashes
- [ ] All existing 17 tests (teleport + governor + bridge) still pass
- [ ] New edge case tests added (≥10 new test functions)
- [ ] `cargo test -p epsilon` passes
- [ ] `cargo +nightly miri test -p epsilon` passes (no UB)

## Files to Modify

- `kernel/epsilon/epsilon/crates/epsilon/src/teleport.rs`
- `kernel/epsilon/epsilon/crates/epsilon/src/bridge.rs`
- `kernel/epsilon/epsilon/crates/epsilon/src/governor.rs`

## Files to Create

- `kernel/epsilon/epsilon/fuzz/` (cargo-fuzz targets)
