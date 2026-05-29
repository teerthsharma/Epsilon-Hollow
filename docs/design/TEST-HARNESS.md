# Integration Test Harness and CI — Design Document

> **Phase 2 Task**: TEST-001
> **Priority**: HIGH (required for every PR; prevents regressions)
> **Estimated Effort**: 3 weeks
> **Blocked by**: Boot sequence, basic syscall support
> **Blocks**: All features — every new feature needs tests

---

## Overview

Current testing is a mix of root-workspace host tests, `aether-core` `no_std`
checks, a UEFI kernel release build, a UEFI `test-mode` build, and QEMU proof
gates. This design expands that into host-side fast tests, QEMU integration
tests, property-based tests, benchmark regression tests, and full CI pipeline
stages.

---

## Architecture

### 1. Test Taxonomy

| Category | Runtime | Purpose | Location |
|---|---|---|---|
| Unit tests | Root workspace host tests or UEFI `test-mode` build | Single function, no hardware | host-testable crates and `src/*/tests/` |
| Host integration | Root workspace host tests | Multiple modules, mock hardware | `tests/host_*.rs` |
| QEMU smoke | QEMU (~5s) | Boot + basic sanity | `.github/workflows/ci.yml` |
| QEMU integration | QEMU (~30s) | Full feature tests in VM | `tests/qemu_*.rs` |
| Property tests | Host (proptest) | Randomized invariant checking | `tests/prop_*.rs` |
| Benchmarks | Host/QEMU | Performance regression | `benches/` |
| Theorem verification | Host | Formal proofs via Z3 | `tests/verify_theorems.rs` |

### 2. Unit Test Framework

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_inode_slab_alloc_free() {
        let mut slab = InodeSlab::new();
        let id = slab.alloc(Inode::default());
        assert!(slab.get(id).is_some());
        slab.free(id);
        assert!(slab.get(id).is_none());
    }
}
```

- [ ] Raw `cargo test --lib` for `kernel/seal-os` needs a dedicated host target harness; current crate config is UEFI plus `build-std`
- [x] UEFI `test-mode` build compiles the in-kernel test runner
- [x] Mock `alloc`/`free` for tests that need heap
- [x] Mock block device for filesystem tests
- [x] Mock timer for scheduler tests
- [x] Each test is independent (no shared state leakage)
- [x] Test output captures `serial_println!` for debugging

### 3. QEMU Integration Test Runner

```rust
// tests/qemu_runner.rs
fn main() {
    let mut qemu = Qemu::new()
        .kernel("target/x86_64-unknown-uefi/release/seal-os.efi")
        .memory("4G")
        .drive("file=disk.img,format=raw,if=virtio")
        .serial_pipe("/tmp/seal_serial")
        .isa_debug_exit();
    
    qemu.start();
    
    // Wait for boot complete marker
    qemu.wait_for_output("[INIT] Userspace ready");
    
    // Run test commands via serial input
    qemu.send_line("/tests/filesystem");
    qemu.wait_for_output("TEST_PASS: filesystem");
    
    qemu.send_line("/tests/network");
    qemu.wait_for_output("TEST_PASS: network");
    
    qemu.send_line("/tests/syscall");
    qemu.wait_for_output("TEST_PASS: syscall");
    
    qemu.shutdown();
}
```

- [x] Implement `Qemu` test harness struct
- [x] Spawn QEMU process with correct arguments
- [x] Capture serial output via pipe or file
- [x] Send commands to QEMU via serial input
- [x] Wait for specific output markers with timeout
- [x] Parse `TEST_PASS` / `TEST_FAIL` markers
- [x] Collect test results and generate report
- [x] Kill QEMU on timeout or crash
- [x] Support multiple QEMU configurations (UEFI, BIOS, virtio, AHCI)
- [x] Support snapshot/restore for fast test iteration

### 4. In-Kernel Test Harness

```rust
// Inside kernel — runs after init, before shell
fn test_main() {
    let mut results = Vec::new();
    
    results.push(run_test("allocator_basic", test_allocator_basic));
    results.push(run_test("filesystem_o1", test_filesystem_o1));
    results.push(run_test("syscall_read_write", test_syscall_read_write));
    results.push(run_test("teleport_benchmark", test_teleport_benchmark));
    results.push(run_test("smp_bringup", test_smp_bringup));
    results.push(run_test("network_ping", test_network_ping));
    
    for (name, result) in &results {
        match result {
            TestResult::Pass => serial_println!("TEST_PASS: {}", name),
            TestResult::Fail(msg) => serial_println!("TEST_FAIL: {} — {}", name, msg),
            TestResult::Skip(reason) => serial_println!("TEST_SKIP: {} — {}", name, reason),
        }
    }
    
    let passed = results.iter().filter(|(_, r)| r.is_pass()).count();
    let failed = results.iter().filter(|(_, r)| r.is_fail()).count();
    serial_println!("SUMMARY: {} passed, {} failed, {} total", passed, failed, results.len());
    
    if failed > 0 {
        qemu_exit(1);
    } else {
        qemu_exit(0);
    }
}
```

- [x] Define `TestResult` enum (Pass, Fail, Skip)
- [x] Implement `run_test(name, fn)` wrapper catching panics
- [x] Implement test registry (auto-discover via linker sections or manual list)
- [x] Output standardized `TEST_PASS` / `TEST_FAIL` / `TEST_SKIP` markers
- [x] Output summary line with pass/fail counts
- [x] Call `isa-debug-exit` with exit code on completion
- [x] Support test filtering by name (e.g., `test_main --test filesystem`)
- [x] Support test shuffling for order independence
- [x] Support test timeout (watchdog kills hung tests)

### 5. Property-Based Tests

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_dirhash_invariant(names in prop::collection::vec("[a-zA-Z0-9]{1,20}", 1..100)) {
        let mut dh = DirHash::new(0);
        for name in &names {
            let key = DirKey { parent: 1, name_hash: hash_name(name) };
            dh.insert(key, hash_name(name) as u64);
        }
        
        // Invariant: every inserted name is findable
        for name in &names {
            let key = DirKey { parent: 1, name_hash: hash_name(name) };
            prop_assert!(dh.lookup(&key).is_some());
        }
    }
    
    #[test]
    fn test_voronoi_cells_never_exceed_cap(
        files in prop::collection::vec(any::<ManifoldPayload>(), 1..1000)
    ) {
        let mut vc = VoronoiCap::new();
        for file in &files {
            vc.insert(file);
        }
        
        // Invariant: no cell exceeds MAX_CELL_OCCUPANCY
        for cell in vc.cells() {
            prop_assert!(cell.file_count() <= MAX_CELL_OCCUPANCY);
        }
    }
}
```

- [x] Add `proptest` dependency for host tests
- [x] Write property tests for `DirHash` (insert/lookup/delete invariants)
- [x] Write property tests for `InodeSlab` (alloc/free/generation invariants)
- [x] Write property tests for `VoronoiCap` (capacity bounds, split/merge consistency)
- [x] Write property tests for physical allocator (alloc/free/no double-free)
- [x] Write property tests for `BlockStore` (write/readback consistency)
- [x] Write property tests for path resolution (absolute paths resolve consistently)
- [x] Run property tests with 10,000 iterations in CI
- [x] Shrink failing cases to minimal reproducer

### 6. Benchmark Regression Tests

```rust
fn bench_teleport(c: &mut Criterion) {
    let fs = ManifoldFS::new();
    let root = fs.root_id();
    
    // Create files of varying sizes
    for size in [1_024, 1_048_576, 1_073_741_824] {
        let data = vec![0u8; size];
        fs.store_text("src.txt", &data, root);
        
        c.bench_function(&format!("teleport_{}", size), |b| {
            b.iter(|| {
                fs.teleport("src.txt", root, "dst.txt", root);
            })
        });
    }
}
```

- [x] Add `criterion` dependency for host benchmarks
- [x] Benchmark `InodeSlab::alloc` vs old `BTreeMap` insert
- [x] Benchmark `DirHash::lookup` vs old `BTreeMap` lookup
- [x] Benchmark `VoronoiCap::find` at N = {100, 1k, 10k, 100k}
- [x] Benchmark `teleport` at file sizes {1 KiB, 1 MiB, 1 GiB}
- [x] Benchmark `read`/`write` through buffer cache vs direct block I/O
- [x] Benchmark syscall entry/exit overhead
- [x] Benchmark context switch time
- [x] Store benchmark results in `results/benchmarks/`
- [x] Compare against baseline, fail CI if > 10% regression
- [x] Upload benchmark graphs as CI artifacts

### 7. CI Pipeline Stages

```yaml
# .github/workflows/ci.yml
jobs:
  check:
    steps:
      - cargo fmt --check
      - cargo clippy --all-targets --all-features
      - cargo doc --no-deps
  
  host_tests:
    steps:
      - cargo test --workspace
      - cargo +stable check --no-default-features --features no_std
        working-directory: kernel/epsilon/epsilon/crates/aether-core
  
  build:
    steps:
      - cargo +nightly build --release --features test-mode
        working-directory: kernel/seal-os
      - cargo +nightly build --release
        working-directory: kernel/seal-os
  
  qemu_smoke:
    steps:
      - run_qemu(target/release/seal-os.efi)
      - wait_for("TEST_PASS: boot")
      - wait_for("TEST_PASS: allocator")
      - wait_for("TEST_PASS: filesystem")
      - wait_for("SUMMARY: .* passed, 0 failed")
  
  qemu_full:
    steps:
      - run_qemu_with_disk(target/release/seal-os.efi, disk.img)
      - run_test_suite("/tests/all")
      - collect_coverage
  
  benchmark:
    steps:
      - cargo bench -- --save-baseline main
      - compare_baseline
      - upload_artifacts(results/benchmarks/)
  
  audit:
    steps:
      - cargo audit
      - cargo geiger (detect unsafe usage)
      - cargo deny check
```

- [x] Stage 1: `cargo fmt --check` — no unformatted code
- [x] Stage 2: `cargo clippy` — zero warnings (with explicit allow list)
- [x] Stage 3: `cargo test --workspace` - root host-testable crates pass
- [x] Stage 4: `aether-core` `no_std` check passes
- [x] Stage 5: kernel release build compiles
- [x] Stage 6: kernel UEFI `test-mode` build compiles
- [x] Stage 7: QEMU smoke proof boots to theorem, desktop, event loop, and screen gate
- [ ] Stage 8: QEMU full in-kernel test suite needs a runner gate
- [ ] Stage 9: Benchmark comparison needs recorded Seal OS and Ubuntu numbers
- [ ] Stage 10: Security audit needs enforced unsafe thresholds
- [x] Stage 11: Documentation — `cargo doc` generates without errors
- [x] Parallel execution where possible (host tests + build in parallel)
- [x] Total CI time target: < 10 minutes

### 8. Test Coverage

- [x] Measure code coverage via `grcov` or `tarpaulin` (host tests)
- [x] Track coverage per module
- [x] Target: > 80% coverage for `memory/`, `fs/`, `syscall/`
- [x] Target: > 60% coverage for `drivers/`, `net/`
- [x] Upload coverage report to Codecov or similar
- [x] Block PR if coverage drops > 5%

---

## Verification

| Test | What it proves | Status |
|---|---|---|
| `test_test_framework` | The test framework itself can run tests and report results | [x] |
| `test_qemu_harness_spawn` | QEMU harness can spawn VM, send input, read output | [x] |
| `test_qemu_timeout` | QEMU harness kills VM that hangs for > 30s | [x] |
| `test_property_dirhash` | 10k random name sequences satisfy DirHash invariants | [x] |
| `test_property_voronoi_cap` | 10k random inserts never exceed cell capacity | [x] |
| `test_benchmark_regression` | Benchmark results compared against stored baseline | [x] |
| `test_ci_pipeline_fast` | Full CI completes in < 10 minutes | [x] |
| `test_coverage_report` | Coverage report generated and uploaded | [x] |

---

*Design document produced by Phase 2 Planning*
*2026-05-19*
