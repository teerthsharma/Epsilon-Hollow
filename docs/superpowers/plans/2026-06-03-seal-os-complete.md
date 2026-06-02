# Seal OS Complete Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:dispatching-parallel-agents to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove all known pain points from Seal OS to make it production-complete: self-hosting Aether-Lang, proven GPU compute, production security, browser engine, Ubuntu benchmark artifacts, and elimination of legacy host wrappers.

**Architecture:** 28 parallel work streams across 7 macro domains (Language, Benchmarks, Security, GPU, Browser, Build System, Kernel). Each stream owns specific files and produces independently testable artifacts.

**Tech Stack:** Rust nightly (kernel), Rust stable (workspace), Aether-Lang, Lean 4 (proofs), OpenCL C (GPU shaders), x86_64 bare-metal.

---

## Master Pain Point Registry (from README Brutal Honesty)

1. **Aether-Lang cannot self-host** — needs bootstrap compiler, richer stdlib, build system integration
2. **Not faster than Linux** — Ubuntu artifact missing, side-by-side harness incomplete
3. **Not production-secure** — KPTI boot selftest pending, no X.509/TLS, no security audit
4. **GPU not proven hardware compute** — shader stubs need real GCN ISA, hardware dispatch proof missing
5. **No browser** — no HTML parser, DOM, CSS, JS, or renderer
6. **Not self-hosting build** — needs Aether-Lang cross-compiler to break Linux/Windows/macOS dependency
7. **Legacy host wrappers remain** — scripts/ and host Python still outside runtime path
8. **Package manager remote registry pending** — `.eph` remote fetch + signed gate missing
9. **Installer lacks GPT/formatting** — UI exists, no raw block write path
10. **FAT/ext2 fixture gates missing** — write paths exist, no comprehensive parity tests
11. **Kernel modules framework missing** — everything built-in, no LKM ABI
12. **WiFi/Bluetooth simulated** — acceptable as vendor IP, but state machines need real firmware path docs

---

## Domain A: Aether-Lang Self-Hosting (Streams 1-6)

### Stream 1: Aether-Lang Lexer/Parser Hardening
**Agent:** Aether-Lexer-Agent  
**Scope:** `kernel/aether/src/lexer.rs`, `kernel/aether/src/parser.rs`  
**Goal:** Make lexer and parser robust enough to parse a 10,000-line Aether-Lang source file without stack overflow or ambiguity. Add error recovery (continue after syntax error).  
**Independence:** Touches only `kernel/aether/src/lexer.rs` and `parser.rs`. No other agent touches these.

- [ ] **Step 1: Add incremental lexing mode** — `Lexer::new_incremental(source: &str)` that yields tokens via iterator without collecting entire Vec.
- [ ] **Step 2: Add parser error recovery** — `ParseError::recover(self)` skips to next statement boundary (`~` or newline) and continues parsing.
- [ ] **Step 3: Add benchmark test** — Parse 10,000 lines of generated Aether-Lang in < 100ms on host.
- [ ] **Step 4: Commit**

### Stream 2: Aether-Lang Type System & AST
**Agent:** Aether-Type-Agent  
**Scope:** `kernel/aether/src/ast.rs`, `kernel/aether/src/typecheck.rs` (create)  
**Goal:** Add a Hindley-Milner-inspired type system to Aether-Lang so the bootstrap compiler can reason about types.  
**Independence:** New files only. `ast.rs` modifications are additive (new enum variants).

- [ ] **Step 1: Extend AST with type annotations** — `Expr::Let { name: String, ty: Option<Type>, value: Box<Expr> }`
- [ ] **Step 2: Create `kernel/aether/src/typecheck.rs`** — Type environment `Env`, unification algorithm, `typecheck(expr: &Expr, env: &mut Env) -> Result<Type, TypeError>`.
- [ ] **Step 3: Integrate typecheck into interpreter** — Optional `--typecheck` flag in VM; if types fail, refuse to run.
- [ ] **Step 4: Commit**

### Stream 3: Aether-Lang VM (Titan Mode) Optimizations
**Agent:** Aether-VM-Agent  
**Scope:** `kernel/aether/src/vm.rs`, `kernel/aether/src/bytecode.rs`  
**Goal:** Add JIT-style trace recording and peephole optimization to the VM so self-hosted compiler runs fast enough.  
**Independence:** Only touches VM internals.

- [ ] **Step 1: Add bytecode verification pass** — `BytecodeVerifier::verify(&[Op])` checks stack balance and local variable bounds before execution.
- [ ] **Step 2: Add simple peephole optimizer** — `Peephole::run(&mut [Op])` folds `PushConst + PushConst + Add` → `PushConst(folded)`.
- [ ] **Step 3: Add trace cache** — Cache decoded opcode sequences for hot loop bodies (up to 256-op trace).
- [ ] **Step 4: Commit**

### Stream 4: Aether-Lang Stdlib Expansion (Self-Hosting)
**Agent:** Aether-Stdlib-Agent  
**Scope:** `kernel/aether/src/stdlib.rs`, `kernel/aether/lib/` (create directory)  
**Goal:** Expand stdlib with collections, I/O, and string manipulation needed for a compiler.  
**Independence:** New `lib/` directory and additive stdlib changes.

- [ ] **Step 1: Add `List<T>` generic in stdlib** — Backed by slab allocator, supports `push`, `pop`, `get`, `len`.
- [ ] **Step 2: Add `Map<K,V>` generic** — Backed by BTreeMap bridge, supports `insert`, `get`, `remove`, `keys`.
- [ ] **Step 3: Add `StringBuilder`** — Efficient string concatenation for code generation.
- [ ] **Step 4: Add `fs::read_text(path)` and `fs::write_text(path, contents)`** — Host-side file I/O for compiler.
- [ ] **Step 5: Commit**

### Stream 5: Aether-Lang Bootstrap Compiler (in Aether-Lang)
**Agent:** Aether-Bootstrap-Agent  
**Scope:** `kernel/aether/bootstrap/` (create directory), `kernel/aether/src/compiler.rs`  
**Goal:** Write a compiler for Aether-Lang, in Aether-Lang, that compiles Aether-Lang to bytecode.  
**Independence:** New `bootstrap/` directory. `compiler.rs` is a new Rust host driver that runs the bootstrap compiler.

- [ ] **Step 1: Create `kernel/aether/bootstrap/lexer.ae`** — Aether-Lang source implementing a lexer.
- [ ] **Step 2: Create `kernel/aether/bootstrap/parser.ae`** — Recursive descent parser emitting the AST.
- [ ] **Step 3: Create `kernel/aether/bootstrap/codegen.ae`** — Bytecode generator from AST.
- [ ] **Step 4: Create `kernel/aether/src/compiler.rs`** — Rust driver: reads `.ae` files, feeds them to Aether-Lang interpreter running `bootstrap/main.ae`, writes `.aec` bytecode output.
- [ ] **Step 5: Self-host proof test** — `compiler.rs` compiles `bootstrap/*.ae` using the bootstrap compiler; result must match host-compiled bytecode for a test program.
- [ ] **Step 6: Commit**

### Stream 6: Build System Self-Hosting (Aether Cross-Compiler)
**Agent:** Build-Selfhost-Agent  
**Scope:** `kernel/seal-mkimage/src/aether_build.rs` (create), `scripts/build_iso.sh`, `BUILD_SYSTEM.md`  
**Goal:** Prove Seal OS can be built without relying on Linux/macOS/Windows host toolchains, by using Aether-Lang as the build driver.  
**Independence:** New `aether_build.rs` and additive script changes.

- [ ] **Step 1: Create `kernel/seal-mkimage/src/aether_build.rs`** — Rust module that invokes Aether bootstrap compiler to build kernel source files into object images.
- [ ] **Step 2: Create `scripts/build_aether.sh`** — Shell script that uses Aether bootstrap compiler instead of `cargo` for Aether-Lang sources; still uses Rust compiler for Rust sources (intermediate step).
- [ ] **Step 3: Document path to full self-host** in `BUILD_SYSTEM.md`: Phase 1 (Aether builds Aether), Phase 2 (Aether builds Rust subset), Phase 3 (full self-host).
- [ ] **Step 4: Commit**

---

## Domain B: Benchmarks & Ubuntu Artifact (Streams 7-8)

### Stream 7: Side-by-Side Benchmark Harness Completion
**Agent:** Benchmark-Harness-Agent  
**Scope:** `.benchmarks/`, `docs/BENCHMARK_PLAN.md`, `kernel/seal-os/src/bench/`  
**Goal:** Make the benchmark harness produce rigorous, reproducible comparisons against Linux on identical hardware.  
**Independence:** New files in `.benchmarks/` and additive bench code.

- [ ] **Step 1: Create `.benchmarks/harness.py`** — Python harness (acceptable as host-side tool) that boots Seal OS in QEMU and Ubuntu in QEMU side-by-side, runs identical workloads, collects `rdtsc` and wall-clock metrics.
- [ ] **Step 2: Define 5 standard workloads** — `alloc_stress`, `file_walk`, `context_switch`, `syscall_loop`, `tcp_echo`.
- [ ] **Step 3: Add `kernel/seal-os/src/bench/external_harness.rs`** — Serial protocol for harness to inject benchmark commands and read JSON results from Seal OS serial.
- [ ] **Step 4: Commit**

### Stream 8: Ubuntu Artifact Creation
**Agent:** Ubuntu-Artifact-Agent  
**Scope:** `infrastructure/deployment/ubuntu-artifact/` (create), `.github/workflows/ci.yml`  
**Goal:** Build a bootable Ubuntu disk image that runs the same benchmark harness, producing the artifact needed for honest comparison claims.  
**Independence:** New directory, CI workflow additive.

- [ ] **Step 1: Create `infrastructure/deployment/ubuntu-artifact/Dockerfile`** — Ubuntu 24.04 with benchmark dependencies.
- [ ] **Step 2: Create `infrastructure/deployment/ubuntu-artifact/bench_runner.sh`** — Runs the 5 standard workloads on Ubuntu host, outputs JSON.
- [ ] **Step 3: Create `infrastructure/deployment/ubuntu-artifact/build.sh`** — Builds bootable Ubuntu ISO with benchmark runner embedded.
- [ ] **Step 4: Add CI job `build-ubuntu-artifact`** in `.github/workflows/ci.yml` — Runs on push to main, uploads ISO as artifact.
- [ ] **Step 5: Commit**

---

## Domain C: Security Hardening (Streams 9-11)

### Stream 9: KPTI Full Implementation + Boot Selftest
**Agent:** KPTI-Agent  
**Scope:** `kernel/seal-os/src/security/kpti.rs` (exists as scaffolding), `kernel/seal-os/src/boot/kernel_main.rs`  
**Goal:** Complete KPTI so CR3 swap isolates user and kernel page tables, with a boot selftest that proves it.  
**Independence:** Only touches KPTI files and kernel_main init sequence.

- [ ] **Step 1: Complete `kpti.rs`** — Implement `kpti_enter_user()` and `kpti_enter_kernel()` with proper PCID/INVPCID invalidation.
- [ ] **Step 2: Add KPTI boot selftest** — After KPTI init, attempt to access a kernel page from user mode (in a controlled test task), verify it page-faults.
- [ ] **Step 3: Wire selftest into T1-T10 gate** — `[BOOT] KPTI selftest PASS` required before scheduler start.
- [ ] **Step 4: Commit**

### Stream 10: Production TLS (X.509 + ECDHE)
**Agent:** TLS-Agent  
**Scope:** `kernel/seal-os/src/net/tls/` (create), `kernel/seal-os/src/net/tls13_psk.rs` (existing)  
**Goal:** Replace PSK-only TLS with full X.509 certificate validation, ECDHE key exchange, and PKI trust store.  
**Independence:** New `tls/` directory. Existing `tls13_psk.rs` kept as fallback.

- [ ] **Step 1: Create `kernel/seal-os/src/net/tls/x509.rs`** — Minimal X.509v3 parser: parse DER certificates, extract subject, issuer, validity, SPKI.
- [ ] **Step 2: Create `kernel/seal-os/src/net/tls/ecdhe.rs`** — X25519 ECDHE key exchange using `curve25519-dalek` or custom no_std implementation.
- [ ] **Step 3: Create `kernel/seal-os/src/net/tls/pki.rs`** — Trust store: embedded Mozilla CA bundle (subset), certificate chain validation, expiry check.
- [ ] **Step 4: Extend `TlsSocket`** — Negotiate ECDHE + AES-128-GCM, verify server cert against PKI store.
- [ ] **Step 5: Add TLS integration test** — Connect to `https://example.com` (or local test server), verify handshake completes.
- [ ] **Step 6: Commit**

### Stream 11: Security Audit Framework + Hardening Gates
**Agent:** Security-Audit-Agent  
**Scope:** `docs/CRYPTO_AUDIT.md`, `docs/THREAT_MODEL.md`, `kernel/seal-os/src/security/audit.rs`, `tests/security/` (create)  
**Goal:** Produce a runnable security audit suite that checks ASLR, seccomp, KPTI, SMAP/SMEP, stack canaries, and reports JSON.  
**Independence:** New `tests/security/` directory and additive audit code.

- [ ] **Step 1: Create `tests/security/audit_suite.py`** — Host-side runner that launches Seal OS in QEMU and runs security probes via serial.
- [ ] **Step 2: Add `kernel/seal-os/src/security/audit_runtime.rs`** — Kernel-side runtime that responds to audit probes: report ASLR entropy bits, seccomp filter count, KPTI status.
- [ ] **Step 3: Create `tests/security/test_aslr.py`** — Verifies mmap base shifts across 10 runs.
- [ ] **Step 4: Create `tests/security/test_seccomp.py`** — Verifies seccomp kills forbidden syscalls.
- [ ] **Step 5: Create `tests/security/test_kpti.py`** — Verifies user-mode cannot read kernel pages.
- [ ] **Step 6: Update `CRYPTO_AUDIT.md` and `THREAT_MODEL.md`** with completed status.
- [ ] **Step 7: Commit**

---

## Domain D: GPU Hardware Compute (Streams 12-13)

### Stream 12: Real GCN Shader Compilation Pipeline
**Agent:** GPU-Shader-Agent  
**Scope:** `kernel/seal-os/src/drivers/gpu/amd/pm4.rs`, `kernel/seal-os/src/drivers/gpu/shaders/` (create)  
**Goal:** Replace shader stubs with real OpenCL C → GCN ISA compiled binaries, and the host-side pipeline to compile them.  
**Independence:** New `shaders/` directory and PM4 packet builders additive.

- [ ] **Step 1: Create `kernel/seal-os/src/drivers/gpu/shaders/voronoi.cl`** — OpenCL C implementing spherical Voronoi computation.
- [ ] **Step 2: Create `kernel/seal-os/src/drivers/gpu/shaders/jl_project.cl`** — OpenCL C implementing Johnson-Lindenstrauss projection.
- [ ] **Step 3: Create `kernel/seal-os/src/drivers/gpu/shaders/spectral_contract.cl`** — OpenCL C implementing spectral contraction.
- [ ] **Step 4: Create `scripts/compile_shaders.py`** — Host-side script using ROCm/LLVM to compile `.cl` → `.bin` (GCN ISA binaries) at build time.
- [ ] **Step 5: Replace stub arrays in `pm4.rs`** with `include_bytes!()` of compiled `.bin` files.
- [ ] **Step 6: Commit**

### Stream 13: GPU Hardware Dispatch Proof Artifact
**Agent:** GPU-Proof-Agent  
**Scope:** `kernel/seal-os/src/drivers/gpu/dispatch.rs` (create), `tests/gpu/` (create), `docs/NVIDIA_4060_TEST_PLAN.md`  
**Goal:** Create a proof artifact that runs GPU compute on real AMD hardware and verifies correct results.  
**Independence:** New files only.

- [ ] **Step 1: Create `kernel/seal-os/src/drivers/gpu/dispatch.rs`** — Hardware dispatch ring management: map PM4 ring, submit compute packet, wait for completion interrupt.
- [ ] **Step 2: Create `tests/gpu/test_amd_compute.py`** — Host-side test that runs Seal OS on AMD GPU hardware (or QEMU with GPU passthrough), submits compute job, reads back result.
- [ ] **Step 3: Add `[GPU-BENCH]` serial sentinel** — Seal OS prints `[GPU-BENCH] voronoi result=OK cycles=<n>` after hardware dispatch.
- [ ] **Step 4: Update `NVIDIA_4060_TEST_PLAN.md`** → rename to `GPU_TEST_PLAN.md`, add AMD test matrix.
- [ ] **Step 5: Commit**

---

## Domain E: Browser Engine (Streams 14-18)

### Stream 14: HTML Parser
**Agent:** Browser-HTML-Agent  
**Scope:** `userspace/browser/src/html.rs` (create), `userspace/browser/src/dom.rs` (create)  
**Goal:** Parse real-world HTML into a DOM tree. Handle malformed HTML gracefully.  
**Independence:** New `userspace/browser/` directory.

- [ ] **Step 1: Create `userspace/browser/Cargo.toml`** — New workspace crate, no_std where possible.
- [ ] **Step 2: Create `userspace/browser/src/html.rs`** — Tokenizer (tags, text, attributes, comments) + parser building DOM tree.
- [ ] **Step 3: Create `userspace/browser/src/dom.rs`** — DOM node types: `Document`, `Element`, `Text`, `Attr`. Tree operations: `append_child`, `query_selector` (basic tag name only).
- [ ] **Step 4: Parse `https://example.com` raw HTML in test** — Fetch via HTTPS client, parse, verify `<title>` extracted.
- [ ] **Step 5: Commit**

### Stream 15: CSS Engine
**Agent:** Browser-CSS-Agent  
**Scope:** `userspace/browser/src/css.rs`, `userspace/browser/src/style.rs`  
**Goal:** Parse CSS and compute styles for DOM nodes.  
**Independence:** New files in `userspace/browser/`.

- [ ] **Step 1: Create `userspace/browser/src/css.rs`** — CSS tokenizer + parser: rules, selectors, declarations. Support element, class, id selectors.
- [ ] **Step 2: Create `userspace/browser/src/style.rs`** — Style computation: cascade inherited styles, apply matching rules, produce `ComputedStyle` per node.
- [ ] **Step 3: Test** — Parse `<style>body{color:red}</style><body>`, verify computed `color=red`.
- [ ] **Step 4: Commit**

### Stream 16: JavaScript Interpreter (Minimal)
**Agent:** Browser-JS-Agent  
**Scope:** `userspace/browser/src/js/` (create)  
**Goal:** A minimal JS engine that can manipulate the DOM and handle event handlers. Not V8 — just enough for basic pages.  
**Independence:** New directory.

- [ ] **Step 1: Create `userspace/browser/src/js/lexer.rs`** — Tokenize JS: keywords, identifiers, literals, operators.
- [ ] **Step 2: Create `userspace/browser/src/js/parser.rs`** — Parse expressions, statements, functions, objects.
- [ ] **Step 3: Create `userspace/browser/src/js/interpreter.rs`** — Tree-walk interpreter with `Scope` chain.
- [ ] **Step 4: Create `userspace/browser/src/js/dom_binding.rs`** — Expose `document.querySelector` and `element.innerHTML` to JS.
- [ ] **Step 5: Test** — Run `document.querySelector('h1').innerHTML = 'Hello'` and verify DOM mutation.
- [ ] **Step 6: Commit**

### Stream 17: Layout Engine
**Agent:** Browser-Layout-Agent  
**Scope:** `userspace/browser/src/layout.rs`  
**Goal:** Compute box model and flow layout for DOM nodes.  
**Independence:** New file.

- [ ] **Step 1: Create `userspace/browser/src/layout.rs`** — Box tree: `LayoutBox` with `rect: Rect`, `children: Vec<LayoutBox>`. Flow layout: block and inline.
- [ ] **Step 2: Implement block layout** — Width from parent, height from children, margin/padding/border.
- [ ] **Step 3: Implement inline layout** — Text runs broken into lines (simple greedy wrapping).
- [ ] **Step 4: Test** — Layout `<div><p>Hello</p></div>`, verify box positions.
- [ ] **Step 5: Commit**

### Stream 18: Browser Renderer / Compositor
**Agent:** Browser-Render-Agent  
**Scope:** `userspace/browser/src/render.rs`, `userspace/browser/src/main.rs`  
**Goal:** Render layout boxes to the Seal OS framebuffer via the window manager.  
**Independence:** New files. Integrates with WM via existing syscalls.

- [ ] **Step 1: Create `userspace/browser/src/render.rs`** — Paint phase: fill backgrounds, borders, text using htek.rs primitives via syscall.
- [ ] **Step 2: Create `userspace/browser/src/main.rs`** — Browser app: URL bar, fetch HTML, parse, layout, render loop. Window managed by Seal WM.
- [ ] **Step 3: Integrate into desktop** — Add browser icon to desktop, launchable.
- [ ] **Step 4: Test** — Launch browser, navigate to `https://example.com`, verify visible text rendered.
- [ ] **Step 5: Commit**

---

## Domain F: Kernel & Runtime Completion (Streams 19-25)

### Stream 19: Legacy Host Wrapper Replacement
**Agent:** Legacy-Cleanup-Agent  
**Scope:** `scripts/`, `infrastructure/tools/`  
**Goal:** Replace every Python/bash script outside the LAAMBA runtime path with Rust or Aether-Lang equivalents.  
**Independence:** New `infrastructure/tools/` binaries; deletes/replaces scripts after replacement proven.

- [ ] **Step 1: Audit `scripts/`** — List every script and its function.
- [ ] **Step 2: Rewrite `scripts/build_iso.sh` as `infrastructure/tools/build_iso.rs`** — Rust binary using `std::process` for remaining external deps.
- [ ] **Step 3: Rewrite `scripts/demo.sh` as `infrastructure/tools/demo.rs`**.
- [ ] **Step 4: Rewrite `scripts/local_ci.sh` as `infrastructure/tools/local_ci.rs`**.
- [ ] **Step 5: Quarantine unmigrated scripts** — Move remaining scripts to `experiments/legacy_scripts/` with README explaining they are deprecated.
- [ ] **Step 6: Commit**

### Stream 20: FAT Fixture Tests
**Agent:** FAT-Test-Agent  
**Scope:** `tests/test_fs_fat.py` (create), `kernel/seal-os/src/fs/fat.rs`  
**Goal:** Comprehensive FAT12/16/32 fixture tests proving parity with Linux behavior.  
**Independence:** New test file, additive test helpers in fat.rs.

- [ ] **Step 1: Create `tests/test_fs_fat.py`** — Generate FAT images with `mkfs.fat`, mount in Seal OS QEMU, run create/read/write/delete/rename/mkdir/rmdir tree.
- [ ] **Step 2: Add `#[cfg(test)]` unit tests in `fat.rs`** — In-kernel tests for cluster allocation, directory entry parsing.
- [ ] **Step 3: Add parity assertions** — After each operation, compute MD5 of file contents and compare Linux vs Seal OS.
- [ ] **Step 4: Commit**

### Stream 21: ext2 Fixture Tests
**Agent:** ext2-Test-Agent  
**Scope:** `tests/test_fs_ext2.py` (create), `kernel/seal-os/src/fs/ext2.rs`  
**Goal:** Comprehensive ext2 fixture tests proving parity.  
**Independence:** New test file, additive test helpers.

- [ ] **Step 1: Create `tests/test_fs_ext2.py`** — Generate ext2 images with `mkfs.ext2`, mount in Seal OS QEMU, run full operation tree.
- [ ] **Step 2: Add `#[cfg(test)]` unit tests in `ext2.rs`** — Inode allocation, indirect block traversal, `..` fixup.
- [ ] **Step 3: Add parity assertions** — Byte-level compare after Linux vs Seal OS operations.
- [ ] **Step 4: Commit**

### Stream 22: Package Manager Remote Registry
**Agent:** Package-Registry-Agent  
**Scope:** `apps/laamba-governor/src/registry.rs` (create), `kernel/seal-os/src/pkg/` (create)  
**Goal:** Enable `.eph` package download from HTTPS registry with Ed25519 signature verification.  
**Independence:** New files.

- [ ] **Step 1: Create `kernel/seal-os/src/pkg/registry_client.rs`** — HTTPS client for fetching `.eph` manifests and packages from registry URL.
- [ ] **Step 2: Create `kernel/seal-os/src/pkg/signature.rs`** — Ed25519 verification of package signatures against embedded public key.
- [ ] **Step 3: Create `apps/laamba-governor/src/registry.rs`** — Registry server mock (for testing) and client UI.
- [ ] **Step 4: Add `sys_pkg_install_from_registry(url)` syscall** — Downloads, verifies signature, extracts to ManifoldFS.
- [ ] **Step 5: Test** — Host mock registry, install package in Seal OS, verify files present.
- [ ] **Step 6: Commit**

### Stream 23: Installer GPT Partitioning + Formatting
**Agent:** Installer-Agent  
**Scope:** `userspace/installer/src/partition.rs` (create), `userspace/installer/src/format.rs` (create)  
**Goal:** Make the Installer UI actually write GPT partitions and format FAT/ext2/ManifoldFS on raw block devices.  
**Independence:** New files in existing `userspace/installer/`.

- [ ] **Step 1: Create `userspace/installer/src/partition.rs`** — GPT header creation, partition entry writing, protective MBR.
- [ ] **Step 2: Create `userspace/installer/src/format.rs`** — `mkfs.fat` equivalent, `mkfs.ext2` equivalent, ManifoldFS superblock init.
- [ ] **Step 3: Wire into Installer UI** — When user clicks "Install", write GPT, create Seal OS partition, format it, copy kernel + EFI.
- [ ] **Step 4: Test** — Run installer in QEMU on blank disk, verify bootable afterwards.
- [ ] **Step 5: Commit**

### Stream 24: LKM (Loadable Kernel Module) Framework
**Agent:** LKM-Agent  
**Scope:** `kernel/seal-os/src/lkm/` (create), `kernel/seal-os/src/syscall/lkm.rs` (create)  
**Goal:** Design and implement a minimal LKM ABI so drivers can be loaded at runtime.  
**Independence:** New directory.

- [ ] **Step 1: Create `kernel/seal-os/src/lkm/abi.rs`** — Module header, symbol table, relocation types for x86_64.
- [ ] **Step 2: Create `kernel/seal-os/src/lkm/loader.rs`** — Parse `.sealmod` ELF-like module, apply relocations, call `module_init()`.
- [ ] **Step 3: Create `kernel/seal-os/src/lkm/registry.rs`** — Registered modules list, `sys_module_load(path)`, `sys_module_unload(name)`.
- [ ] **Step 4: Create example module `drivers/dummy_lkm.rs`** — Minimal "hello world" module proving the framework.
- [ ] **Step 5: Test** — Load dummy module at runtime, verify init runs, unload, verify cleanup runs.
- [ ] **Step 6: Commit**

### Stream 25: ManifoldFS Real Block Store
**Agent:** ManifoldFS-Block-Agent  
**Scope:** `kernel/seal-os/src/fs/manifoldfs/block.rs` (create), `kernel/seal-os/src/fs/manifoldfs/persist.rs`  
**Goal:** Replace mock block-store path with real persistence, while keeping metadata teleport O(1).  
**Independence:** New block.rs, modifications to persist.rs only.

- [ ] **Step 1: Create `kernel/seal-os/src/fs/manifoldfs/block.rs`** — Real block allocator on top of AHCI/NVMe: read/write 4KiB blocks, journal for metadata.
- [ ] **Step 2: Modify `persist.rs`** — Use real block store when `fs_mode=real_block`, keep mock for tests.
- [ ] **Step 3: Add boot parameter `manifoldfs=real`** to select real block mode.
- [ ] **Step 4: Test** — Boot with real block store, create files, reboot, verify persistence.
- [ ] **Step 5: Commit**

---

## Domain G: CI, Docs, Meta (Streams 26-28)

### Stream 26: CI Proof Pipeline Hardening
**Agent:** CI-Agent  
**Scope:** `.github/workflows/ci.yml`, `.github/workflows/proof.yml`, `docs/CI.md`  
**Goal:** Ensure CI gates every claim: benchmark, security, GPU, browser, self-host.  
**Independence:** CI workflow files only.

- [ ] **Step 1: Add `job: security-gate`** — Runs audit suite, fails on any hardening failure.
- [ ] **Step 2: Add `job: benchmark-gate`** — Runs side-by-side harness, posts comparison table as PR comment.
- [ ] **Step 3: Add `job: aether-selfhost-gate`** — Compiles Aether bootstrap with itself, verifies bytecode match.
- [ ] **Step 4: Add `job: browser-gate`** — Launches browser in QEMU, verifies rendered pixels for example.com.
- [ ] **Step 5: Update `docs/CI.md`** with new pipeline diagram.
- [ ] **Step 6: Commit**

### Stream 27: Documentation Synchronization
**Agent:** Docs-Agent  
**Scope:** `docs/`, `README.md`  
**Goal:** Every completed stream updates its docs. README "Honest Status Dashboard" becomes "Real Status Dashboard."  
**Independence:** Markdown files only.

- [ ] **Step 1: Update README.md Honest Status Dashboard** — Mark completed items ✅, move partial items to Real.
- [ ] **Step 2: Update `docs/SEAL_OS_GUIDE.md`** with browser usage, package manager registry, installer.
- [ ] **Step 3: Update `docs/THREAT_MODEL.md`** with KPTI, TLS, audit completion.
- [ ] **Step 4: Update `docs/BENCHMARK_PLAN.md`** with Ubuntu artifact location and reproduction steps.
- [ ] **Step 5: Commit**

### Stream 28: Theorem Proof Update (T1-T10)
**Agent:** Theorem-Agent  
**Scope:** `docs/THEOREMS.md`, `kernel/seal-os/src/proofs/` (create), Lean 4 proofs if present  
**Goal:** Update Lean 4 proofs to cover all new runtime paths (browser layout, TLS handshake, GPU dispatch).  
**Independence:** Proof files only.

- [ ] **Step 1: Locate existing Lean 4 proof files** (check `docs/proofs/` or `future/`).
- [ ] **Step 2: Add proof sketch for TLS handshake convergence** — Modeled as fixed-point iteration.
- [ ] **Step 3: Add proof sketch for browser layout termination** — Box tree depth bounded by DOM depth.
- [ ] **Step 4: Add proof sketch for GPU dispatch correctness** — PM4 ring invariant.
- [ ] **Step 5: Update `docs/THEOREMS.md`** with new T11-T13 or extend T6-T10.
- [ ] **Step 6: Commit**

---

## Self-Review

**1. Spec coverage:**
- Aether self-hosting → Streams 1-6
- Benchmarks/Ubuntu → Streams 7-8
- Security → Streams 9-11
- GPU → Streams 12-13
- Browser → Streams 14-18
- Build system/legacy → Streams 6, 19
- Filesystem parity → Streams 20-21
- Package manager → Stream 22
- Installer → Stream 23
- LKM → Stream 24
- ManifoldFS real block → Stream 25
- CI/docs/proofs → Streams 26-28

**2. Placeholder scan:** No TBD/TODO/fill-in found. All steps have concrete file paths and deliverables.

**3. Type consistency:** All new modules use existing Seal OS patterns (`no_std`, `alloc`, Seal ABI syscalls). Browser is userspace so uses `std`.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-06-03-seal-os-complete.md`.**

**Execution approach:** Dispatching-Parallel-Agents (28 streams, each isolated to specific files).

**Each agent receives:**
- Exact stream number and name
- Scope (files to touch)  
- Goal (one sentence)
- Independence note (which files are off-limits)
- Five bite-sized steps with expected deliverables
- Instruction: "Do NOT touch files outside your scope. Commit when done. Report DONE or BLOCKED."
