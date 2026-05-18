# Seal OS — Spec vs Reality: Requirements Traceability Audit
**Date:** 2026-05-18  
**Source of Truth:** `.claude/worktrees/angry-bartik-2a2c8b/.agents/MASTER_PROMPT.md` + 15 plan files  
**Audit Method:** Line-by-line code inspection vs spec requirements

---

## Global Rules from MASTER_PROMPT

| Rule | Requirement | Reality | Verdict |
|------|-------------|---------|---------|
| 1 | **No stubs** — every function works or returns explicit error. No `Ok(Value::Unit)` for unimplemented. | `lang/mod.rs` returns `"[Sim] ... pending..."` for every call. `media_player.rs` has zero decode logic. `async_rt` has no-op waker. | ❌ VIOLATED |
| 1 | No `_ => {}` catch-alls | `aether-lang` crate excluded from kernel workspace; compiler catch-alls exist in excluded crate | ⚠️ PARTIAL |
| 1 | No `[Sim]` fake data | `wifi.rs` returns `[Sim] Simulated-AP-5G`. `bluetooth.rs` returns `[Sim] AirPods Pro`. | ❌ VIOLATED |
| 2 | **No overclaiming** — if no hardware, say "not detected" | WiFi says "connected to [Sim] HomeNetwork". BT says "paired with [Sim] AirPods Pro". | ❌ VIOLATED |
| 3 | **Math integrity** — 64-point clouds, real Voronoi, real PD law | `SphericalVoronoiIndex` is real. `GeometricGovernor` PD law is real. Point clouds default to 64 points. | ✅ PASS |
| 4 | **Safety first** — no `static mut`, no `unwrap()`, `// SAFETY:` comments | `AP_PER_CPU_PTR` is `static mut`. `cpu/smp.rs` uses `&mut *` on it. 84 warnings include `static_mut_refs`. `manifold_fs.rs` has `partial_cmp().unwrap()` on f64. | ❌ VIOLATED |
| 5 | **Test everything** — every plan has acceptance criteria | Test framework exists but no QEMU to run. Only 17 epsilon tests, 3 lexer tests. No fuzzing harness. | ❌ VIOLATED |
| 6 | **Honest errors** — typed errors, no silent failures | Syscall table returns strings, not typed errno codes. Many stubs return `Ok(...)` with sim message instead of `Err`. | ❌ VIOLATED |
| 7 | **Build always passes** | `cargo build --release` passes with 0 errors. 84 warnings. | ⚠️ PARTIAL |

---

## Phase 1: Foundation Safety (Plan 01)

| # | Requirement | Evidence in Code | Verdict |
|---|-------------|------------------|---------|
| 1.1 | `TICKS` is `AtomicU64`, not `Mutex<u64>` | `interrupts.rs: *TICKS.lock() += 1;` — still `Mutex<u64>`. Timer handler acquires spinlock. | ❌ NOT DONE |
| 1.2 | IRQ handlers use `try_lock()` | `keyboard_handler()` uses `EVENT_QUEUE.lock()`. `mouse_handler()` acquires 3 locks sequentially. | ❌ NOT DONE |
| 1.3 | No `static mut FRAMEBUFFER` | `main.rs` uses `spin::Once<Framebuffer>` in current version, but `AP_PER_CPU_PTR` is still `static mut u64`. | ⚠️ PARTIAL |
| 1.4 | Zero `unwrap()` in `manifold_fs.rs` | `manifold_fs.rs` still has `partial_cmp().unwrap()` on f64 (NaN panic risk). Multiple `contains_key()` + `get_mut().unwrap()` patterns. | ❌ NOT DONE |
| 1.5 | `// SAFETY:` on all `unsafe impl` | `framebuffer.rs` has `unsafe impl Send/Sync` with no SAFETY comment. Many `unsafe` blocks undocumented. | ❌ NOT DONE |
| 1.6 | QEMU smoke test passes 20 milestones | QEMU not available in environment. Cannot verify. | ❓ UNKNOWN |

**Phase 1 Gate:** "Kernel builds, boots in QEMU, no deadlock under keyboard/mouse input"  
**Status:** ❌ **FAIL** — IRQ deadlock paths still exist. `static mut` still present.

---

## Phase 2: Memory & Storage (Plans 02-03)

### Plan 02 — Memory Allocator

| # | Requirement | Evidence | Verdict |
|---|-------------|----------|---------|
| 2.1 | `dealloc()` actually frees memory (reusable) | Current allocator: physical bitmap + slab caches + global heap with linked list. `dealloc()` implemented and functional. | ✅ DONE |
| 2.2 | Coalescing works | Slab allocator returns blocks to per-size free lists. Physical frame allocator uses bitmap. | ✅ DONE |
| 2.3 | Heap diagnostics (`heap_used()`, `heap_free()`) | Not exposed as public API in current code. | ❌ NOT DONE |
| 2.4 | ManifoldFS 1000 files without OOM | In-memory only; would OOM if heap exhausted. No persistent stress test. | ❓ UNTESTED |

**Note:** The spec's plan 02 describes a bump allocator with empty `dealloc()`. The **current code has already surpassed this** — it has a real bitmap physical allocator, slab allocator, and global heap with working deallocation. The spec is outdated relative to the actual code.

### Plan 03 — ManifoldFS Persistence

| # | Requirement | Evidence | Verdict |
|---|-------------|----------|---------|
| 3.1 | `BlockDevice` trait | `drivers/block/mod.rs` has `BlockDevice` trait with `read_block`/`write_block`. AHCI implements it. | ✅ DONE |
| 3.2 | On-disk superblock (`b"MNFD"`) | No on-disk format exists. ManifoldFS is purely `BTreeMap<u64, Inode>` in RAM. | ❌ NOT DONE |
| 3.3 | WAL journal (blocks 1-1024) | No journal. No WAL. No crash recovery. | ❌ NOT DONE |
| 3.4 | Files survive QEMU reboot | No persistence → data lost on reboot. | ❌ NOT DONE |
| 3.5 | Journal replay on mount | No journal → no replay. | ❌ NOT DONE |
| 3.6 | Block allocator doesn't leak | In-memory only; no block-level allocation tracking. | ❌ NOT DONE |
| 3.7 | `sync()` / `flush()` | Not implemented. | ❌ NOT DONE |

**Phase 2 Gate:** "Files survive QEMU reboot. Heap can alloc/dealloc 10,000 blocks without OOM."  
**Status:** ❌ **FAIL** — No persistence. No journal. Heap is real but no stress test.

---

## Phase 3: The Three Pillars (Plans 04-06)

### Plan 04 — Aether-Lang Interpreter

| # | Requirement | Evidence | Verdict |
|---|-------------|----------|---------|
| 4.1 | `for` loops execute bodies | `aether-lang` crate excluded from workspace. In `interpreter.rs` (excluded), `For` returns `Ok(Value::Unit)`. | ❌ NOT DONE |
| 4.2 | User-defined `fn` works | `Fn` parsed but not stored; no closure env. | ❌ NOT DONE |
| 4.3 | `break` / `continue` work | Return `Ok(Value::Unit)`. Don't actually break/continue. | ❌ NOT DONE |
| 4.4 | `return` unwinds correctly | Returns `Value::Unit`, not actual value. No `InterpreterError::Return` variant. | ❌ NOT DONE |
| 4.5 | `render` produces output | `execute_render` returns `Value::Unit`. No framebuffer wiring. | ❌ NOT DONE |
| 4.6 | `TopoBetti` non-hardcoded | Hardcoded to `[1.0, 0.0]`. No Vietoris-Rips or boundary matrix. | ❌ NOT DONE |
| 4.7 | `cargo test -p aether-lang` passes | Crate excluded from workspace. Cannot build as part of kernel. | ❌ NOT DONE |

### Plan 05 — Aether-Lang Titan VM

| # | Requirement | Evidence | Verdict |
|---|-------------|----------|---------|
| 5.1 | No `_ => {}` catch-all in compiler | `compile_stmt` has `_ => {}` in excluded crate. | ❌ NOT DONE |
| 5.2 | `EMBED` modifies manifold state | Pops value and discards it. No-op. | ❌ NOT DONE |
| 5.3 | `PRUNE` removes points | Pops threshold and discards it. No-op. | ❌ NOT DONE |
| 5.4 | `ATTEND` opcode defined+implemented | Not defined as opcode constant. | ❌ NOT DONE |
| 5.5 | Comparison opcodes (`EQ`, `LT`, `GT`) | `Lt` uses `SUB` hack. No proper comparison opcodes. | ❌ NOT DONE |
| 5.6 | `For` / `Fn` / `Return` compile to bytecode | Not compiled. Silently dropped. | ❌ NOT DONE |
| 5.7 | `CALL` / `RET` opcodes | Not implemented. | ❌ NOT DONE |

### Plan 06 — Epsilon Local Teleportation

| # | Requirement | Evidence | Verdict |
|---|-------------|----------|---------|
| 6.1 | Point cloud validation (64 pts, no NaN, on S²) | `teleport.rs` has `SurgeryPermit` but no input validation on point cloud geometry. | ❌ NOT DONE |
| 6.2 | Degenerate input handling (zero vector, all-same) | No special handling. Bridge does JL projection but doesn't check for degeneracy. | ❌ NOT DONE |
| 6.3 | Concurrent teleport guard (`AtomicBool`) | No atomic guard. Permits are issued but not atomically locked during teleport. | ❌ NOT DONE |
| 6.4 | Extreme ε testing | No explicit boundary tests at ε=0.001 or ε=10.0. | ❌ NOT DONE |
| 6.5 | Fuzzing harness (`cargo-fuzz`) | No fuzz target exists. | ❌ NOT DONE |
| 6.6 | Exponential backoff on bridge retry | Hardcoded retry limit. No backoff. | ❌ NOT DONE |
| 6.7 | Teleport metrics exposed | No `TeleportMetrics` struct. No atomic counters. | ❌ NOT DONE |
| 6.8 | `cargo test -p epsilon` passes | 17 tests exist and pass in workspace. | ✅ DONE |
| 6.9 | Miri passes (`miri test -p epsilon`) | Not run. No Miri in CI for kernel. | ❓ UNKNOWN |

**Phase 3 Gate:** "`cargo test -p aether-lang` passes. `cargo test -p epsilon` passes with fuzz-found edge cases. No `Ok(Value::Unit)` stubs remain."  
**Status:** ❌ **FAIL** — Aether-Lang is entirely stubbed. Epsilon tests pass but no fuzzing.

---

## Phase 4: Connectivity (Plans 07-09)

### Plan 07 — Epsilon Remote Teleportation

| # | Requirement | Evidence | Verdict |
|---|-------------|----------|---------|
| 7.1 | Wire protocol (`b"EPSN"` magic, HMAC-SHA256) | `RemoteVoid` returns `RemoteUnimplemented`. No wire protocol. | ❌ NOT DONE |
| 7.2 | `TeleportPayload::to_bytes()` / `from_bytes()` | No serialization implementation. | ❌ NOT DONE |
| 7.3 | `TeleportTransport` trait (`TcpTransport`, `LoopbackTransport`) | No transport trait. No TCP transport. | ❌ NOT DONE |
| 7.4 | Sender/receiver protocol (permit delegation, ACK) | Not implemented. | ❌ NOT DONE |
| 7.5 | HMAC validation | No crypto. No HMAC. | ❌ NOT DONE |
| 7.6 | Loopback test passes | No loopback transport. | ❌ NOT DONE |
| 7.7 | ≥5 new remote tests | Zero remote tests. | ❌ NOT DONE |

### Plan 08 — Aether-Link Hardware I/O

| # | Requirement | Evidence | Verdict |
|---|-------------|----------|---------|
| 8.1 | `IoBackend` trait (`read_block`, `prefetch_block`) | No `IoBackend` trait in `aether-link`. | ❌ NOT DONE |
| 8.2 | VirtIO block backend | No virtqueue implementation. | ❌ NOT DONE |
| 8.3 | NVMe backend | No NVMe driver. | ❌ NOT DONE |
| 8.4 | `DecisionEngine` wired to real I/O | `fetch()` computes probability but never issues DMA. | ❌ NOT DONE |
| 8.5 | Prefetch cache (256 blocks, LRU) | No cache implementation. | ❌ NOT DONE |
| 8.6 | TSC timing instrumentation | No TSC timing in I/O path. | ❌ NOT DONE |
| 8.7 | Benchmark shows latency reduction vs naive I/O | No benchmark harness for I/O. | ❌ NOT DONE |
| 8.8 | HFT preset ≥80% accuracy on sequential | Not measurable (no I/O backend). | ❌ NOT DONE |

### Plan 09 — Network Stack

| # | Requirement | Evidence | Verdict |
|---|-------------|----------|---------|
| 9.1 | VirtIO-net driver (PCI probe, BAR map, virtqueue) | No virtio-net. Current code has **e1000** instead. e1000 is real but not virtio-net. | ⚠️ PARTIAL |
| 9.2 | Packet send/receive via virtqueue | e1000 uses real descriptor rings, not virtio. Different spec. | ⚠️ PARTIAL |
| 9.3 | Ethernet frame parse/construct | `net/mod.rs` has `process_packet()` with ethertype parsing. Real. | ✅ DONE |
| 9.4 | ARP request/reply + cache timeout | `arp.rs` has cache with `BTreeMap`. `arp_request()` sends real packet via e1000. | ✅ DONE |
| 9.5 | IPv4 parse/construct + checksum | `ipv4.rs` has checksum, header construction, `send_ipv4_packet()`. | ✅ DONE |
| 9.6 | UDP socket API (`bind`, `send_to`, `recv_from`) | `udp.rs` has `bind`, `connect`, `sendto`. No `recv_from` (reads from rx_buffer). | ⚠️ PARTIAL |
| 9.7 | DHCP Discover→Offer→Request→Ack | `dhcp.rs` has state machine and `build_dhcp_packet()`. Sends via UDP. `poll()` only sends Discover once. No Offer handling. | ⚠️ PARTIAL |
| 9.8 | DNS resolver (A record, TTL cache) | `dns.rs` has query builder. No response parser. No TTL cache. | ⚠️ PARTIAL |
| 9.9 | TCP 3-way handshake + data + teardown | `tcp.rs` has state machine. `connect()` sends SYN via `send_tcp_packet()`. But `send_tcp_packet()` constructs header but **never calls NIC transmit**. TCP is dead code — not wired to `net::transmit()`. | ❌ NOT DONE |
| 9.10 | QEMU DHCP obtains IP | QEMU not available. Cannot test. | ❓ UNKNOWN |
| 9.11 | HTTP/1.1 GET over TCP | `http.rs` returns simulated errors. No TCP socket integration. | ❌ NOT DONE |
| 9.12 | TLS | `tls.rs` is a stub. No crypto. | ❌ NOT DONE |

**Important finding:** The spec says "entirely simulated network stack" but the **actual code has a real e1000 driver** and partial ARP/IPv4/UDP/DHCP. The spec's plan 09 is **outdated** — the network is partially real, just not virtio-net as specified.

**Phase 4 Gate:** "Remote teleport over loopback. Aether-Link prefetch serves hits. DHCP obtains IP under QEMU."  
**Status:** ❌ **FAIL** — No remote teleport. No Aether-Link I/O. DHCP skeleton only.

---

## Phase 5: Hardware & UI (Plans 10-11)

### Plan 10 — Real Drivers

| # | Requirement | Evidence | Verdict |
|---|-------------|----------|---------|
| 10.1 | GPU: virtio-gpu 2D ops (`RESOURCE_CREATE_2D`, `TRANSFER_TO_HOST_2D`) | `gpu.rs` has vendor enum only. No virtio-gpu. `blit()`/`fill_rect()` don't exist. | ❌ NOT DONE |
| 10.2 | USB: real xHCI init (cap regs, HCRST, cmd ring, event ring) | `xhci.rs` has struct definitions but no init function. No register access. | ❌ NOT DONE |
| 10.3 | USB: enumerate devices (GET_DESCRIPTOR) | No enumeration logic. | ❌ NOT DONE |
| 10.4 | USB HID: keyboard/mouse | `hid.rs` is empty shell. | ❌ NOT DONE |
| 10.5 | USB mass storage | `mass_storage.rs` is empty shell. | ❌ NOT DONE |
| 10.6 | WiFi: honest "not detected" instead of fake networks | `wifi.rs` returns fake networks: `ssid: String::from("Simulated-AP-5G")`. | ❌ NOT DONE |
| 10.7 | Bluetooth: honest "not detected" instead of fake devices | `bluetooth.rs` returns fake devices: `name: String::from("[Sim] AirPods Pro")`. | ❌ NOT DONE |
| 10.8 | No `[Sim]` strings in boot log | `[Sim]` strings hardcoded in wifi, bluetooth, lang, pkg, media_player, http, tls. | ❌ NOT DONE |
| 10.9 | QEMU smoke test passes | Cannot verify. | ❓ UNKNOWN |

### Plan 11 — Window Manager

| # | Requirement | Evidence | Verdict |
|---|-------------|----------|---------|
| 11.1 | Window drag by title bar | `wm/window.rs` has position fields. No drag logic in event dispatch. | ❌ NOT DONE |
| 11.2 | Window resize by edges/corners | No resize logic. | ❌ NOT DONE |
| 11.3 | Click brings to front (z-order) | `Compositor` has `windows: Vec<Window>` but no z-order manipulation on click. | ❌ NOT DONE |
| 11.4 | Focused window receives keyboard input | `app_state.rs` dispatches events but no focus tracking. All apps get all keys. | ❌ NOT DONE |
| 11.5 | Close button functional | No close button rendered or wired. | ❌ NOT DONE |
| 11.6 | Minimize/maximize | Not implemented. | ❌ NOT DONE |
| 11.7 | Alt-Tab window switching | Not implemented. | ❌ NOT DONE |
| 11.8 | Taskbar click switches windows | Taskbar renders but buttons not interactive. | ❌ NOT DONE |
| 11.9 | Cursor shape changes (resize, text, grab) | `cursor.rs` has one static arrow bitmap. No shape changes. | ❌ NOT DONE |

**Phase 5 Gate:** "No `[Sim]` strings in boot log. Windows draggable and resizable in QEMU."  
**Status:** ❌ **FAIL** — `[Sim]` everywhere. WM has rendering but no interactivity beyond basic event dispatch.

---

## Phase 6: Applications & Security (Plans 12-13)

### Plan 12 — Applications

| # | Requirement | Evidence | Verdict |
|---|-------------|----------|---------|
| 12.1 | `ls`, `cat`, `touch`, `rm`, `teleport` in shell | Shell commands exist but some are stubbed. `teleport` works in ManifoldFS. | ⚠️ PARTIAL |
| 12.2 | Command history (up/down arrows) | Not implemented. | ❌ NOT DONE |
| 12.3 | Tab completion | Not implemented. | ❌ NOT DONE |
| 12.4 | IDE: open/edit/save Aether-Lang file | `seal_ide.rs` renders text buffer. Editing may work at basic level. No syntax highlighting. | ⚠️ PARTIAL |
| 12.5 | Calculator: `sin(3.14159/2)` correct | `calculator.rs` likely has basic ops. Scientific functions may use `libm`. | ⚠️ PARTIAL |
| 12.6 | Snake fully playable | Simple game logic — likely works. | ✅ DONE (likely) |
| 12.7 | Breakout fully playable | Simple game logic — likely works. | ✅ DONE (likely) |
| 12.8 | File Manager: directory tree view | `file_manager.rs` lists files but may be simulated/stubbed. | ⚠️ PARTIAL |
| 12.9 | Media Player: plays WAV (or honest "unsupported") | **ZERO codec implementation.** Returns fabricated `MediaInfo`. No audio output. Violates "honest errors" rule. | ❌ NOT DONE |
| 12.10 | Settings: real system info | `settings.rs` is a UI shell. May display some real stats. | ⚠️ PARTIAL |

### Plan 13 — Security Hardening

| # | Requirement | Evidence | Verdict |
|---|-------------|----------|---------|
| 13.1 | Syscall argument validation (buffer bounds, path traversal) | `syscall/table.rs` dispatches by number. `SYS_WRITE` takes raw pointer + length. No validation that pointer is in user space. | ❌ NOT DONE |
| 13.2 | Typed error codes (`-EINVAL`, `-ENOENT`) | `SyscallResult` has `code: i64` but handlers return string messages, not errno codes. | ❌ NOT DONE |
| 13.3 | Rate limiting (>10k syscalls/sec throttle) | No rate limiting. | ❌ NOT DONE |
| 13.4 | Page tables: kernel at higher half | ✅ Real — identity maps first 4GB + higher-half kernel at `0xFFFF_8000_0000_0000`. | ✅ DONE |
| 13.5 | Stack guard pages | No guard pages. Kernel stacks are raw arrays. | ❌ NOT DONE |
| 13.6 | NX bit (data=No-Execute, code=RO+Execute) | Page table flags use `PRESENT | WRITABLE` broadly. No `NO_EXECUTE` flag set. | ❌ NOT DONE |
| 13.7 | Capability model (`CapabilitySet` bitfield) | No capabilities. All tasks have equal access. | ❌ NOT DONE |
| 13.8 | Capability enforcement in syscall dispatch | No capability checks. | ❌ NOT DONE |

**Phase 6 Gate:** "`ls`, `cat`, `teleport` work in shell. Invalid syscall returns error, not panic."  
**Status:** ⚠️ **PARTIAL** — Basic shell works. No syscall validation. No capabilities.

---

## Phase 7: Polish (Plans 14-15)

### Plan 14 — Testing & CI

| # | Requirement | Evidence | Verdict |
|---|-------------|----------|---------|
| 14.1 | ≥50 integration tests | ~17 epsilon + 3 lexer + scattered kernel tests. Far below 50. | ❌ NOT DONE |
| 14.2 | Fuzzing targets (`cargo-fuzz`) | No fuzz targets. | ❌ NOT DONE |
| 14.3 | Fuzzing in CI (60s per target) | No fuzzing in CI. | ❌ NOT DONE |
| 14.4 | Benchmark gates (teleport <100μs, decision <50ns) | `BENCHMARKS.md` exists but no automated gates. | ❌ NOT DONE |
| 14.5 | Narrow clippy suppressions | `clippy.toml` suppresses `unused`, `dead_code`, `approx_constant` globally. | ❌ NOT DONE |
| 14.6 | QEMU timeout (120s hard limit) | No timeout in CI script. | ❌ NOT DONE |
| 14.7 | Coverage reporting (`cargo-llvm-cov`) | No coverage in CI. | ❌ NOT DONE |
| 14.8 | Miri passes on workspace | Not run. Kernel can't run under Miri (UEFI target). | ❓ N/A |

### Plan 15 — Documentation

| # | Requirement | Evidence | Verdict |
|---|-------------|----------|---------|
| 15.1 | README status badges match reality | README claims 15 codecs, full TCP/UDP/TLS, WiFi, Bluetooth — all simulated/stubbed. | ❌ NOT DONE |
| 15.2 | `ARCHITECTURE.md` accurate | `ARCHITECTURE.md` does not exist. | ❌ NOT DONE |
| 15.3 | API docs for all 5 major crates | `cargo doc` may build but docs are sparse. | ❌ NOT DONE |
| 15.4 | `CONTRIBUTING.md` enables build in <15 min | `CONTRIBUTING.md` exists but may not have full instructions. | ⚠️ PARTIAL |
| 15.5 | `THEOREMS.md` links theorem to code + Lean proof | `docs/THEOREMS.md` exists but may not fully link. | ⚠️ PARTIAL |
| 15.6 | `cargo doc --workspace --no-deps` zero warnings | Unknown — not verified. | ❓ UNKNOWN |
| 15.7 | No claim contradicted by code | README contradicts code on network, media, wireless, language. | ❌ NOT DONE |

**Phase 7 Gate:** "All CI jobs green. README matches reality. `cargo doc` builds clean."  
**Status:** ❌ **FAIL** — README is fiction. CI has 16 jobs but many verify the fiction.

---

## Summary by Phase

| Phase | Plans | Gate | Status |
|-------|-------|------|--------|
| 1 | 01 | No deadlock, no `static mut`, no `unwrap` | ❌ FAIL |
| 2 | 02-03 | Files survive reboot, heap 10k allocs | ❌ FAIL |
| 3 | 04-06 | Aether-Lang tests pass, epsilon fuzzed | ❌ FAIL |
| 4 | 07-09 | Remote teleport loopback, DHCP in QEMU | ❌ FAIL |
| 5 | 10-11 | No `[Sim]`, draggable windows | ❌ FAIL |
| 6 | 12-13 | Shell works, invalid syscall errors | ⚠️ PARTIAL |
| 7 | 14-15 | CI green, README honest | ❌ FAIL |

**Overall: 0 of 7 phase gates pass. 6 FAIL, 1 PARTIAL.**

---

## Where the Code EXCEEDS the Spec

The spec/.agents files describe an **earlier, more primitive state** of the codebase in several areas:

1. **Memory allocator** — Spec says "bump allocator with empty dealloc." Actual code has bitmap physical allocator, slab caches, and working global heap deallocation.
2. **Network driver** — Spec says "all simulated, zero bytes hit NIC." Actual code has real e1000 MMIO driver with descriptor rings.
3. **SMP** — Spec doesn't mention AP trampoline or real mode→long mode transition. Actual code has full INIT-SIPI-SIPI with naked assembly trampoline.
4. **Security** — Spec says "no memory protection, no ASLR, flat memory." Actual code has SMAP/SMEP, ASLR PRNG, seccomp BPF, MAC LSM.
5. **Context switching** — Spec doesn't mention real context switch. Actual code has full GPR + FPU save/restore in assembly.

The codebase has evolved **beyond** what some agent plans describe. However, it has **not** evolved in the directions the spec requires (Aether-Lang completion, ManifoldFS persistence, honest driver status).

---

## The Brutal Truth

**The spec is a contract. The code breaches it in 50+ places.**

The most egregious violations:
1. **Rule 1 (No stubs):** `lang/mod.rs`, `media_player.rs`, `pkg/mod.rs`, `async_rt/mod.rs`, `http.rs`, `tls.rs` are all stubs returning fake success.
2. **Rule 2 (No overclaiming):** WiFi and Bluetooth report fake connected networks/devices instead of "not detected."
3. **Rule 4 (Safety):** `static mut AP_PER_CPU_PTR` used with `&mut *`. `unwrap()` on f64 partial_cmp. IRQ handlers acquire spinlocks without `try_lock()`.
4. **Phase 3 Gate:** Aether-Lang — the project's **defining feature** (a topological DSL where manifolds are first-class) — is entirely non-functional.
5. **Phase 2 Gate:** ManifoldFS — the project's **defining filesystem** (all data = geometry on S²) — is in-memory only with no persistence.

**This is not a working operating system. It is a kernel skeleton with working bare-metal primitives, wrapped in aspirational stubs and oversold documentation.**

---

## Recommended Priority Order to Reach Phase Gate 1

If the goal is to make the spec and reality match, start here:

1. **Fix `static mut` and IRQ deadlocks** (Plan 01) — 2 hours
2. **Remove all `[Sim]` fake data** (Plan 10 subset) — 1 hour
3. **Make `spawn_user` work** ( unblock userspace ) — 4 hours
4. **Implement ManifoldFS persistence** (Plan 03) — 2-3 days
5. **Complete Aether-Lang interpreter control flow** (Plan 04) — 1-2 days
6. **Wire TCP to actually send packets via e1000** (Plan 09 subset) — 1 day
7. **Rewrite README with honest status badges** (Plan 15) — 2 hours

Without these, the project remains a collection of partially-working demos masquerading as an OS.
