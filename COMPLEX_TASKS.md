# Epsilon-Hollow — Complexity Escalation Tracker (Phase X)

> **Rule**: No feature is too hard. Every deferred task has a design document and a concrete implementation plan.

---

## Current Phase X Status

| Phase | Status |
|-------|--------|
| Initial Attempts (Agents 1–60) | **COMPLETE** |
| Initial Attempts (Agents 61–100) | **COMPLETE** |
| Phase X Planning | **IN PROGRESS** |
| Phase X Implementation | **PENDING** |
| Phase X Verification | **PENDING** |

---

## Deferred Tasks

### CX-001: HollowLang Compiler Bootstrap (Tier 0)

**Complexity**: Multi-year effort. Requires full compiler pipeline.

**Why Deferred**: Building a self-hosting systems language from scratch exceeds the scope of a single development sprint. The kernel must be functional before migrating to a new language.

**Design Document**: `docs/design/hollowlang_bootstrapping.md` (to be written)

**Sub-tasks**:
1. Write language specification (grammar, type system, memory model)
2. Implement lexer in Rust
3. Implement recursive-descent parser
4. Implement AST -> LLVM IR codegen
5. Implement minimal stdlib (alloc, string, vec)
6. Write HollowLang compiler in HollowLang (self-host)
7. Port kernel modules incrementally

**Estimated Effort**: 12 months

---

### CX-002: SMP / Multi-Core APIC Support (Tier 2)

**Complexity**: Requires ACPI MADT parsing, per-CPU data structures, IPI, and lockless algorithms.

**Why Deferred**: The single-core scheduler works correctly. SMP introduces cache coherency, TLB shootdowns, and complex locking.

**Design Document**: `docs/design/smp_apic.md` (to be written)

**Sub-tasks**:
1. Parse ACPI RSDP -> RSDT -> MADT
2. Detect local APIC IDs for all CPUs
3. Allocate per-CPU stacks and TSS
4. Send INIT-SIPI-SIPI to APs
5. AP startup code (real mode -> protected mode -> long mode)
6. Per-CPU scheduler runqueues
7. IPI for TLB shootdown and reschedule
8. Lockless RCU for read-mostly data

**Estimated Effort**: 2 months

---

### CX-003: Full TCP Congestion Control (Tier 4)

**Complexity**: TCP is notoriously difficult (slow start, congestion avoidance, fast retransmit, SACK, timestamps).

**Why Deferred**: Minimal TCP 3-way handshake and data transfer exist. Production TCP requires extensive tuning.

**Design Document**: `docs/design/tcp_congestion_control.md` (to be written)

**Sub-tasks**:
1. Implement dynamic RTO calculation (Jacobson/Karn algorithm)
2. Implement sliding window with SACK
3. Implement slow start and congestion avoidance
4. Implement fast retransmit / fast recovery
5. Add TCP timestamps for PAWS
6. Validate with `iperf3` and `curl`

**Estimated Effort**: 1 month

---

### CX-004: TLS 1.3 Implementation (Tier 4)

**Complexity**: Cryptographic protocol requiring formal verification for safety.

**Why Deferred**: No crypto primitive library exists in the kernel yet.

**Design Document**: `docs/design/tls13_kernel.md` (to be written)

**Sub-tasks**:
1. Port or implement ChaCha20-Poly1305 and AES-GCM
2. Implement X25519 key exchange
3. Implement HKDF key derivation
4. Implement TLS 1.3 state machine
5. Formal verification of handshake state machine (TLA+)

**Estimated Effort**: 3 months

---

### CX-005: Wayland Compositor (Userspace) (Tier 4)

**Complexity**: Complex protocol with shared-memory buffers and event loops.

**Why Deferred**: Current compositor is kernel-internal and sufficient for boot demo.

**Design Document**: `docs/design/wayland_userspace.md` (to be written)

**Sub-tasks**:
1. Implement Wayland wire protocol parser
2. Implement `wl_compositor`, `wl_surface`, `wl_shell`
3. Move compositor from kernel to userspace process
4. Add DRM/KMS ioctl interface for modesetting
5. Add evdev input event reading

**Estimated Effort**: 2 months

---

### CX-006: musl libc / BusyBox Port (Tier 4)

**Complexity**: Cross-compilation toolchain and POSIX compatibility layer.

**Why Deferred**: Syscall ABI is still stabilizing.

**Design Document**: `docs/design/posix_userspace.md` (to be written)

**Sub-tasks**:
1. Stabilize syscall numbers and calling convention
2. Port musl libc syscall wrappers
3. Cross-compile BusyBox with musl
4. Build initramfs with coreutils
5. Port Bash or implement SealShell as default shell

**Estimated Effort**: 2 months

---

### CX-007: Journaling Filesystem (Tier 1)

**Complexity**: Write-ahead logging with crash recovery.

**Why Deferred**: Block layer exists but no persistent format designed.

**Design Document**: `docs/design/manifoldfs_journal.md` (to be written)

**Sub-tasks**:
1. Design superblock format for ManifoldFS-on-block
2. Implement inode table serialization
3. Implement write-ahead log (journal)
4. Implement fsck recovery tool
5. Validate with power-fail testing in QEMU

**Estimated Effort**: 1 month

---

### CX-008: Real Hardware Validation (Tier 5)

**Complexity**: Physical hardware has infinite variability.

**Why Deferred**: QEMU provides reproducible test environment.

**Design Document**: `docs/design/hardware_validation.md` (to be written)

**Sub-tasks**:
1. Create bootable USB image
2. Test on ThinkPad X230 (classic test machine)
3. Test on modern UEFI laptop
4. Test on Intel NUC (NUC8i5)
5. Document hardware compatibility list

**Estimated Effort**: 2 weeks

---

## Phase X Schedule

| Quarter | Focus |
|---------|-------|
| Q3 2026 | SMP/APIC, TCP congestion control, Journaling FS |
| Q4 2026 | TLS 1.3, musl/BusyBox port, Wayland userspace |
| Q1 2027 | HollowLang bootstrap (lexer/parser) |
| Q2 2027 | HollowLang codegen, self-hosting attempt |
| Q3 2027 | Kernel migration to HollowLang (incremental) |
| Q4 2027 | Real hardware validation, final certification |

---

## Phase X Success Criteria

1. All deferred tasks have design documents by end of Q3 2026.
2. SMP boots on `-smp 4` in QEMU with 80% scalability efficiency.
3. TCP passes `iperf3` at ≥100 Mbps in QEMU.
4. musl/BusyBox produce a working shell and coreutils.
5. HollowLang compiles a "Hello World" program.
6. Real hardware boots to desktop on at least 3 machines.

---

*Phase X orchestrated by Epsilon-Hollow Ultimate Audit Orchestrator*
*2026-05-18*
