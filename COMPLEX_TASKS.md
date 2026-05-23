# Complexity Escalation Phase (Phase X) — Deferred Tasks

## Philosophy
No feature remains "too hard". Every deferred task gets:
1. Research plan (existing crates, Linux code, papers)
2. Design doc in `docs/design/complex_<task>.md`
3. Atomic sub-tasks (each ≤1 day)
4. Incremental implementation with tests

---

## CX-1: NVMe Driver (Agent 25)

**Complexity:** PCIe MSI-X, admin SQ/CQ, I/O SQ/CQ, PRP lists, doorbell registers.
**Why deferred:** Queue pair management + DMA PRP list construction is ~500 lines of精密 register work.
**Research:**
- NVMe 1.4 spec (chapter 3-6)
- `nvme_driver` crate (Rust embedded)
- Linux `drivers/nvme/host/pci.c`
**Sub-tasks:**
- [ ] Define register structs (CAP, VS, CC, CSTS, ASQ, ACQ)
- [ ] Allocate aligned admin SQ/CQ in physically contiguous memory
- [ ] Submit Identify Controller / Identify Namespace commands
- [ ] Build PRP list for 4KB page scatter-gather
- [ ] Create I/O completion queue with MSI-X interrupt vector
- [ ] Benchmark: `fio` randread 10M target >10k IOPS

---

## CX-2: USB xHCI (Agent 26)

**Complexity:** MMIO extended capabilities, port routing, endpoint context, transfer rings.
**Why deferred:** USB descriptor parsing + HID report decoding is state-machine heavy.
**Research:**
- xHCI spec 1.2 (chapters 4-6)
- `usb-host` crate (Rust no_std)
- Linux `drivers/usb/host/xhci.c`
**Sub-tasks:**
- [ ] Read extended capabilities (USBLEGCTL, XECP)
- [ ] Initialize DCBAAP, CRCR command ring
- [ ] Port reset → enable → slot assignment
- [ ] Address Device command
- [ ] Configure Endpoint for HID interrupt IN
- [ ] Parse HID report descriptor for keyboard scancodes
- [ ] Type "hello" in shell as end-to-end test

---

## CX-3: HDA Audio Driver (Agent 30)

**Complexity:** Codec verb negotiation, DMA buffer descriptors, PCM ring buffer.
**Why deferred:** Audio requires real-time buffer management + codec discovery graph.
**Research:**
- Intel HDA spec (chapter 3: codec, chapter 4: controller)
- `hda-driver` (Redox OS, MIT license)
- Linux `sound/pci/hda/hda_intel.c`
**Sub-tasks:**
- [ ] Reset HDA controller, read CORB/RIRB capabilities
- [ ] Send GET_PARAM verbs to codec to discover widgets
- [ ] Find DAC (pin 0x02) and configure output path
- [ ] Allocate DMA buffer for 48kHz stereo 16-bit
- [ ] Program stream descriptor with buffer address
- [ ] Play sine wave generated in kernel
- [ ] `aplay` equivalent in userspace

---

## CX-4: musl libc Port + Cross-Compilation Toolchain (Agents 31, 33, 34, 52)

**Complexity:** musl needs ~300 syscall wrappers, startup files (crt1.o), linker scripts.
**Why deferred:** Requires a working cross-compiler (x86_64-seal-linux-musl-gcc) or Rust-based libc.
**Research:**
- musl libc `arch/x86_64/syscall_arch.h`
- `rustix` crate (no_std syscall wrappers in Rust)
- Use `rustc` with custom target JSON for userspace
**Sub-tasks:**
- [ ] Define `x86_64-seal-os.json` target spec (no red-zone, soft-float, no-default-libs)
- [ ] Write `crt0.S` — stack setup, call main, exit syscall
- [ ] Implement 20 most common syscalls in Rust (`write`, `read`, `open`, `close`, `mmap`, etc.)
- [ ] Compile `hello.c` with `clang --target=x86_64-seal-os` and run under kernel
- [ ] Port BusyBox config for seal-os (disable features requiring missing syscalls)
- [ ] Port Bash 5.2 (minimal readline + job control)
- [ ] Build script `scripts/build_userspace.sh` produces ELF binaries + `.eph` packages

---

## CX-5: Dynamic Linker (Agent 32)

**Complexity:** ELF PT_DYNAMIC parsing, PLT/GOT relocation, symbol hash table.
**Why deferred:** Needs working userspace C toolchain first (produces `.so` files).
**Research:**
- ELF gABI (chapter 2: dynamic linking)
- `ld.so` source (glibc `elf/` directory)
- `mold` linker source for relocation types
**Sub-tasks:**
- [ ] Parse `.dynamic` section (DT_NEEDED, DT_HASH, DT_STRTAB, DT_SYMTAB)
- [ ] Process `R_X86_64_RELATIVE`, `R_X86_64_GLOB_DAT`, `R_X86_64_JUMP_SLOT`
- [ ] Load shared libraries from `/lib` in dependency order
- [ ] Implement `dlopen`, `dlsym`, `dlclose` syscalls
- [ ] Test: shared library with global variable, executable reads it correctly

---

## CX-6: pthread + ThreadSanitizer (Agent 38)

**Complexity:** `clone()` flags (CLONE_VM, CLONE_FS, CLONE_FILES), futex emulation.
**Why deferred:** Needs userspace libc + TLS (thread-local storage) support.
**Research:**
- Linux `clone(2)` man page
- `futex(2)` emulation using kernel wait queues
- `pthread` implementation in musl (`src/thread/`)
**Sub-tasks:**
- [ ] Implement `clone()` syscall with stack pointer argument
- [ ] Set up `%fs` base for TLS (arch_prctl equivalent)
- [ ] Implement `futex_wait` / `futex_wake` in kernel
- [ ] Build `pthread_create`, `pthread_join`, `pthread_mutex_lock` in userspace
- [ ] ThreadSanitizer pass on 10-thread counter test

---

## CX-7: Secure Boot Shim (Agent 58)

**Complexity:** UEFI PE/COFF signature parsing, PKCS#7 verification, embedded public key.
**Why deferred:** Cryptographic verification of PE binaries requires ASN.1 + RSA/ECDSA.
**Research:**
- UEFI spec (chapter 32: secure boot)
- `sbsigntools` source
- `minicrypto` or `rsa` crate for no_std
**Sub-tasks:**
- [ ] Embed ed25519 public key in UEFI shim binary
- [ ] Parse PE data directory for certificate table
- [ ] Verify signed kernel hash against embedded key
- [ ] Reject unsigned kernel with error message
- [ ] Chain-load verified kernel via ExitBootServices

---

## CX-8: ext4 Driver (Agent 11)

**Complexity:** Extents, journal replay, checksums, large directory hashing.
**Why deferred:** Full ext4 is ~10k lines; read-only ext2 exists but ext4 needs extents.
**Research:**
- `rust-ext4` crate (if no_std compatible)
- Linux `fs/ext4/` — especially `ext4_extents.c`
- Write simple extent tree parser first
**Sub-tasks:**
- [ ] Parse ext4 superblock (s_inodes_count, s_blocks_count, s_feature_incompat)
- [ ] Read inode via extent tree (i_block as ext4_extent_header)
- [ ] Implement directory traversal (htree or linear)
- [ ] Add journal replay for crash safety (optional v2)
- [ ] Mount ext4 at `/mnt/ext4`, pass `fsck` after write

---

## Phase X Process

1. Pick one CX task
2. Write `docs/design/complex_<task>.md`
3. Implement sub-tasks one at a time
4. Each sub-task: compile → unit test → integration test → commit
5. When all sub-tasks done, mark agent ✅ in TASKS.md
6. Repeat until no ❌ or 🔄 remain

**Current Phase X queue:** CX-1, CX-2, CX-3, CX-4, CX-5, CX-6, CX-7, CX-8
