# Repository Layout

This map is organized by build and ownership boundaries. It is more useful for
contribution work than a flat file listing because similarly named modules in
the root workspace, kernel, and future trees are not interchangeable.

## Root control files

| Path | Role |
| --- | --- |
| `Cargo.toml` | Stable root workspace membership, shared dependencies, and profiles |
| `Cargo.lock` | Locked dependency graph for the root workspace |
| `rust-toolchain.toml` | Root stable toolchain selection |
| `mkdocs.yml` | Root MkDocs navigation for this system documentation |
| `deny.toml` | Dependency license, source, duplicate, and ban policy |
| `.github/workflows/ci.yml` | CI job graph and release/proof automation |
| `README.md` | Project overview and claim ledger; useful entry point, not the only source of truth |
| `SECURITY.md` | Vulnerability reporting and security policy |

## `kernel/seal-os`: the bootable kernel

`kernel/seal-os/Cargo.toml` has its own workspace declaration. Treat this
directory as a standalone kernel project.

| Directory | Responsibility |
| --- | --- |
| `src/boot` | UEFI entry, boot information, AP trampoline, early boot setup |
| `src/cpu` | CPU and per-CPU/SMP state |
| `src/memory` | Physical allocator, virtual memory, page tables, heap, slab, swap, TopoRAM |
| `src/process` | Tasks, scheduler, context switching, ELF loader, userspace, signals |
| `src/syscall` | Seal ABI dispatcher and specialized pipe/time/ioctl syscall paths |
| `src/fs` | VFS, block abstractions, ManifoldFS, FAT, ext2, procfs, sysfs, pipes, encoding |
| `src/drivers` | ACPI, APIC, PCI, storage, GPU, USB, audio, network, entropy, RTC, watchdog |
| `src/net` | Protocol implementations and packet state |
| `src/security` | SMAP/SMEP, KPTI, ASLR, MAC/ACL, seccomp, audit, passwords/shadow/groups, TopCrypt guard |
| `src/graphics` | Framebuffer, fonts, console, splash, login, HTEK drawing, topology renderer |
| `src/wm` | Window state, compositor, desktop, taskbar, cursor, menus, input events |
| `src/lang` | Aether interpreter embedding and kernel callback registration |
| `src/apps` | Shell, terminal, installer, settings, games, viewers, benchmark apps |
| `src/pkg` | Manifold package metadata, registry, resolver, carrier format |
| `src/async_rt` | Kernel async task, timer, channel, and I/O scaffolding |
| `src/ml_engine` | Kernel-side topology/ML visualization and assembly-adjacent paths |
| `src/sync` | Kernel synchronization primitives and tests |
| `src/testing` | In-kernel test harness and test runner |

## `kernel/seal-mkimage`: image and proof tooling

This stable host crate builds the raw image from the kernel EFI artifact. It
also contains the audit CLI used by CI and VM runbooks. `src/aether_build.rs`
is the current Aether source discovery/bootstrap-check path; `src/main.rs`
contains image construction and repository proof gates.

The image tool is not the kernel. A passing image verifier proves image layout
and headers, not that the kernel reaches the desktop.

## Root stable workspace crates

| Path | Crate role |
| --- | --- |
| `kernel/epsilon/epsilon/crates/aether-core` | Shared topology, manifold, theorem-adjacent, memory, and ML primitives |
| `kernel/epsilon/epsilon/crates/epsilon` | Epsilon context-transfer research library |
| `kernel/epsilon/epsilon/crates/epsilon-os` | Host/user-space Epsilon runtime surface |
| `kernel/aether/aether-link` | I/O/prefetch research crate with examples and benches |
| `kernel/aether/aether-verified` | Rust theorem checkers and proof-facing helpers |
| `kernel/aether/Aether-Lang/crates/aether-lang` | Lexer, parser, AST, interpreter, bytecode, VM, render/export helpers |
| `kernel/aether/Aether-Lang/crates/aether-cli` | Host CLI for Aether scripts |
| `kernel/aether/Aether-Lang/crates/aegis-core` | AEGIS memory and ML experiments |
| `kernel/aether/Aether-Lang/crates/aegis-cli` | Host CLI for AEGIS runtime experiments |
| `kernel/aether/Aether-Lang/crates/repl-core` | Shared REPL/runner support |
| `tools/ubuntu-alloc-bench` | Native Ubuntu allocator comparison harness |

## Aether documentation boundary

The nested MkDocs site is defined by
`kernel/aether/Aether-Lang/mkdocs.yml` and starts at
`kernel/aether/Aether-Lang/docs/index.md`. It documents Aether-Lang itself:
syntax, runtime surface, concepts, architecture, examples, backends, and
language-specific evidence policy. Link to it when the question is about the
language; use this root site when the question is about the complete system.

## Proof and research trees

| Path | Status |
| --- | --- |
| `kernel/aether/aether-verified/lean` | Lean formalization and theorem modules; formal strength varies by theorem |
| `docs/` | Root system, build, VM, theorem, security, design, and research documentation |
| `docs/superpowers/plans` | Historical/active implementation plans, not runtime sources |
| `tests/` | Host Python tests and GPU tests; do not confuse them with kernel integration tests |
| `experiments/` | Experimental material outside the stable kernel contract |
| `infrastructure/` | Host orchestration, training, deployment, and tooling material |
| `apps/` | The LAAMBA Governor application and its own Rust/TypeScript/Python surfaces |
| `future/apeiron-runtime` | Future runtime/application tree excluded from the root workspace |

## Where to make a change

| Change | Start with |
| --- | --- |
| Boot order or proof marker | `kernel/seal-os/src/lib.rs`, `docs/boot-and-runtime.md`, relevant `seal-mkimage` gate |
| Memory allocation or O(1) claim | `kernel/seal-os/src/memory/`, `docs/MEMORY.md`, `docs/TOPOLOGICAL_OS_CONTRACT.md`, matching benchmark |
| Filesystem behavior | `kernel/seal-os/src/fs/`, `docs/MANIFOLDFS.md`, `docs/SYSCALLS.md` if ABI-visible |
| Syscall | `kernel/seal-os/src/syscall/table.rs`, specialized syscall module, `docs/SYSCALLS.md`, user proof/test |
| Aether language behavior | `kernel/aether/Aether-Lang/crates/aether-lang/`, nested Aether docs, language tests |
| Theorem or proof status | `aether-core`, `aether_verified`, Lean source, `docs/THEOREMS.md` |
| Image or VM proof | `kernel/seal-mkimage/`, `scripts/`, `docs/VM_RUNBOOK.md`, `docs/CI.md` |

## Generated and evidence artifacts

Build output under `target/` and VM artifacts under kernel release/proof
directories are evidence products, not source. Keep an artifact tied to the
commit, image, command, and environment that produced it. An old serial log or
screenshot is not current proof merely because it contains the right words.
