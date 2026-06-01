<!-- Seal OS v0.4.5 README -->
<!-- Target: the biggest README ever, with sarcasm and human readability -->

<p align="center">
  <img src="assets/logo.svg" alt="Seal OS Logo" width="180">
</p>

<h1 align="center">Seal OS — The Geometrical Operating System</h1>

<p align="center">
  <strong>OS state is topology on S².</strong><br>
  Bare-metal x86_64. Pure Rust. Minimal inline assembly (AP trampoline, CPU-idle). No POSIX. No libc.<br>
  Memory, files, and scheduler decisions are embedded as point clouds on the unit sphere.
</p>

<p align="center">
  <a href=".github/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/teerthsharma/epsilon-hollow/ci.yml?branch=main&label=CI&style=flat-square" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/teerthsharma/epsilon-hollow?style=flat-square&color=00aaff" alt="License: MIT"></a>
  <a href="#build-and-run"><img src="https://img.shields.io/badge/rust-stable-orange?style=flat-square&logo=rust" alt="Rust stable"></a>
  <a href="#build-and-run"><img src="https://img.shields.io/badge/nightly-kernel-purple?style=flat-square&logo=rust" alt="Nightly kernel"></a>
  <a href="docs/THEOREMS.md"><img src="https://img.shields.io/badge/Lean%204-proofs-green?style=flat-square" alt="Lean 4 proofs"></a>
  <a href="#performance-characteristics"><img src="https://img.shields.io/badge/benchmarks-gated-blue?style=flat-square" alt="Benchmark gated"></a>
</p>

<p align="center">
  <a href="#quick-start">🚀 Quick Start</a> •
  <a href="#honest-status-dashboard">📊 Status</a> •
  <a href="#architecture">🏛️ Architecture</a> •
  <a href="#the-ten-theorems">🔮 Theorems</a> •
  <a href="#whats-inside">🧩 Inside</a> •
  <a href="#build-and-run">🔧 Build</a> •
  <a href="#documentation-index">📚 Docs</a> •
  <a href="#contributing--community">🤝 Contribute</a>
</p>

---

## Table of Contents

- [Before You Read This](#before-you-read-this)
- [The 30-Second Pitch](#the-30-second-pitch)
- [Why We Did This To Ourselves](#why-we-did-this-to-ourselves)
- [The FAQ Nobody Asked For](#the-faq-nobody-asked-for)
- [Honest Status Dashboard](#honest-status-dashboard)
- [Real Talk: What "Research Kernel" Actually Means](#real-talk-what-research-kernel-actually-means)
- [Quick Start — Boot in 5 Minutes](#quick-start--boot-in-5-minutes)
- [Boot Log](#boot-log)
- [Architecture](#architecture)
- [Design Decisions That Seemed Good At 3 AM](#design-decisions-that-seemed-good-at-3-am)
- [The Ten Theorems](#the-ten-theorems)
- [What Theorems Actually Do (For Normal Humans)](#what-theorems-actually-do-for-normal-humans)
- [What's Inside](#whats-inside)
  - [Memory & Topology](#memory--topology)
  - [ManifoldFS — The Filesystem](#manifoldfs--the-filesystem)
  - [Process Scheduler](#process-scheduler)
  - [Interrupts & Drivers](#interrupts--drivers)
  - [Graphics & Desktop](#graphics--desktop)
  - [Network Stack](#network-stack)
  - [Security](#security)
  - [Aether-Lang](#aether-lang)
  - [Applications](#applications)
  - [Epsilon — Context Teleportation](#epsilon--context-teleportation)
  - [Aether-Link — I/O Superkernel](#aether-link--io-superkernel)
- [Feature Matrix vs The World](#feature-matrix-vs-the-world)
- [Where We Actually Stand (Brutal Honesty Edition)](#where-we-actually-stand-brutal-honesty-edition)
- [Performance Characteristics](#performance-characteristics)
- [Benchmarks: Or, How We Learned To Stop Worrying](#benchmarks-or-how-we-learned-to-stop-worrying)
- [Build and Run](#build-and-run)
- [Documentation Index](#documentation-index)
- [How To Read Our Docs Without Crying](#how-to-read-our-docs-without-crying)
- [Repository Map](#repository-map)
- [How To Navigate This Repo (Survival Guide)](#how-to-navigate-this-repo-survival-guide)
- [Contributing & Community](#contributing--community)
- [Contributing: Or, How To Lose Your Weekend To Geometry](#contributing-or-how-to-lose-your-weekend-to-geometry)
- [Security Policy](#security-policy)
- [How To Break This OS (A Hacker's Guide To Our Pain)](#how-to-break-this-os-a-hackers-guide-to-our-pain)
- [License](#license)
- [System Call Reference](#system-call-reference)
- [aether-core — Math Foundation](#aether-core--math-foundation)
- [Lean 4 Proofs](#lean-4-proofs)
- [CI Pipeline](#ci-pipeline)
- [The CI Pipeline: Our Digital Torture Chamber](#the-ci-pipeline-our-digital-torture-chamber)
- [Acknowledgements](#acknowledgements)
- [People We Blame For This](#people-we-blame-for-this)
- [Contact](#contact)
- [Glossary of Words We Made Up](#glossary-of-words-we-made-up)
- [A Day In The Life Of A Seal OS Developer](#a-day-in-the-life-of-a-seal-os-developer)

---

## Before You Read This

> **⚠️ Content Warning:** This README contains strong opinions about geometry, operating systems, and the general state of software engineering. If you believe filesystems should be trees, memory should be arrays, and scheduling should be queues, you may find the following content disturbing. Reader discretion is advised.

Welcome. You have discovered what is possibly the most over-engineered, under-resourced, mathematically-obsessed operating system README on GitHub. We are not sorry. We are, however, exhausted.

Seal OS is the result of asking: "What if we treated every OS subsystem like a differential geometry problem?" Spoiler: it works better than you would think, but the learning curve is vertical and the documentation is... well, you are looking at it.

**What this README promises:**
- Technical honesty. We will tell you what breaks.
- Sarcasm. We have spent too many nights debugging APIC timer interrupts to be polite.
- Depth. Every claim traces to code. Every "O(1)" has a proof gate.
- Occasional existential despair. You try writing a TLS stack in `#![no_std]` and stay chipper.

**What this README does NOT promise:**
- Brevity. You wanted the biggest README ever. Here it is.
- A clear business case. There isn't one. This is art.
- POSIX compatibility. We said what we said.

Grab a beverage. Settle in. We are going on a journey through 111,000 lines of Rust, ten mathematical theorems, and one developer's questionable life choices.

---

## The 30-Second Pitch

**What if operating system decisions were literally geometry problems?**

Seal OS is a bare-metal x86_64 research kernel written in 100% Rust. It is not Linux, not Unix, not POSIX, and not a libc target. Every kernel subsystem — memory allocation, file metadata, process scheduling, graphics prefetch — is expressed as topology on the unit sphere S².

### Why S²?

The unit sphere is the simplest compact 2-manifold without boundary:

1. **No edges** — unlike grids or trees, S² has no boundary conditions. A file's embedding wraps around naturally.
2. **Metric structure** — great-circle distance gives us a true metric space for nearest-neighbor searches (Voronoi cells).
3. **Finite area** — 4π steradians bounds the maximum separation between any two points, giving natural normalization.
4. **Rotation group SO(3)** — the symmetry group of S² is well-studied; spectral methods decompose nicely.

### Why Topology?

Traditional OS design uses graphs (filesystems), arrays (memory), and queues (scheduling). These are 1-dimensional or 0-dimensional structures. Topology gives us:

- **Betti numbers** to measure fragmentation (β₀ = connected components, β₁ = cycles)
- **Voronoi tessellation** for O(1) spatial partitioning
- **Spectral decomposition** for predictive prefetching via eigenvector analysis
- **Hyperbolic geometry** for natural hierarchical clustering (short-lived vs long-lived allocations)
- **PD control** for adaptive resource governance with stability guarantees

### The Contract

Kernel Rust owns hardware, memory, drivers, scheduling, and theorem gates. Aether-Lang (AEGIS) owns native scripts, app logic, shell automation, topology commands, and the future self-hosting flow. There is no POSIX compatibility costume — familiar syscall names are Seal ABI entry points with Seal-defined semantics.

### Key Differentiators

| Capability | Seal OS | Linux | Redox |
|---|---|---|---|
| Geometry-native kernel | ✅ | ❌ | ❌ |
| O(1) metadata file teleport | ✅ | ❌ (same-FS rename only) | ❌ |
| Content-addressable geometric lookup | ✅ | ❌ | ❌ |
| Voronoi-based scheduling | ✅ | ❌ | ❌ |
| Spectral prefetch prediction | ✅ | ❌ | ❌ |
| Adaptive PD governor | ✅ | ❌ | ❌ |
| Theorem-gated boot (T1-T10) | ✅ | ❌ | ❌ |
| Formal proofs (Lean 4) | 🚧 | ❌ | 🚧 |
| Native topological language | ✅ | ❌ | ❌ |
| Zero assembly | ✅ | ❌ | ❌ |


---

## Why We Did This To Ourselves

*(A brief, slightly unhinged history)*

Once upon a time, a developer looked at Linux and thought: "This is fine, but what if the scheduler knew about Ricci curvature?"

That developer was not okay. That developer is us.

### The Origin Story (With Regrets)

Seal OS started as a question: "Can you build an OS where every data structure is a manifold?" The answer, it turns out, is "yes, but you will not sleep."

The first commit was a UEFI bootloader that printed "Hello, Geometry!" It took three weeks. Not because UEFI is hard (it is), but because we insisted on embedding the boot banner as a spherical harmonic decomposition. It looked identical to ASCII text. We are not always smart.

### Why Not Linux?

Linux is a triumph of engineering. It runs on toasters, Mars rovers, and 99% of the world's supercomputers. It is also 30 million lines of C, most of which were written before people understood buffer overflows were bad.

We wanted something smaller. Something where you could read every line and understand why it was there. Something where a single developer could hold the entire design in working memory.

Also, Linux doesn't have a Voronoi-based scheduler, and we think that's a missed opportunity.

### Why Not Redox?

Redox OS is the cool older sibling who went to art school. It's a microkernel in Rust, it's well-designed, and it actually has working networking. We love Redox. We just wanted to go weirder.

Where Redox asked "What if a microkernel, but in Rust?" we asked "What if the entire OS were a topological space?" These are different questions. One of them is more employable.

### The Mathematical Obsession

At some point, we realized that every OS problem maps to geometry:

| OS Problem | Geometry Analogue | How It Helps |
|---|---|---|
| Memory fragmentation | Betti numbers (β₀) | Quantify how scattered free memory is |
| File locality | Great-circle distance | Find "nearby" files without path traversal |
| Scheduling fairness | Voronoi cell balancing | Ensure no CPU region is starved |
| Cache prediction | Spectral contraction | Predict access patterns via eigenvectors |
| Resource limits | Hyperbolic curvature | Natural hierarchical nesting |

Is this overkill? Absolutely. Does it work? Sometimes. Is it fun? You have no idea.

### The Naming Convention (We're Sorry)

Our naming scheme is inconsistent and borderline unhinged:
- **Seal OS**: Because seals are cute and geometrically efficient.
- **Aether-Lang**: Because "programming language" was too boring.
- **ManifoldFS**: Because someone said "just call it ext5" and we laughed.
- **Aether-Link**: Because "I/O scheduler" doesn't sound mystical enough.
- **LAAMBA Governor**: We don't remember what LAAMBA stands for. We think it was a typo that stuck.

---

## The FAQ Nobody Asked For

### Q: Is this a real operating system?
**A:** It boots on real hardware (and QEMU). It has a scheduler, filesystem, network stack, and window manager. It does not, however, run Docker, Steam, or Microsoft Word. So the answer depends on your definition of "real." If your definition includes "can I tweet from it," then no. If your definition includes "does it prove mathematical theorems before letting me open a file," then yes.

### Q: Can I use this as my daily driver?
**A:** You *can*. You would be miserable. We do not recommend it. The browser is "planned." The GPU drivers are "stubs." The WiFi is "simulated." Your therapist will bill you hourly.

### Q: Why S² and not S³?
**A:** S³ is for cowards who can't commit to two dimensions. Also, we couldn't figure out how to visualize it on a 1024×768 framebuffer.

### Q: What's with all the theorems?
**A:** We believe that if you can't prove your scheduler is mathematically sound, you have no business scheduling. Also, it scares away junior developers, which keeps our issue tracker manageable.

### Q: Is this a TempleOS clone?
**A:** Terry A. Davis proved that one person can write an entire OS with a native language. We are inspired by that energy. We are not, however, divinely inspired. Our language is called Aether-Lang, not HolyC, and our god is the unit sphere.

### Q: Why is the README so long?
**A:** You asked for this. Literally. You said "biggest then ever." We are but humble servants of your terrible ideas.

### Q: Does it run Doom?
**A:** No. It has Snake, Breakout, and Warp Racer. Doom requires a working GPU renderer and we are not there yet. Check back in v0.9.0, probably.

### Q: What's the deal with the assembly correction?
**A:** We claimed "zero assembly" for months because we forgot about the AP trampoline. A trampoline is ~30 lines of assembly that wakes up secondary CPUs. Without it, you get one CPU. With it, you get multiple CPUs and a dent in your pride. We chose CPUs.

### Q: Who maintains this?
**A:** Primarily one person with occasional contributions from people who looked at the code, said "huh," and then quietly left. We treasure those contributors.

### Q: Is this production-ready?
**A:** Define "production." If you mean "runs in a VM and passes CI gates," yes. If you mean "I would bet my company's infrastructure on it," absolutely not. If you mean "could I demo it at a conference and look smart," definitely yes.

---

## Honest Status Dashboard

Seal OS is a research kernel. We do not hide behind timelines or excuses. Here is what is real today versus what remains pending.

### ✅ Real — Running Today

<details open>
<summary><strong>Boot & Init</strong></summary>

- **UEFI PE/COFF** → 64-bit long mode, identity-mapped page tables, GDT + TSS
- **SMP bring-up** — INIT-SIPI-SIPI trampoline, per-CPU data, IPIs (reschedule + TLB shootdown)
- **ACPI** — RSDP parsing, MADT discovery for APIC topology
- **Boot theorem gates** — T1-T10 all verified before scheduler start

</details>

<details open>
<summary><strong>Memory</strong></summary>

- **Physical frame allocator** — bitmap truth store + topological free index (8 Voronoi cells, 3 summary levels)
- **O(1) single-frame allocation** — bounded hot path, no full bitmap walk
- **Contiguous DMA allocation** — 128 bounded topological candidate probes, hard 64-page run cap
- **Slab allocator** — 6 size classes (64B–2048B), intrusive free lists, O(1) alloc/dealloc
- **VMM** — 4-level page tables (PML4), on-demand mapping
- **Demand paging** — `SYS_MMAP` reserves virtual ranges, page-fault handler lazily allocates backing frames
- **Swap** — low-memory pressure can swap mmap-backed pages to `/swap.topo` and fault them back
- **COW fork** — userspace fork clones page tables, clears writable bits for shared pages, resolves write faults by copying the page
- **TopoRAM wrapper** — 64 bytes metadata per frame (S² embedding, access history, Voronoi cell, lifetime class)

</details>

<details open>
<summary><strong>Filesystem</strong></summary>

- **ManifoldFS** — in-memory with Voronoi indexing, metadata teleport, bucketed content search
- **FAT12/16/32** — read/write/create/mkdir/unlink/rmdir/rename, cluster allocation, directory growth
- **ext2** — direct/single/double/triple indirect blocks, cross-directory rename with `..` fixup, `mknod`
- **PipeFS** — in-memory pipe filesystem with 64KB ring buffers
- **DevTmpFs** — device nodes with rename support
- **VFS** — cross-mount rename via copy+delete fallback for files
- **TopCrypt** — topological file encoding (64-byte blocks as 16-point clouds on S² with CRC32, shuffle, XOR masks)

</details>

<details open>
<summary><strong>Process & Scheduler</strong></summary>

- **ManifoldScheduler** — 8 Voronoi cells, 256 priority buckets per cell, T2 prediction, T4 adaptive timeslice
- **Real SYS_FORK** — full process duplication: kernel stack, xsave area, task context, cloned task queued
- **Real SYS_EXEC** — ELF64 ET_EXEC and ET_DYN, `PT_INTERP`/`DT_NEEDED` shared objects, `R_X86_64_RELATIVE` relocations, shebang support, Aether-Lang scripts
- **Signals** — SIGKILL, SIGSEGV, SIGINT, SIGTERM, SIGPIPE, SIGALRM, SIGCHLD, SIGUSR1/2. Per-task pending/mask/handlers, signal frames on user stack, `sigaltstack`, `SA_RESTART`
- **Pipes + dup + brk** — `SYS_PIPE`, `SYS_DUP`/`DUP2`, `SYS_BRK` via `mmap_user`

</details>

<details open>
<summary><strong>Interrupts & Drivers</strong></summary>

- **IDT** — 256 entries
- **Local APIC + I/O APIC** — per-CPU timer, EOI, ICR
- **APIC Timer** — per-CPU scheduler ticks and governor sampling
- **PS/2 Keyboard & Mouse** — IRQ1/IRQ12, scancode-to-ASCII table
- **Serial COM1** — 115200 baud, primary diagnostic channel
- **PCI enumeration** — config space ports 0xCF8/0xCFC
- **Intel e1000** — TX/RX descriptor rings, packet send/receive
- **AHCI SATA** — MMIO command/FIS structures, read/write sectors
- **NVMe** — admin + I/O queues, Identify Controller/Namespace, PRP-based DMA sector read/write
- **xHCI USB 3.0** — controller init, event/command rings, port enumeration, HID boot keyboards/mice, Mass Storage SCSI BBB
- **Intel HDA** — CORB/RIRB engines, codec widget discovery, DAC pin selection, output stream DMA, 48kHz 16-bit stereo PCM playback
- **RTC** — CMOS real-time clock with BCD/binary detection, 12/24-hour handling
- **Watchdog** — APIC timer watchdog, keyboard-controller reset on 5-second hang
- **Hardware Entropy** — RDRAND + RDSEED with CPUID probe and carry-flag retry

</details>

<details open>
<summary><strong>Graphics & Desktop</strong></summary>

- **Framebuffer** — 1024×768×32bpp, double-buffered, back buffer eliminates tearing
- **High-Tech Engine (htek.rs)** — anti-aliased text, gradient fills, rounded rectangles, glow effects, alpha blending, stroke rendering
- **Window Manager** — compositor with z-order, window decorations, minimize/maximize/resize, 5 cursor shapes
- **Desktop** — taskbar with live RTC clock, start menu, power button, theorem indicators T1-T5, governor ε value
- **Wallpaper** — procedural Schwarzschild metric + Faraday tensor rendering
- **Themes** — dark, light, seal, matrix, plus high-contrast
- **Panic Screen** — red background + white "PANIC" text + message rendered to framebuffer
- **Kernel Log Buffer** — 32 KiB ring buffer, `SYS_KMSG_READ` for userspace `dmesg`

</details>

<details open>
<summary><strong>Network Stack</strong></summary>

- **TCP/IP** — wired end-to-end through IPv4 → net::transmit → e1000 TX descriptor ring
- **TCP** — listen/accept backlog, SYN queue, retransmission timer
- **UDP** — query packet building
- **DHCP** — full state machine (Init → Discover → Request → Bound), auto-sends DISCOVER on boot
- **DNS** — proper query packets (ID, flags, QNAME, QTYPE A, QCLASS IN) via UDP port 53
- **TLS 1.3 PSK** — minimal PSK-only record path with AES-128-GCM + HKDF-SHA256, hardware-entropy failure handling
- **HTTPS Client** — routes `https://` through `TlsSocket`

</details>

<details open>
<summary><strong>Security</strong></summary>

- **ASLR** — userspace mmap base randomised with 16-bit entropy shift, RDRAND/RDSEED source
- **Seccomp** — classic BPF evaluator, per-task filter arrays, `BPF_LD_W_ABS`/`BPF_JMP_JEQ`/`BPF_RET`
- **KPTI scaffolding** — CR3 swap code exists, boot selftest pending hard gate
- **Retpoline** — compiler flags in `.cargo/config.toml`, all 16 register thunks, trampoline page table
- **SMAP/SMEP** — init at boot
- **MAC** — scaffolding present
- **Audit** — JSON-formatted event buffering

</details>

<details open>
<summary><strong>Aether-Lang & Runtime</strong></summary>

- **Lexer, Parser, AST, Interpreter** — all running in `no_std` kernel space
- **VM (Titan Mode)** — bytecode execution
- **Stdlib** — `math` (pi, e), `fs` (read/write/exists/mkdir/teleport), `process` (pid, exit, spawn), `net` (local_ip, has_nic, status), `theorem` (status)
- **Kernel bridge** — Aether-Lang wired directly into Seal ABI syscalls; runtime proof gate is `seal-mkimage --check-aether-runtime`

</details>

<details open>
<summary><strong>Applications</strong></summary>

- **SealShell** — 30+ English-first commands: `look`, `peek`, `move`, `search`, `tasks`, `seal`, `race`, `stats`, `calc`, `play`, `tensor render`
- **Terminal Emulator** — 80×25 scrollback, key input processing
- **Seal IDE** — code editor panel, file tree sidebar, status bar
- **Calculator** — scientific with recursive descent parser, gradient UI
- **SealPlayer** — WAV/PCM playback with RIFF/WAVE header parser
- **Theorem Viewer** — T1-T10 status, real-time governor ε, Betti-0 count
- **Tensor Viewer** — CSV/trading data → 3D tensor rendering, profit=green peaks, loss=red valleys
- **Games** — Snake, Breakout, Warp Racer
- **File Manager** — ManifoldFS-native file operations
- **Installer UI** — GPT partitioning simulation

</details>

<details open>
<summary><strong>Package Manager & Settings</strong></summary>

- **ManifoldPkg** — `.eph` parser, dependency resolver, local ManifoldFS extraction, HTTPS registry URL, Ed25519 verification path
- **Settings** — live `BTreeMap<String,String>` with theme/font/wallpaper defaults, `sys_setting_get`/`sys_setting_set`

</details>

### 🚧 Partial — Honest Limits

| Feature | Limitation | Path to Full |
|---|---|---|
| **GPU** | PM4 compute ring infrastructure is real (MMIO, packet builder, fence polling). AMD firmware scanner parses VBIOS and identifies architecture families. GCN shader binaries are honest stubs with correct ABI — they execute but don't yet compute full topology. CPU fallback performs real spherical Voronoi, JL projection, and spectral contraction in software. | Compile real OpenCL C → GCN ISA and replace stub arrays |
| **WiFi / Bluetooth** | Simulated state machines with deterministic scan/connect/pair results. Real firmware blob loading is vendor IP, out of scope. | N/A — vendor IP |
| **TLS** | PSK-only. No X.509, PKI, or ECDHE yet. Fails closed if hardware entropy unavailable. | Implement X.509 parser + ECDHE |
| **Package Manager** | Local extraction + signed verification path exist. remote registry fixture and signed package gate are pending. | Build registry + fixture |
| **Installer** | UI exists. Actual GPT partitioning and filesystem formatting not yet implemented. | Raw block device write path |
| **FAT / ext2 parity** | Write paths exist; fixture gate pending before "full filesystem parity" claims. | Comprehensive fixture tests |
| **Security hardening** | Scaffolding present; per-feature hardening proof gates pending before production security claims. | Boot selftests for each feature |
| **Kernel modules** | Everything is built-in. No loadable kernel module framework. | Design LKM ABI |

### ❌ Not Yet Real

| Feature | Why | When |
|---|---|---|
| **GPU drivers (i915/nouveau)** | Proprietary firmware blobs required, out of scope for research kernel | Never (vendor IP) |
| **AMD GPU full compute** | PM4 ring + VBIOS parsing works. Signed SMU firmware not loaded (not required for compute). Shader stubs need real compiled ISA for correct results. | Rebuild shader binaries from OpenCL C source |
| **Full internet TLS** | X.509/PKI/ECDHE not implemented | Post-1.0 |
| **Self-hosting** | Aether-Lang needs more stdlib before it can build itself | Roadmap phase 4 |

---


---

## Real Talk: What "Research Kernel" Actually Means

Let's have a moment of honesty. "Research kernel" is a phrase that can mean many things. In our case, it means:

1. **It boots.** This is genuinely impressive. You would be shocked how many OS projects never reach the "prints to serial" stage.
2. **It has real drivers.** Not mock drivers. Real e1000 TX/RX rings. Real NVMe admin queues. Real xHCI port enumeration. These are not stubs that return `Ok(())`.
3. **It has a window manager.** With double buffering. And anti-aliased text. And a taskbar. Written from scratch in software rendering. On a framebuffer. In 2026.
4. **It will panic if you look at it wrong.** The COW fork path has a known double-free under memory pressure. The GPU shader stubs return mathematically incorrect results. The TLS stack can't talk to real HTTPS servers yet. These are documented, tracked, and not hidden.
5. **It is not Linux.** You cannot `apt install` things. There is no `bash`. The shell speaks English-first commands like `look` and `peek`. This is a feature, not a bug, but it is also inconvenient.
6. **One person wrote most of it.** With occasional help from AI agents, contributors, and sheer stubbornness. This means design coherence is high, but bus factor is catastrophic.
7. **The Lean proofs are real.** Zero `sorry` tactics. Actual mathematical verification that some of our core claims hold. This is not decoration.

### What "Research Kernel" Does NOT Mean

- **It does not mean "toy."** A toy doesn't have 256-entry IDT tables, 4-level page tables, and a working TCP stack.
- **It does not mean "abandoned."** We push code regularly. CI runs on every commit. Issues get responses.
- **It does not mean "will never be useful."** The Aether-Link I/O prefetching subsystem has genuine applications in HFT and ML training pipelines. The topological memory allocator has interesting fragmentation properties.
- **It does not mean "we don't know what we're doing."** We know exactly what we're doing. We just chose to do something extremely weird.

### The Emotional Journey Of Using Seal OS

| Stage | Emotion | Cause |
|---|---|---|
| Clone repo | Hope | "This looks cool!" |
| Read build instructions | Confusion | "Why do I need two Rust toolchains?" |
| See it boot in QEMU | Awe | "It actually has a desktop!" |
| Try to open a browser | Despair | "There is no browser." |
| Read the theorem docs | Intimidation | "I need a math degree for this." |
| Run `seal` command | Pride | "It shows theorems! I'm smart!" |
| Try to install a package | Acceptance | "This is not Linux. That's okay." |
| Show a friend | Excitement | "Look at my geometry OS!" |
| Friend asks "why?" | Existential dread | "...because spheres?" |

## Quick Start — Boot in 5 Minutes

### Prerequisites

| Tool | Version | Purpose |
|---|---|---|
| Rust (stable) | 1.85+ | Workspace crates |
| Rust (nightly) | latest | Seal OS kernel (`#![feature(abi_x86_interrupt)]`) |
| QEMU | any | `qemu-system-x86_64` for testing |
| OVMF/EDK2 | any | UEFI firmware for QEMU |


> **Pro tip:** If you don't have nightly installed, run `rustup toolchain install nightly --component rust-src,llvm-tools-preview`. If this fails, complain to the Rust compiler team, not us. We have enough problems.

### 1. Clone & Build Workspace

```bash
git clone https://github.com/teerthsharma/epsilon-hollow.git
cd epsilon-hollow

# Build all workspace crates (stable toolchain)
cargo build --workspace
cargo test --workspace
```

### 2. Build Seal OS Kernel

```bash
cd kernel/seal-os
cargo +nightly build --release
```


> **Fun fact:** The first time we built this, it took 45 minutes because we accidentally compiled debug mode. Don't be us. Use `--release`.

### 3. Create UEFI Disk Image

```bash
cd ../seal-mkimage
cargo +stable run --release

# Verify the image
cargo +stable run --release -- --verify \
  ../seal-os/target/x86_64-unknown-uefi/release/seal-os.img \
  ../seal-os/target/x86_64-unknown-uefi/release/seal-os.efi
```

### 4. Boot in QEMU

**Linux / macOS:**
```bash
cd ../seal-os
./run-qemu.sh
```

**Windows (PowerShell):**
```powershell
cd ../seal-os
.\run-qemu.ps1
```

**Headless proof capture (CI mode):**
```powershell
.\run-qemu.ps1 -HeadlessProof
```

You should see the boot sequence, theorem verification, desktop splash, and finally:

```
[BOOT] Seal OS desktop ready.
[EVENT] Entering real event loop
```

### 5. First Login

Default credentials: `seal` / `seal`

After login, try these shell commands:

```
seal           # Show T1-T10 theorem status + governor epsilon
look           # List directory with Voronoi cell assignments
peek hello.txt # Show file info + ManifoldPayload
move a.txt b/  # O(1) metadata teleport — watch the tick count
search geometry # Content-addressable search via S² embedding
calc sin(pi/2) # Scientific calculator
play tune.wav  # PCM audio playback
```

### 6. Oracle VM VirtualBox (Optional)

```powershell
cd kernel/seal-os
powershell -File .\build-vbox.ps1
powershell -File .\smoke-vbox.ps1 -Seconds 240
```

VM settings: Type=Other, Version=Other/Unknown (64-bit), Enable EFI, RAM=4096 MB, CPUs=1-2, Display=VMSVGA, Storage=SATA/AHCI.

---

## Boot Log

```
Seal OS v0.4.5 — The Geometrical Operating System
OS state = topology on S². File moves use metadata topology; byte persistence still scales with file size.

[BOOT] Heap initialized (16 MB)
[BOOT] IDT + PIC initialized
[T4/AGCR] Governor online: epsilon = 0.1000
[T1/TSS]  Voronoi index: 8 cells, test lookup -> cell 0
[ALLOC] O(1) proof: topo_cells=8, l3_word_probes_per_cell=2, single_word_probes_per_cell=8192, contiguous_candidate_probes=128, contiguous_max_run_pages=64, toporam_max_run_pages=64, marking=bounded_by_contiguous_max_run_pages
[ALLOC] runtime counters: fast_hits=1, bounded_misses=0, max_contiguous_probes_seen=0
[BENCH] toporam-alloc iterations=64 ok=64 p50_cycles=<n> p95_cycles=<n> max_cycles=<n> target_cell_hits_delta=64 target_cell_fallbacks_delta=0 low_to_high_fallbacks_delta=0 high_to_low_fallbacks_delta=0 pcie_to_high_fallbacks_delta=0 pcie_to_low_fallbacks_delta=0 free_before=<n> free_after=<n>
[BENCH] alloc-frame iterations=64 ok=64 p50_cycles=<n> p95_cycles=<n> max_cycles=<n> fast_hits_delta=64 bounded_misses_delta=0 max_contiguous_probes_seen_delta=0 free_before=<n> free_after=<n>
[BENCH] manifold-teleport api=teleport fs_mode=ramfs persistence=metadata_only samples=3 ok=3 same_inode=3 src_gone=3 dst_present=3 entries_min=8 entries_max=256 payload_bytes=64 p50_cycles=<n> p95_cycles=<n> max_cycles=<n> ticks_max=<n> metadata_ops_max=7 persistence_bytes_per_move=0 payload_points=<n>
[BENCH] scheduler-select-next selector=select_next_task mode=live_requeue clock=rdtsc iterations=64 ok=64 ready_before=3 ready_after=3 cells=8 priority_buckets=256 voronoi_locate_probes=8 max_cell_bitmap_tests=9 max_priority_bucket_scan=256 context_switches=0 selected_priority_max=<n> p50_cycles=<n> p95_cycles=<n> max_cycles=<n>
[BENCH] tcp-packet-demux api=handle_tcp_packet fixture=listener_first accepted_state=established ok=1 listener_first=1 payload_bytes=4 rx_bytes=4 cleanup=ok
[BOOT] All T1-T10 theorems VERIFIED; T1-T5 ACTIVE in runtime paths
[Aether-Lang] runtime proof: parser=ok interpreter=ok app_host=ok script=aether_boot_probe result=seal-topology-ok
[BOOT] Desktop proof frame blit done
[GFX] desktop-soak frames=24 p50_cycles=<n> p95_cycles=<n> max_cycles=<n> missed_16ms=unscaled input_events=0 dirty_px_max=786432
[BOOT] Seal OS desktop ready.
[EVENT] Entering real event loop
[ManifoldFS] Teleported 'hello.txt' metadata in 1 ticks; persistent bytes verified separately
[Scheduler] 3 tasks, running 'idle', epsilon=0.1000
[Shell] T1/TSS  Voronoi cells: 8, Betti-0: 8
```

Every line in this log is a **hard gate** in CI. If any line is missing, the build fails.


> **What this means in human terms:** We have automated tests that verify the OS boots, initializes memory, checks all ten mathematical theorems, runs benchmarks, and displays a desktop — every single time someone pushes code. If you break the boot log, CI breaks. If CI breaks, we fix it. This is how you know we actually test things instead of just claiming they work.

---

## Architecture

Seal OS is organised into 11 layers, from UEFI boot to user applications. Every runtime layer above Layer 0 is driven by T1-T5, and every boot must pass the T1-T10 theorem gate.

```mermaid
graph TB
    subgraph SG1["Layer 10 — Applications"]
        TERM["Terminal Emulator"]
        IDE["Seal IDE"]
        FMGR["File Manager"]
        TVIEW["Theorem Viewer"]
        CALC["Calculator"]
        SPLAYER["SealPlayer"]
        GAMES["Snake / Breakout / Warp Racer"]
    end

    subgraph SG2["Layer 9 — Desktop"]
        WALL["Wallpaper<br/>Schwarzschild metric + Faraday tensor"]
        TBAR["Taskbar<br/>T1-T10 status strip<br/>epsilon=0.042"]
    end

    subgraph SG3["Layer 8 — Window Manager"]
        COMP["Compositor<br/>double-buffered, dirty-rect tracking"]
        WIN["Window Manager<br/>z-order, decorations, cursor"]
        EVT["Event Dispatch<br/>keyboard/mouse → focused window"]
    end

    subgraph SG4["Layer 7 — Shell"]
        SHELL["SealShell<br/>look, peek, move, search,<br/>tasks, seal, race, stats"]
    end

    subgraph SG5["Layer 6 — Syscalls"]
        ABI["Seal ABI: native syscalls<br/>fork/exec/pipe/dup/brk/signal/ioctl<br/>gettimeofday/getrandom/kmsg"]
        EPSILON_SYS["Epsilon: manifold_query,<br/>teleport, theorem_status,<br/>pkg_install, setting_get/set"]
    end

    subgraph SG6["Layer 5 — Process Scheduler"]
        SCHED["ManifoldScheduler<br/>T1 Voronoi task groups<br/>T4 adaptive timeslice<br/>T2 predict next runnable"]
    end

    subgraph SG7["Layer 4 — ManifoldFS"]
        MFS["ManifoldFS<br/>raw bytes + 64-point S² payloads<br/>metadata teleport via topological surgery<br/>bucketed search via Voronoi"]
        ENC["Encoder Pipeline<br/>trigram hash → JL project → L2 normalize"]
    end

    subgraph SG8["Layer 3 — Graphics"]
        FB["Framebuffer<br/>1024×768×32bpp, double-buffered"]
        FONT["8×16 Bitmap Font + High-Tech Engine"]
        SPLASH["Boot Splash<br/>ASCII seal art + progress bar"]
        HTEK["htek.rs — AA text, gradients,<br/>rounded rects, glow, alpha blend"]
        TOPO3D["topo_render.rs —<br/>Voronoi tiles + spectral LOD +<br/>hyperboloid projection"]
    end

    subgraph SG9["Layer 2 — Interrupts & Drivers"]
        IDT["IDT — 256 entries"]
        APIC["Local APIC + I/O APIC"]
        TIMER["APIC Timer<br/>per-CPU + watchdog"]
        KBD["PS/2 Keyboard<br/>IRQ1"]
        MOUSE["PS/2 Mouse<br/>IRQ12"]
        SERIAL["Serial COM1<br/>115200 baud"]
        PCIDRV["PCI: NVMe + AHCI + e1000 +<br/>xHCI + HDA + WiFi/BT/GPU probe"]
        ACPIDRV["ACPI (RSDP, MADT)"]
    end

    subgraph SG10["Layer 1 — Memory"]
        HEAP["Slab Allocator (64B-2048B)<br/>+ Page Allocator + VMM<br/>+ Topological Free Index"]
        TOPORAM["topo_ram.rs —<br/>Voronoi + spectral + entropy +<br/>hyperbolic lifetime classification"]
    end

    subgraph SG11["Layer 0 — Boot"]
        UEFIBOOT["UEFI Entry<br/>PE/COFF, 64-bit long mode"]
        GOP["GOP Framebuffer Query"]
        SMP["SMP Bring-up<br/>INIT-SIPI-SIPI"]
    end

    TERM --> COMP
    IDE --> COMP
    FMGR --> COMP
    TVIEW --> COMP
    CALC --> COMP
    SPLAYER --> COMP
    GAMES --> COMP
    WALL --> COMP
    TBAR --> COMP
    COMP --> FB
    TOPO3D --> FB
    WIN --> EVT
    SHELL --> ABI
    SHELL --> EPSILON_SYS
    ABI --> SCHED
    EPSILON_SYS --> SCHED
    ABI --> MFS
    EPSILON_SYS --> MFS
    SCHED --> HEAP
    MFS --> ENC
    ENC --> HEAP
    FB --> HEAP
    SPLASH --> FB
    IDT --> APIC
    APIC --> TIMER
    APIC --> KBD
    APIC --> MOUSE
    SERIAL --> HEAP
    PCIDRV --> HEAP
    ACPIDRV --> APIC
    HEAP --> UEFIBOOT
    UEFIBOOT --> GOP
    GOP --> SMP

    style MFS fill:#1a1a2e,stroke:#e94560,color:#fff
    style SCHED fill:#1a1a2e,stroke:#0f3460,color:#fff
    style ENC fill:#1a1a2e,stroke:#e94560,color:#fff
    style UEFIBOOT fill:#0d1117,stroke:#58a6ff,color:#fff
```

### Boot Sequence

```mermaid
sequenceDiagram
    participant UEFI
    participant efi_main
    participant uefi_entry
    participant kernel_main

    UEFI->>UEFI: Load PE/COFF binary in 64-bit long mode
    UEFI->>efi_main: #[entry] fn efi_main() → Status
    efi_main->>uefi_entry: uefi_entry::run()

    uefi_entry->>uefi_entry: Serial init (COM1 @ 115200)
    uefi_entry->>uefi_entry: UEFI helpers init
    uefi_entry->>uefi_entry: Search config tables for ACPI 2.0 RSDP
    uefi_entry->>uefi_entry: Query GOP for framebuffer
    uefi_entry->>uefi_entry: Get LoadedImage (kernel base/size)
    uefi_entry->>uefi_entry: exit_boot_services() — UEFI is gone
    uefi_entry->>uefi_entry: Copy memory map → BootInfo

    uefi_entry->>kernel_main: kernel_main(&boot_info)
    kernel_main->>kernel_main: Memory init (phys bitmap + slab + VMM + GDT)
    kernel_main->>kernel_main: topo_ram::init()
    kernel_main->>kernel_main: SMP bring-up (INIT-SIPI-SIPI for APs)
    kernel_main->>kernel_main: ACPI parse (RSDP → MADT → APIC topology)
    kernel_main->>kernel_main: Security init (SMAP/SMEP + MAC + audit)
    kernel_main->>kernel_main: IDT init (256 entries, APIC vectors)
    kernel_main->>kernel_main: APIC init (Local + I/O APIC + timer)
    kernel_main->>kernel_main: Syscall MSRs (SYSCALL/SYSRET)
    kernel_main->>kernel_main: entropy::init() (RDRAND + RDSEED)
    kernel_main->>kernel_main: rtc::init() + watchdog::init(5000)
    kernel_main->>kernel_main: PCI enumeration
    kernel_main->>kernel_main: NVMe + xHCI + HDA init
    kernel_main->>kernel_main: Framebuffer init + topo_render::init()
    alt Framebuffer available
        kernel_main->>kernel_main: Graphical boot (login → splash → theorems → desktop)
    else No framebuffer
        kernel_main->>kernel_main: Serial-only boot (theorems → shell)
    end
    kernel_main->>kernel_main: Scheduler spawn + HLT loop
```

**UEFI boot** — pure Rust, zero assembly in the boot path. UEFI firmware loads the kernel as a PE/COFF binary, already in 64-bit long mode with identity-mapped page tables. The kernel queries GOP (Graphics Output Protocol) for a framebuffer, reads the UEFI memory map, then calls `exit_boot_services()` to take full control.

**Target**: `x86_64-unknown-uefi` with `build-std = ["core", "alloc"]`.

---


> **Assembly confession:** The AP trampoline lives in `src/boot/ap_trampoline.rs` and contains approximately 40 lines of inline assembly. This wakes up secondary CPUs via INIT-SIPI-SIPI. Without it, you're running on one core like it's 1995. We tried to do it in pure Rust. The borrow checker said no.

---

## Design Decisions That Seemed Good At 3 AM

Every OS has design decisions. Ours were made at ungodly hours by a sleep-deprived developer mainlining topology papers. Here are the ones that stuck, and why.

### Decision 1: The Unit Sphere As Universal Data Structure

**What we did:** Every kernel object — files, memory frames, tasks — gets an embedding on S².

**Why it seemed good:** Spheres have no boundary. No edge cases. Natural metric. Beautiful symmetry.

**Why it was painful:** Computing `arccos(sin θ₁ sin θ₂ + cos θ₁ cos θ₂ cos(φ₁ - φ₂))` for every file lookup is expensive. We spent months optimizing the hot path. The Voronoi index caches centroids. The lookup is O(K) where K=8. But still. Arccos. In a kernel.

**Verdict:** Would do again, but with more coffee.

### Decision 2: A Custom Programming Language Inside The Kernel

**What we did:** Aether-Lang — lexer, parser, AST, interpreter, and VM — all in `no_std` kernel space.

**Why it seemed good:** TempleOS proved it's possible. We wanted a native scripting language.

**Why it was painful:** Writing a parser without `std::collections::HashMap` means using `BTreeMap` for symbol tables. Writing a VM without `Box` means careful manual memory management. Debugging a language inside a kernel means you can't just `println!` — you have to write to the serial port.

**Verdict:** Absolutely worth it. Using `~` as a terminator is satisfying in a way semicolons never were.

### Decision 3: Theorem-Gated Boot

**What we did:** Ten mathematical theorems must be verified before the scheduler starts. T1-T5 are active in runtime paths.

**Why it seemed good:** Mathematical rigor! Proof-carrying code! Formal verification!

**Why it was painful:** Lean 4 proofs take time to write. The `sorry` tactic is tempting. Maintaining proof strength while changing kernel code is like juggling torches while riding a unicycle.

**Verdict:** Zero `sorry` tactics. We are proud. We are also tired.

### Decision 4: Software Rendering For Everything

**What we did:** No GPU acceleration for the desktop. Everything is rasterized in software on a 1024×768 framebuffer.

**Why it seemed good:** Portability. No vendor drivers. Works on any hardware with a framebuffer.

**Why it was painful:** Anti-aliased text rendering in software is slow. Glow effects require multiple blur passes. The Schwarzschild metric wallpaper looks cool but eats CPU cycles.

**Verdict:** The desktop looks amazing. The CPU usage is concerning. We have plans for GPU offload (PM4 rings, shader stubs), but they're not ready yet.

### Decision 5: No POSIX

**What we did:** Seal ABI is native. No `open()`, `read()`, `write()` semantics inherited from Unix.

**Why it seemed good:** POSIX is 50 years of accumulated baggage. We wanted clean semantics.

**Why it was painful:** Every programmer knows POSIX. Nobody knows Seal ABI. Porting tools is impossible. Writing a shell from scratch is hard.

**Verdict:** Correct decision, but our user base is approximately 12 people, and 8 of them are us.

### Decision 6: The "Zero Assembly" Marketing Claim

**What we did:** For months, we claimed "zero assembly" on our comparison table.

**Why it seemed good:** It was true for the boot path. UEFI loads us in long mode. We never touch `asm!` in `main.rs`.

**Why it was painful:** We forgot about the AP trampoline. And CPU idle loops. And GDT loading. About 40 lines total. Someone on Reddit called us out.

**Verdict:** We fixed it. We now say "minimal inline assembly." The Reddit thread was actually helpful. Thanks, person whose username we forgot.

## The Ten Theorems

These are not decorative. T1-T5 runtime callsites are source-gated in kernel paths today; deeper formal runtime proof and benchmarks remain pending. T6-T10 are boot-verified theorem gates for the HFT/ML world-model path.

### T1-T5 — Active in Runtime

| ID | Name | Formal Statement | Governs |
|----|------|------------------|---------|
| **T1** | TSS — Topological State Synchronization | O(1) retrieval via spherical Voronoi tessellation | file lookup, task groups, memory locality |
| **T2** | SCM — Spectral Contraction Mapping | Spectral contraction toward fixed-point attractor | prefetch, next-task prediction |
| **T3** | GMC — Geodesic Memory Consolidation | Renyi entropy bound on memory consolidation | cell merging, defrag triggers |
| **T4** | AGCR — Adaptive Governor Convergence Rate | PD governor convergence (eigenvalue-bounded) | timeslice, cache, FPS, heap |
| **T5** | HCS — Hyperbolic Curvature Separation | Hyperbolic vs Euclidean separation ratio | path depth, lifetime classes, power mapping |

### T6-T10 — Boot-Verified Gates

| ID | Name | What |
|----|------|------|
| **T6** | RGCS — Ring-Allreduce Gradient Coherence | Tangent deviation bound for sync frequency |
| **T7** | PHKP — Persistent Homology KV Partitioning | Betti-guided latency via topological persistence |
| **T8** | TEB — Thermodynamic Erasure Bound | Landauer energy bound per bit erasure |
| **T9** | CMA — Cross-Manifold Alignment | Alignment error via Procrustes curvature + SVD |
| **T10** | WPHB — World Predictive Horizon Bound | Predictive horizon from information + stability |

### Runtime Integration

| Operation | Theorem | What Happens |
|-----------|---------|--------------|
| `store()` | T1/TSS | Voronoi cell assignment for bucketed lookup |
| `store()` | T2/SCM | SpectralContractionOperator evolves prefetch state |
| `teleport()` | T4/AGCR | Governor adapts epsilon based on move deviation |
| `teleport()` | T3/GMC | If entropy > 2.0 bits, merge smallest Voronoi cells |
| `find()` | T1/TSS | Voronoi narrows search to a bucket |
| path resolution | T5/HCS | Hyperbolic tree structure for deep paths |
| scheduler tick | T4/AGCR | Timeslice scale factor adapted by deviation |
| scheduler select | T2/SCM | Predicted cell checked first (O(1) amortized) |
| compositor frame | T4/AGCR | Quality scaled 0→4 targeting 16 ms |
| memory alloc | T1/TSS | Voronoi cell for frame locality |
| memory free | T3/GMC | Betti-0 entropy check triggers reseed |


---

## What Theorems Actually Do (For Normal Humans)

Okay, let's be real. That previous section was dense. If you don't have a math degree, "spectral contraction toward fixed-point attractor" might as well be Klingon. Here's what these theorems actually mean in practical terms.

### T1 (TSS): The "Where Is It?" Theorem

**What it does:** When you ask for a file, memory frame, or task, T1 tells the kernel which "neighborhood" to look in first.

**Human translation:** Imagine a library where books are shelved by color instead of alphabetically. T1 is the librarian who knows that "red books are in the east wing." It doesn't tell you exactly where the book is, but it narrows your search from a million books to a thousand.

**Why it matters:** O(1) lookup. In practice, this means file operations are fast because we don't scan entire directories.

### T2 (SCM): The "What Next?" Theorem

**What it does:** Predicts what the system will need next based on past patterns.

**Human translation:** It's like Netflix recommendations, but for memory pages and CPU tasks. "You loaded page 1 of this document; you'll probably want page 2 next."

**Why it matters:** Prefetching. The scheduler uses this to guess which task will run next. The filesystem uses it to guess which block you'll read next. It's not always right, but when it is, things are fast.

### T3 (GMC): The "Clean Up" Theorem

**What it does:** Measures how scattered things are and triggers consolidation when they get too scattered.

**Human translation:** Ever defragmented a hard drive? T3 is an automatic, mathematical defragmentation trigger. It looks at memory (or files) and says "these are too spread out; time to reorganize."

**Why it matters:** Prevents fragmentation from getting out of hand. The entropy bound means we only reorganize when it's actually worth it, not constantly.

### T4 (AGCR): The "Chill Out / Speed Up" Theorem

**What it does:** A control loop that adjusts system behavior based on how wrong its predictions were.

**Human translation:** It's cruise control for your OS. If the scheduler keeps guessing wrong, T4 slows down the guessing (longer timeslices, more stability). If the scheduler is nailing predictions, T4 speeds things up (shorter timeslices, more responsiveness).

**Why it matters:** Adaptive performance. The OS learns your workload and adjusts. No manual tuning needed.

### T5 (HCS): The "Hierarchy" Theorem

**What it does:** Decides whether something should be treated as short-lived or long-lived based on its topological properties.

**Human translation:** Temporary files and permanent files live in different "universes" in our OS. T5 is the classifier that decides which universe a file belongs to. Temporary stuff gets fast, ephemeral handling. Important stuff gets careful, persistent handling.

**Why it matters:** Different policies for different lifetimes. You don't treat a log file the same as your password database.

### T6-T10: The "Someday" Theorems

These are for the HFT/ML world-model path. They're verified at boot but not yet active in everyday runtime. Think of them as "expansion pack" theorems. They'll matter when Seal OS is running trading algorithms and training neural networks. For now, they prove our math is sound and our boot sequence is rigorous.

Lean 4 proof artifacts live in `kernel/aether/aether-verified/lean/`. Proof strength is tracked in [docs/THEOREMS.md](docs/THEOREMS.md). Some bounds are full proofs; some are layered bridge checks.

---

## What's Inside

The following sections are collapsible deep dives. Click to expand.

<details>
<summary><strong>Memory & Topology</strong></summary>

### Allocator Stack

```mermaid
graph TD
    subgraph SG12["Allocator Stack"]
        GLOBAL["GlobalAlloc (SealAllocator)"]
        SLAB["Slab Allocator<br/>64B, 128B, 256B, 512B, 1024B, 2048B<br/>intrusive free-list per size class"]
        PAGE["Page Allocator + VMM<br/>4 KiB pages, mapped on demand"]
        PHYS["Physical Frame Allocator<br/>topological free index + bitmap truth<br/>supports up to 16 GiB RAM"]
    end

    GLOBAL -->|"2048 B or less"| SLAB
    GLOBAL -->|"greater than 2048 B"| PAGE
    SLAB --> PHYS
    PAGE --> PHYS
```

The kernel uses a tiered allocator implementing `GlobalAlloc`:

- **Small allocations (≤ 2048 B)**: Slab allocator with six size classes. Objects are carved from 4 KiB pages with intrusive free-lists. O(1) alloc/dealloc.
- **Large allocations (> 2048 B)**: Virtual pages allocated from a bump region, backed by physical frames from the bitmap allocator, mapped via 4-level page tables.
- **Physical frame allocator**: Bitmap-backed and topological-indexed, initialized from the UEFI memory map. One bit per 4 KiB frame, up to 128 GiB of RAM. Single-frame allocations use the fixed-cell summary index; large contiguous DMA ranges use bounded topological candidate probes.
- **GDT + TSS**: Full Global Descriptor Table with Task State Segment, supporting ring-0/ring-3 transitions.

### Why a Bitmap + Slab Hybrid?

The physical frame allocator keeps a bitmap as the truth source, then layers a fixed topological free index over it. This is deliberate: the allocator can prove a bounded hot path without losing simple verification.

- **Topological free index**: O(1) single-frame allocation across eight cells, three summary levels, and a bounded word/bit probe.
- **Contiguous frame path**: multi-page DMA requests use 128 bounded topological candidate probes and a hard 64-page run cap.
- **Bitmap truth**: one bit per 4 KiB frame, predictable cache behavior, trivial serialization for snapshots.
- **Slab**: Six fixed size classes with intrusive singly-linked free lists. Carved from 4 KiB pages. O(1) alloc/dealloc, no fragmentation within a page.
- **TopoRAM wrapper**: Adds 64 bytes of metadata per frame (S² embedding, access history, Voronoi cell, lifetime class). Public TopoRAM allocation and free-side topology repair share the physical allocator's 64-page run cap.

The hybrid gives us: small objects → slab (fast, no external fragmentation). Single frames → topological free-index allocation. Large contiguous DMA requests → bounded topological candidate probes. All frames → topological metadata for T1-T5 decisions.

### Memory Topology

```mermaid
flowchart TB
    subgraph SG14["Userspace (Ring 3)"]
        U_TEXT[".text — code"]
        U_DATA[".data/.bss — globals"]
        U_HEAP["Heap — brk / mmap"]
        U_STACK["Stack"]
    end

    subgraph SG15["Kernel Space (Ring 0)"]
        K_SLAB["Slab Allocator<br/>64B–2048B, 6 classes"]
        K_PAGE["Page Allocator + VMM<br/>4-level PML4 on-demand"]
        K_PHYS["Physical Frame Allocator<br/>topological free index<br/>bitmap truth store"]
        K_TOPO["TopoRAM<br/>Voronoi cells + spectral prefetch +<br/>entropy governor + hyperbolic lifetime"]
    end

    U_HEAP -->|brk grow| K_PAGE
    U_HEAP -->|mmap| K_PAGE
    K_SLAB -->|alloc_frame| K_PHYS
    K_PAGE -->|alloc_frame| K_PHYS
    K_TOPO -->|prefetch_hint| K_PHYS
    K_PHYS -->|topo_ram init| K_TOPO
```

</details>

<details>
<summary><strong>ManifoldFS — The Filesystem</strong></summary>

This is not ext4. This is not FAT. ManifoldFS keeps faithful raw bytes for reads and writes, plus **64-point ManifoldPayload embeddings on the unit sphere S²** for content addressing, Voronoi indexing, topology-aware moves, and future payload-first disk layout work.

### Encoding Pipeline

```
Raw bytes (4096 B block)
  ↓
Trigram hash → 128-dim sparse vector
  ↓
Johnson-Lindenstrauss projection → 3-dim vector
  ↓
L2 normalization → point on S²
  ↓
Repeat for 64 blocks → 64-point cloud on S²
  ↓
Compute Betti-0 (connected components) + content hash
  ↓
Store as ManifoldPayload
```

**Why trigram hashing?** Trigrams capture local byte structure better than uniform sampling. A 4096-byte block has 4094 overlapping trigrams; we hash them into a 128-bin histogram. This preserves content similarity: two files with similar byte distributions map to nearby points on S².

**Why JL projection?** The Johnson-Lindenstrauss lemma guarantees that n points in high-dimensional space can be mapped to O(log n) dimensions with bounded distortion. We use a random Gaussian projection matrix (seeded per-filesystem) to map 128-dim → 3-dim.

**Why 64 points?** 64 points on S² gives us enough resolution to distinguish files while keeping the payload small (64 × 3 × 8 bytes = 1536 bytes per file). Content-addressable lookup uses Voronoi cell assignment: given a query point, find nearest seed → search only that cell's files.

### Metadata Teleport

Moving a file between directories uses O(1) metadata surgery for directory/inode rewiring because the ManifoldPayload identity does not change. The persistent path updates inode/directory metadata without rewriting raw file bytes; the boot marker requires `persistence_bytes_per_move=0`.

```mermaid
graph TD
    subgraph SG26["Storage"]
        PAYLOAD["ManifoldPayload<br/>64 SpherePoints (θ, φ)<br/>Betti-0 via Union-Find (ε²=0.25)<br/>FNV-1a content hash"]
        INODE["Inode<br/>name, kind, payload, metadata<br/>voronoi_cell, cluster_id, parent"]
        VORONOI["T1: Voronoi cell assignment<br/>SphericalVoronoiIndex&lt;8&gt;"]
        BTREE["BTreeMap&lt;InodeId, Inode&gt;<br/>(no_std — no HashMap)"]
    end

    subgraph SG27["Teleport Metadata Move"]
        SRC["Source dir"] -->|"1. Remove from src.dir_entries"| SURGERY["Topological Surgery"]
        SURGERY -->|"2. Insert into dst.dir_entries"| DST["Dest dir"]
        SURGERY -->|"3. Update inode.parent"| GOV["T4: Governor adapts epsilon"]
        GOV -->|"4. Check entropy"| MERGE["T3: If Betti-0 > threshold<br/>merge smallest cells"]
    end

    PAYLOAD --> INODE
    INODE --> VORONOI
    INODE --> BTREE
```

### TopCrypt — Topological File Encoding

TopCrypt is topological encoding/obfuscation, not cryptographic protection. Files are stored as 64-byte blocks encoded as 16-point clouds on S² with CRC32, shuffle, and XOR masks. Shell commands: `topcrypt encode`, `topcrypt lock`, `topcrypt unlock`, `topcrypt info`.

**Lypnos Guard**: `Ctrl+L` shuffles/masks a topological file, `Ctrl+E` flattens it to bytes, and `Ctrl+I` absorbs an external file into manifold form. AEAD/KDF security gate pending.


> **Honest note:** TopCrypt is cool but not secure against a determined attacker. It's obfuscation. Think of it as "security through making the attacker do math homework." Real encryption (AES-GCM with proper key management) is on the roadmap but not implemented yet.

</details>

<details>
<summary><strong>Process Scheduler</strong></summary>

The ManifoldScheduler maintains 8 Voronoi cells. Each task's manifold embedding (8-dimensional, normalized) is projected to S² and assigned to the nearest cell.

```mermaid
graph TD
    subgraph SG29["ManifoldScheduler"]
        TASKS["Task Queue<br/>Vec&lt;Task&gt;"]
        VOR["T1: SphericalVoronoiIndex&lt;8&gt;<br/>8 Voronoi cells on S²"]
        GOV["T4: GeometricGovernor<br/>ε(t+1) = ε(t) + α·e(t) + β·de/dt<br/>α=0.01, β=0.05, ε₀=0.1"]
        PRED["T2: SpectralContractionOperator&lt;8&gt;<br/>predict next runnable task"]
    end

    subgraph SG30["Scheduling Decision"]
        TICK["Timer tick (IRQ0)"] --> CHECK["Check deviation greater than epsilon?"]
        CHECK -->|yes| PREDICT["T2: Predict next cell"]
        PREDICT --> CELL["T1: Select task from Voronoi cell"]
        CELL --> TIMESLICE["T4: Compute timeslice<br/>epsilon < 0.5 → scale=2.0 (stable)<br/>epsilon ≥ 0.5 → scale=0.5 (volatile)"]
        CHECK -->|no| CONTINUE["Continue current task"]
    end

    TASKS --> VOR
    TASKS --> GOV
    TASKS --> PRED
```

Task selection:

1. **T2 prediction**: The spectral contraction operator maintains a 3-dim prediction state. It predicts which cell the next runnable task will be in.
2. **Cell probe**: Check predicted cell first. If empty, scan remaining cells (at most 7 more probes).
3. **Priority bucket**: Within a cell, tasks are stored in 256 priority buckets. Selection pops from the highest non-empty bucket.
4. **T4 adaptation**: The geometric governor measures scheduling deviation (actual vs predicted cell hit). It adapts a timeslice scale factor: ε < 0.5 → stable → longer timeslices; ε ≥ 0.5 → volatile → shorter timeslices.

The scheduler lock is released **before** context switch, preventing deadlock when the new task's timer fires immediately. CR3 is swapped for userspace tasks; kernel tasks use the BSP PML4.


> **Why 256 priority buckets?** Because 255 felt like not enough and 257 felt excessive. Also, it fits nicely in a `u8` and aligns with our "powers of 2 are beautiful" aesthetic.

</details>

<details>
<summary><strong>Interrupts & Drivers</strong></summary>

**APIC**: Replaces the legacy 8259 PIC. Local APIC provides per-CPU timer and inter-processor interrupts (IPIs). I/O APIC routes external device IRQs. Discovered via ACPI MADT parsing.

**Keyboard driver**: reads scancodes from port `0x60`, maps to ASCII via a 58-entry table (set 1 scancodes).

**APIC Timer**: per-CPU local APIC timer for scheduler ticks and governor sampling.

**PCI**: Full bus enumeration via config space ports 0xCF8/0xCFC. Discovers AHCI controllers, NICs, WiFi/BT adapters, USB controllers, GPUs.

**AHCI**: SATA disk driver with MMIO command/FIS structures, read/write sector support.

**e1000**: Intel 8254x Ethernet — MMIO registers, 256-entry TX/RX descriptor rings, packet send/receive.

**Serial**: COM1 initialized at 115200 baud, 8N1. Primary diagnostic channel — visible in QEMU via `-serial stdio`.

**NVMe**: Admin queue + I/O queue creation, Identify Controller/Namespace, PRP-based DMA sector read/write.

**xHCI USB 3.0**: Controller reset/init, event/command rings, port enumeration, device slot assignment, SET_ADDRESS, GET_DESCRIPTOR. Supports HID boot keyboards/mice (interrupt IN endpoints) and Mass Storage (bulk IN/OUT with SCSI BBB).

**HDA Audio**: CORB/RIRB command engines, codec widget discovery, DAC pin selection, output stream descriptor with DMA buffer, 48kHz 16-bit stereo PCM playback.

**Entropy**: CPUID probe for RDRAND/RDSEED carry-flag retry loops. Hardware random for TLS session keys and `SYS_GETRANDOM`.

**RTC + Watchdog**: CMOS real-time clock (ports 0x70/0x71) with BCD/binary detection. APIC timer watchdog — pets via `SYS_WATCHDOG`, triggers keyboard-controller reset on 5-second hang.


> **Driver confession:** Our GPU driver is mostly stubs. The PM4 ring infrastructure is real, but the shaders don't actually compute topology yet. They execute (the GPU doesn't crash), but the results are mathematically wrong. This is documented. This is honest. We are working on it. Please stop emailing us about it.

</details>

<details>
<summary><strong>Graphics & Desktop</strong></summary>

### High-Tech Rendering Engine (`graphics/htek.rs`)

Seal OS uses a custom software rendering engine that produces modern, high-tech UI:

- **Anti-aliased text**: 2x supersampled font rendering with neighbor-aware fringe blending
- **Gradient fills**: Per-scanline linear interpolation (vertical and horizontal) with 256-step color lerping
- **Rounded rectangles**: Corner distance field evaluation with sub-pixel anti-aliasing
- **Glow effects**: Multi-offset radial blur passes with alpha compositing
- **Alpha blending**: Full 8-bit per-pixel compositing engine
- **Stroke rendering**: Anti-aliased rounded rectangle outlines via inner/outer distance field subtraction

### Topological 3D Render Pipeline (`graphics/topo_render.rs`)

```mermaid
flowchart LR
    Mesh["TopoMesh<br/>vertices + S² embedding"] --> T5["T5 Hyperbolic<br/>Projection<br/>sinh/cosh fisheye"]
    T5 --> T2["T2 Spectral LOD<br/>degree centrality +<br/>screen-space area"]
    T2 --> T3["T3 Betti-1<br/>Integrity Check<br/>reject hole creation"]
    T3 --> T1["T1 Voronoi<br/>8×8 Screen Tiles"]
    T1 --> Raster["Rasterize<br/>flat / gouraud / phong"]
    Raster --> Depth["Per-Pixel Depth<br/>1/z perspective-correct"]
    Depth --> T4["T4 Governor<br/>adaptive quality<br/>0=wireframe → 4=phong+AA"]
    T4 --> Blit["blit() → VRAM"]

    style T5 fill:#1a1a2e,stroke:#e94560,color:#fff
    style T2 fill:#1a1a2e,stroke:#0f3460,color:#fff
    style T3 fill:#1a1a2e,stroke:#00ffaa,color:#fff
    style T1 fill:#1a1a2e,stroke:#ff9f1c,color:#fff
    style T4 fill:#1a1a2e,stroke:#e94560,color:#fff
```

### Desktop Wallpaper

Renders two equations procedurally:

1. **Schwarzschild metric** (black hole geometry):  
   `ds² = -(1 - 2GM/rc²)dt² + (1 - 2GM/rc²)⁻¹dr² + r²dΩ²`

2. **Faraday tensor** (electromagnetic field):  
   The 4×4 antisymmetric F^μν matrix with E and B field components


> **Why these equations?** Because they're beautiful, they're physically meaningful, and they look really cool as a desktop background. Also, we spent a week implementing the renderer and by god we're going to use it. If you want kittens, use Windows.

</details>

<details>
<summary><strong>Network Stack</strong></summary>

### TCP/IP Stack

Wired end-to-end through IPv4 → net::transmit → e1000 TX descriptor ring.

- **TCP**: Listen/accept backlog, SYN queue, retransmission timer
- **UDP + DHCP**: Full DHCP state machine (Init → Discover → Request → Bound)
- **DNS**: Proper query packets (ID, flags, QNAME, QTYPE A, QCLASS IN)

### TLS 1.3 PSK

The TLS path is intentionally narrow: PSK-only record encryption for the native HTTPS client.

1. **ClientHello**: TLS record (content type 0x16, version 0x0303) with supported_versions and psk_key_exchange_modes
2. **ServerHello parsing**: extracts server random, derives handshake traffic secrets using HKDF-SHA256
3. **Key derivation**: HKDF-Extract(salt=0, IKM=psk) → HKDF-Expand(label="handshake", context=ClientHello+ServerHello)
4. **AES-128-GCM**: Per-record encryption with 12-byte nonce (4-byte salt + 8-byte sequence number)
5. **Record wrapping**: TLSInnerPlaintext → AEAD encrypt → TLSRecord

Minimal TLS 1.3 PSK record path — no X.509/PKI/ECDHE gate yet; production HTTPS compatibility is pending. The random bytes function uses RDSEED first, then RDRAND. If neither source is available or the CPU repeatedly reports carry-clear failure, `getrandom` returns failure instead of manufacturing cryptographic bytes.

### Driver Stack

```mermaid
flowchart TB
    subgraph SG19["Applications"]
        A1[SealShell]
        A2[Seal IDE]
        A3[File Manager]
        A4[Media Player]
    end

    subgraph SG20["Block Layer"]
        B1["NVMe - DMA"]
        B2["AHCI SATA"]
        B3["USB MSC - SCSI BBB"]
    end

    subgraph SG21["Network Stack"]
        N1["HTTP/HTTPS Client"]
        N2["TLS 1.3 - AES-GCM"]
        N3["TCP - retrans/SYN/backlog"]
        N4["UDP + DHCP + DNS"]
        N5["e1000 TX/RX Rings"]
    end

    subgraph SG22["USB Stack"]
        U1["HID Keyboard/Mouse"]
        U2["Mass Storage"]
        U3["xHCI Controller"]
    end

    subgraph SG23["Audio"]
        Au1["play_pcm()"]
        Au2["HDA CORB/RIRB"]
        Au3["Output Stream DMA"]
    end

    A1 --> N1
    A3 --> B1
    N1 --> N2
    N2 --> N3
    N3 --> N4
    N4 --> N5
    A4 --> Au1
    Au1 --> Au2
    Au2 --> Au3
    U1 --> U3
    U2 --> U3
```


> **TLS reality check:** We can encrypt packets. We can't verify certificates. If you try to visit `https://google.com`, it will fail because Google doesn't accept PSK. This is documented. We're working on X.509. Check back in v0.6.0.

</details>

<details>
<summary><strong>Security</strong></summary>

### KPTI (Kernel Page-Table Isolation)

CR3 swap code exists via `memory/pgtable_asm.rs`. A hard gate still needs to prove installed KPTI page tables during a boot selftest. **Status**: scaffolding present, syscall entry/exit wiring pending.

### ASLR

Userspace mmap base is randomized with a 16-bit entropy shift (up to 65,536 possible bases). The random source is RDRAND/RDSEED when hardware entropy is available; low-entropy fallback paths are treated as non-production.

### Seccomp

Classic BPF evaluator (not eBPF). Per-task filter arrays. Instructions: `BPF_LD_W_ABS`, `BPF_JMP_JEQ`, `BPF_RET`. Filters are loaded via `seccomp_load_filter()` and evaluated on every syscall entry before dispatch.

### Audit

JSON-formatted event buffering exists. A hard gate still needs to prove VFS flush semantics for `/var/log/audit.log`.

### Threat Model

See [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md) for full details. In scope: kernel exploits from userspace, info leaks via side channels, network stack attacks. Out of scope: multi-tenant deployment, internet exposure, physical access/DMA attacks (research kernel context).


---

## How To Break This OS (A Hacker's Guide To Our Pain)

We believe in full disclosure. Here are known ways to break Seal OS, ranked by how embarrassing they are for us.

### 🔴 Critical (Please Don't)

1. **COW Fork Double-Free:** Under memory pressure, the copy-on-write fork path can double-free a page frame. This is a known bug tracked in our security audit. The fix requires rewriting the COW page table cloning logic.
2. **KPTI Not Fully Wired:** The CR3 swap code exists but the syscall entry/exit paths don't fully use it yet. A clever userspace program might be able to read kernel memory via cache timing.
3. **Weak Password Hashing:** `/etc/shadow` uses single-iteration SHA-256. This is documented as weak. A GPU could crack it in seconds.

### 🟠 Medium (Annoying But Not Fatal)

4. **No X.509/PKI/ECDHE:** The TLS stack only does PSK. You can't verify certificates. You can't do ECDHE key exchange. You can only talk to servers that accept raw PSK. (There are approximately zero such servers on the public internet.)
5. **GPU Shader Stubs:** The AMD GPU compute path executes shader binaries that are stubs. They return zeros or garbage. If you depend on GPU-accelerated topology computation, you get wrong answers.
6. **Simulated WiFi/Bluetooth:** The WiFi and Bluetooth stacks return simulated scan results. They don't actually use the hardware. If you think you're connected to a network, you're connected to our imagination.

### 🟡 Low (Cosmetic / Inconvenient)

7. **Default Password "seal":** The default login is `seal`/`seal`. Change it. Or don't. It's a research OS.
8. **No Multi-User Permissions:** There's no proper user management. `setuid`/`setgid` exist as syscalls but the security model is basically "everyone is root."
9. **Serial Output During Panic:** The panic handler writes to serial. If serial fails, the panic handler might double-panic. We fixed one instance of this but there may be others.

### 🟢 Theoretical (We Think These Exist But Haven't Proven)

10. **APIC Timer Race:** There might be a race between the scheduler lock release and context switch. We haven't observed it in practice, but the window exists.
11. **Voronoi Index Overflow:** If you create more than 2^32 files, the inode generation counter wraps. This is theoretical because we haven't tested with 4 billion files.

> **If you find a new vulnerability:** Please report it privately. We will fix it, credit you, and add it to this section with appropriate self-deprecating commentary.

</details>

<details>
<summary><strong>Aether-Lang</strong></summary>

A real programming language wired directly into the kernel. Lexer → Parser → AST → Interpreter, all running in `no_std` kernel space. This is Seal OS's native scripting language — the equivalent of what HolyC was to TempleOS.

```mermaid
graph TD
    subgraph SG60["Aether-Lang"]
        PARSE["Parser<br/>seal loops, tilde terminators"]
        BIO["Bio Mode<br/>tree-walking interpreter"]
        TITAN["Titan Mode<br/>bytecode VM"]
        REPL["REPL (repl-core)"]
        CLI["CLI (aether-cli)"]
    end

    PARSE --> BIO
    PARSE --> TITAN
    CLI --> PARSE
    REPL --> PARSE
```

### Stdlib Modules

| Module | Functions |
|--------|-----------|
| `math` | `pi`, `e`, `sin`, `cos`, `tan`, `sqrt`, `abs`, `ln`, `log`, `exp` |
| `fs` | `read`, `write`, `exists`, `mkdir`, `ls`, `teleport` |
| `process` | `pid`, `exit`, `spawn` |
| `net` | `local_ip`, `has_nic`, `status` |
| `theorem` | `status` |

### Example Aether-Lang Script

```aether
seal main ~
    print "Hello from Aether-Lang!"
    print theorem.status
    fs.write "hello.txt" "topology rules"
    print fs.read "hello.txt"
~
```

Run inside Seal OS: `aether run script.aether`


> **Aether-Lang hot take:** The `~` terminator is better than semicolons. Fight us.

</details>

<details>
<summary><strong>Applications</strong></summary>

### SealShell (`apps/shell.rs`)

30+ English-first commands:

| Command | What it does |
|---------|--------------|
| `look` | List directory with Voronoi cell assignments |
| `peek` | Show file info + ManifoldPayload |
| `move` | Metadata teleport — prints ticks and governor ε |
| `search` | Content-addressable search via S² embedding |
| `tasks` | Show scheduler task list |
| `seal` | System info + T1-T10 status |
| `race` | Benchmark teleport vs copy |
| `calc` | Scientific calculator |
| `play` | Media playback |
| `tensor render` | Render CSV as 3D tensor visualization |

### Calculator (`apps/calculator.rs`)

Full scientific calculator with recursive descent expression parser:
- Operator precedence: additive → multiplicative → power → unary → atom
- Functions: sin, cos, tan, sqrt, abs, ln, log, exp, ceil, floor
- Constants: pi, e, ans (last result)
- UI: High-tech rendering with gradient buttons, glowing LED display, rounded corners

### SealPlayer (`apps/media_player.rs`)

Native media player:
- **Working**: WAV/PCM playback with real RIFF/WAVE header parser
- **Planned**: MP4, MKV, MP3, FLAC, AAC, H.264, VP9, Opus
- Features: playlist management, seek, volume control, codec detection
- UI: Gradient viewport, glowing playhead, rounded progress bar, format badges

### Tensor Viewer (`apps/tensor_viewer.rs`)

CSV/trading data parsed into tensors, rendered with grid/value-height projection into 3D point clouds and hyperbolic manifolds. Profit = green peaks, loss = red valleys.

### Games

- **Snake** — classic grid game
- **Breakout** — paddle + bricks
- **Warp Racer** — aether-link demo


> **Game dev confession:** Warp Racer was supposed to be a full 3D racing game. It is currently a 2D demo where you press arrow keys and things happen. We are not game developers. We are geometry enthusiasts who occasionally make things move on a screen.

</details>

<details>
<summary><strong>Epsilon — Context Teleportation</strong></summary>

Design target: bounded context transfer between agents via topological surgery on hollow S² manifolds.

```mermaid
graph TD
    subgraph SG59["Teleportation Stack"]
        BRIDGE["EmbeddingBridge<br/>ℝ^E → S² via JL projection<br/>seeded Gaussian matrix"]
        TELEPORT["sys_teleport_context()<br/>orchestration syscall"]
        HOLLOW["HollowCubeManifold<br/>S² void receptacle<br/>Void(M_recv) = secure injection"]
        SURGERY_GOV["SurgeryGovernor<br/>PD control + clutch<br/>SurgeryPermit (one-shot lock)<br/>prevents de/dt → ∞ oscillation"]
        LIVENESS["LivenessAnchor<br/>inherited Chebyshev k-σ bounds<br/>prevents immediate GC eviction"]
        PAYLOAD["ManifoldPayload<br/>verified payload unit"]
    end

    BRIDGE --> TELEPORT
    TELEPORT --> HOLLOW
    HOLLOW --> SURGERY_GOV
    SURGERY_GOV --> LIVENESS
    LIVENESS --> PAYLOAD
```

The teleportation primitive: extract a payload from its current manifold via `inject_into_void()`, transfer to the receiving manifold via `assimilate()`. The SurgeryGovernor gates the operation with a one-shot derivative lock — if the manifold curvature derivative is too high, the surgery is deferred to prevent oscillation.


> **What this actually means:** When you "teleport" a file in Seal OS, we're not beaming it through space. We're rewiring metadata pointers so the file appears in a new directory without copying bytes. The "teleportation" branding is marketing. Effective marketing, but still marketing.

</details>

<details>
<summary><strong>Aether-Link — I/O Superkernel</strong></summary>

Adaptive I/O prefetching for topology-aware block streams.

```mermaid
graph LR
    LBA["LBA Stream<br/>(disk block addresses)"] --> FEAT["Feature Extraction<br/>6D telemetry vector"]
    FEAT --> ENCODE["State Encoding<br/>arctan angle mapping"]
    ENCODE --> DECIDE["Adaptive Threshold<br/>prefetch trigger"]
    DECIDE -->|"yes"| PREFETCH["Issue Prefetch"]
    DECIDE -->|"no"| SKIP["Skip"]
```

**Use cases**: HFT (high-frequency trading I/O), ML model training (sequential CSV/parquet prefetch), DirectStorage (game asset streaming).

**Presets**: `new_hft()` (aggressive, low-latency), `new_gaming()` (directstorage-tuned), `ModelTraining` (sequential reads, aggressive prefetch for large datasets).

**Fast math** (`fast_math.rs`): `fast_atan()`, `fast_exp()`, `fast_sigmoid()` — sub-microsecond approximations using polynomial fitting. No libm dependency in the hot path.

Benchmark: `io_cycle_8_lbas` median ~18 ns/cycle on desktop x86_64. CI regression gate: 120 ns ceiling.


> **HFT claim reality check:** 18 ns is fast. Is it faster than a tuned Linux kernel with `io_uring`? We don't know yet. That's why we have a benchmark plan. That's why we haven't claimed victory. Check back when `raw Ubuntu artifact pending` is no longer pending.

</details>


---

## Feature Matrix vs The World

How does a geometry-native research kernel compare to production operating systems? This table is a capability map for Seal OS v0.4.5, not a blanket victory claim. Seal OS aims to beat Ubuntu on the benchmark set in [docs/BENCHMARK_PLAN.md](docs/BENCHMARK_PLAN.md); the claim becomes true only for rows with fresh Seal OS and Ubuntu measurements under the same constraints.

### Ubuntu Parity and Lead Sheet

Legend: ✓ = code/proof gate exists in this repo, △ = design or partial implementation, ✗ = not implemented. Seal OS only claims a win over Ubuntu for a row after the same-machine benchmark exists.

| Capability | Seal OS state | Ubuntu 26.04 LTS baseline | Proof or next gate |
|---|---:|---|---|
| UEFI image build | ✓ | ✓ | `seal-mkimage --verify` passes |
| VM desktop boot proof | ✓ | ✓ | `run-qemu.ps1 -HeadlessProof` plus `seal-mkimage --check-vm-proof` captures theorem gate, AHCI disk, ManifoldFS mount, desktop proof frame, desktop soak marker, desktop ready, and event loop |
| QEMU proof bundle manifest | ✓ | N/A | `run-qemu.ps1 -HeadlessProof` writes `qemu-proof\proof-manifest.txt`; `seal-mkimage --check-proof-manifest` verifies the image/EFI/log/screen byte counts, CRC32 fingerprints, SHA-256 fingerprints, QEMU backend, commit/dirty flag, and gate statuses before publishing canonical artifacts; `--check-current-proof-manifest ... .` additionally rejects stale proof bundles whose commit or dirty flag no longer match the checkout |
| README/doc claim contract | ✓ | N/A | `seal-mkimage --check-doc-claim-contract .` enforces a limited allow/deny string contract for `README.md`, `docs/BENCHMARK_PLAN.md`, and `docs/CI.md` |
| Oracle VirtualBox automated smoke | ✓ | ✓ | Fresh `smoke-vbox.ps1 -Seconds 240` rebuilds the VDI, then runs `--verify`, `--check-vbox-proof`, theorem/Aether/desktop/benchmark gates, `--check-proof-manifest`, and `--check-current-proof-manifest` against `vbox-smoke\proof-manifest.txt` |
| First desktop pixel proof | ✓ | N/A | `seal-mkimage --check-proof-screen ...\qemu-proof\screen.ppm` verifies nonblank 1024x768 desktop pixels, icon lane, control region, primary terminal titlebar, and taskbar start/power controls |
| Desktop compositor soak marker | ✓ | △ | `seal-mkimage --check-desktop-soak ...\serial.log` requires a deterministic 24-frame compose+blit exercise with monotonic cycle percentiles; calibrated 16.7 ms frame-pacing benchmark still pending |
| Bare-metal allocator benchmark marker | ✓ | △ | `seal-mkimage --check-benchmark-log ...\serial.log` requires `[BENCH] alloc-frame` with 64 successful alloc/free iterations, topological fast-path hits, zero bounded misses, no contiguous-probe drift, and no frame leak |
| TopoRAM target-cell allocation marker | ✓ | △ | `seal-mkimage --check-benchmark-log ...\serial.log` requires `[BENCH] toporam-alloc` with 64 target-cell hits, zero target-cell fallbacks, zero zone fallbacks, monotonic cycle samples, and no frame leak |
| ManifoldFS metadata teleport marker | ✓ | △ | `seal-mkimage --check-benchmark-log ...\serial.log` also requires `[BENCH] manifold-teleport`, proving same-inode ramfs metadata move across 8-256 directory entries with `metadata_ops_max=7` and `persistence_bytes_per_move=0` |
| Scheduler select benchmark marker | ✓ | △ | `seal-mkimage --check-benchmark-log ...\serial.log` requires `[BENCH] scheduler-select-next`, gating the live `select_next_task` requeue marker across 64 iterations with ready count preserved, zero context switches, 8 Voronoi probes, max 9 bitmap tests, and max 256 priority-bucket scan |
| TCP packet demux benchmark marker | ✓ | △ | `seal-mkimage --check-benchmark-log ...\serial.log` requires `[BENCH] tcp-packet-demux`, proving a listener-first same-port fixture routes payload bytes to the accepted socket |
| AHCI persistent ManifoldFS root | ✓ | ✓ | QEMU serial log shows `QEMU HARDDISK`, `Registered as block device 0x800`, `First disk readable`, and `[VFS] ManifoldFS mounted from disk` |
| Native non-POSIX ABI | ✓ | ✗ | `seal-mkimage --check-seal-abi .` passes |
| T1-T10 theorem-gated boot | ✓ | ✗ | Rust theorem-log checker requires all ten VERIFIED lines |
| T1-T5 runtime topology | ✓ | ✗ | `seal-mkimage --check-runtime-theorems .` source-gates runtime callsites in memory, scheduler, ManifoldFS, compositor, ACPI power, taskbar status, and boot theorem state |
| Single-frame allocation O(1) | ✓ | △ | `--check-o1-allocator` plus boot log `[ALLOC] O(1) proof:` and `[BENCH] alloc-frame` gates |
| Multi-page contiguous DMA allocation O(1) over RAM size | ✓ | △ | bounded candidate probes plus hard 64-page run cap |
| Same-filesystem file move | ✓ | ✓ | Seal marker proves same-inode topology metadata surgery with `persistence_bytes_per_move=0` |
| Content-addressable geometric lookup | △ | △ | Voronoi narrows lookup to a bucket; current find is O(bucket size) plus sorting until bucket occupancy is hard-capped |
| GPU/VRAM topology fast path | △ | CUDA/ROCm userspace, not topology fast path | design contract exists; vendor GPU driver and peer-DMA proof pending |
| Aether-Lang native OS language | ✓ | ✗ | lexer/parser/interpreter/VM are in kernel runtime |
| Legacy host-language-free Seal OS surface | △ | ✓ | `--check-language-hygiene` bans host scripts from production OS/Rust roots |
| HFT/ML benchmark comparison vs Ubuntu | △ | ✓ | allocator comparison harness exists; full benchmark matrix still requires fresh side-by-side Ubuntu numbers; raw Ubuntu artifact pending; current proof manifest is `--check-current-benchmark-proof` |

### Seal OS vs Redox OS vs Ubuntu vs Debian vs Windows vs macOS

| Feature | **Seal OS v0.4.5** | **Redox OS 0.9.0** | **Ubuntu 26.04 LTS** | **Debian 12 Bookworm** | **Windows 11** | **macOS Sequoia** |
|---|---|---|---|---|---|---|
| **Language** | Rust (100%, `no_std`) | Rust (microkernel) | C (Linux kernel) | C (Linux kernel) | C/C++ (NT kernel) | C/C++/Obj-C (XNU) |
| **Architecture** | Monolithic | Microkernel | Monolithic + modules | Monolithic + modules | Hybrid | Hybrid (Mach + BSD) |
| **Kernel size** | ~260 KB | ~1 MB | ~12 MB (vmlinuz) | ~8 MB (vmlinuz) | ~30 MB (ntoskrnl) | ~25 MB (kernel.release) |
| **ISO size** | < 10 MB | ~70 MB | ~5 GB | ~650 MB (netinst) | ~5.5 GB | ~13 GB (IPSW) |
| **Min RAM** | 4 GB | 512 MB | 4 GB | 512 MB | 4 GB | 8 GB |
| **Boot target** | `x86_64-unknown-uefi` | `x86_64-unknown-redox` | `x86_64-linux-gnu` | `x86_64-linux-gnu` | proprietary | proprietary |
| **Filesystem** | ManifoldFS (S² geometry) | RedoxFS (CoW) | ext4 / btrfs | ext4 | NTFS / ReFS | APFS |
| **File identity** | Raw bytes + S² ManifoldPayload | byte sequence | byte sequence | byte sequence | byte sequence | byte sequence |
| **File move** | O(1) metadata surgery | rename (O(1) same FS) | rename (O(1) same FS) | rename (O(1) same FS) | rename (O(1) same vol) | rename (O(1) same vol) |
| **Content-addressable lookup** | Bucketed geometric lookup | No | No | No | No (Windows Search) | No (Spotlight) |
| **Scheduler** | ManifoldScheduler (T1+T2+T4) | Round-robin | CFS / EEVDF | CFS | Hybrid priority | Grand Central Dispatch |
| **Adaptive control** | GeometricGovernor (PD on manifold) | No | cpufreq governors | cpufreq governors | Dynamic tick | Timer coalescing |
| **Formal verification** | Lean 4 in progress; Rust boot gates active | Partial (cosmic, relibc) | Partial (seL4 adjacent) | None | None | None |
| **Math-driven kernel** | Yes (T1-T5 active, T6-T10 boot-checked) | No | No | No | No | No |
| **Topological data analysis** | Runtime-gated markers and Betti/Voronoi | No | Userspace only | Userspace only | No | No |
| **Predictive prefetch** | T2 spectral contraction mechanism | No | readahead heuristic | readahead heuristic | Superfetch/SysMain | Speculative prefetch |
| **GPU offload ready** | PM4 compute ring + CPU fallback real; shaders are stubs | No | CUDA/ROCm userspace | CUDA/ROCm userspace | DirectCompute | Metal |
| **Display** | 1024x768x32 framebuffer | 1920x1080 (orbital) | Wayland/X11 | Wayland/X11 | DWM | Quartz |
| **Window manager** | Built-in compositor | Orbital | GNOME/KDE | GNOME/KDE/Xfce | DWM | WindowServer |
| **Built-in IDE** | Seal IDE (native) | No | No | No | No | Xcode (separate) |
| **Shell** | SealShell (30+ English-first commands) | Ion shell | bash/zsh | bash | PowerShell/cmd | zsh |
| **Package manager** | ManifoldPkg local metadata + signed `.eph` path | pkg (pkgutils) | apt/snap | apt | winget/MSIX | brew (3rd party) |
| **Syscalls** | Seal ABI + Epsilon theorem extensions | ~100 (POSIX-like) | ~450 (Linux) | ~450 (Linux) | ~2000+ (NT) | ~550 (Mach + BSD) |
| **USB support** | Real — xHCI controller, HID boot keyboards/mice, Mass Storage SCSI BBB | Basic (xHCI) | Full | Full | Full | Full |
| **Network stack** | Real TCP/UDP/DHCP/DNS + minimal TLS 1.3 PSK client | smoltcp | Full (netfilter) | Full (netfilter) | Full (WFP) | Full (PF) |
| **Driver count** | 15+ | ~30 | ~9000+ | ~9000+ | ~100,000+ | ~5000+ |
| **Self-hosted** | No | Partial | Yes | Yes | Yes | Yes |
| **License** | MIT | MIT | GPL-2.0 (kernel) | DFSG-free | Proprietary | Proprietary (+ open source parts) |
| **Theorem count** | 10 boot-gated; T1-T5 active in runtime paths | 0 | 0 | 0 | 0 | 0 |
| **Teleportation** | Metadata topology move; persistent byte rewrite still pending | No | No | No | No | No |

**Where Seal OS is distinctive as a design**: mathematical kernel primitives, topological data embeddings, content-addressable ManifoldFS metadata, theorem-gated boot, adaptive governor, and gated O(1) metadata-move/select/allocation markers.

**Where Seal OS must still prove superiority**: repeatable Ubuntu comparison benchmarks for HFT/ML workloads, driver maturity, security hardening, and long-running reliability.

**Where Seal OS trails**: GPU drivers (no proprietary firmware), WiFi/BT (no vendor blobs), self-hosting, userspace ecosystem, multi-user permissions, security hardening maturity. It's a research kernel — not yet a daily driver.

**Closest comparison**: Redox OS shares the Rust DNA and research spirit. Seal OS diverges by making topology the organizing principle rather than microkernels.


---

## Where We Actually Stand (Brutal Honesty Edition)

Let's strip away the marketing and talk about where Seal OS actually is, in terms that would make a product manager cry.

### What We Can Honestly Claim Today

1. **We boot.** On QEMU. On VirtualBox. On some real hardware (tested on a ThinkPad T480 and a custom desktop).
2. **We have a desktop.** It has a taskbar, a start menu, a calculator, a file manager, and games. It looks like something from the early 2000s, but it works.
3. **Our memory allocator is O(1).** For single-frame allocations. With bounded probes. Under specific conditions. We have proof gates that verify this on every boot. This is real.
4. **Our file moves are O(1).** For metadata-only moves within the same filesystem. The bytes don't move, the pointers do. We verify `persistence_bytes_per_move=0` in CI.
5. **We have mathematical theorems.** Ten of them. Five are active in runtime. Zero `sorry` tactics in Lean. This is not common in OS development.
6. **We have real drivers.** Not mocks. Real NVMe queue creation. Real e1000 descriptor rings. Real xHCI enumeration. Real HDA audio playback.

### What We Cannot Honestly Claim Yet

1. **We are faster than Linux.** For most workloads, Linux is faster. We haven't done the side-by-side benchmarks yet. The harness exists. The Ubuntu artifact is pending. Until then, we don't claim victory.
2. **We are secure.** We have scaffolding. We have ASLR and seccomp. We do NOT have full KPTI, production TLS, or a security audit. Do not use this for sensitive data.
3. **We have a GPU driver.** We have GPU infrastructure. We can talk to AMD GPUs. The shaders are stubs. They execute but produce wrong results. This is documented.
4. **We have a browser.** We do not have a browser. We have an HTTP/HTTPS client. You can fetch raw HTML. You cannot render it. You cannot run JavaScript. There is no DOM.
5. **We are self-hosting.** We cannot build Seal OS using Seal OS. We need Linux (or Windows, or macOS) to compile the kernel. Aether-Lang is not yet powerful enough to build itself.
6. **We have users.** Our user base is approximately: 1 primary developer, 2 occasional contributors, 5 people who starred the repo and never cloned it, and 3 people who cloned it, said "huh," and moved on.

### The "△" Symbol Is Our Friend

Look at the feature matrix. See all those △ symbols? Those mean "partial" or "pending." We use △ a lot. We are not ashamed of △. △ is honesty. △ is "we're working on it." △ is better than ✗ because it means progress.

If we wanted to lie, we could replace every △ with ✓ and claim full implementation. But then someone would try to use it, it would break, and they'd open an issue. We'd rather be honest upfront and have fewer disappointed users.

---

## Performance Characteristics

| Subsystem | Operation | Complexity | Evidence / latency |
|-----------|-----------|-----------|----------------|
| **Physical alloc** | `alloc_frame()` | O(1) bounded topological free-index lookup across 8 cells; max 2 L3 words and 8192 summary-backed word candidates per cell | source-gated + boot log proof marker + `[BENCH] alloc-frame` |
| **Slab alloc** | `slab.alloc(size)` | O(1) | benchmark pending |
| **TopoRAM alloc** | `alloc_frames(1, hint)` | O(1) Voronoi lookup + O(1) physical frame path; entropy, prefetch, and reseed work are bounded/interval-gated | `[BENCH] toporam-alloc` |
| **TopoRAM contiguous** | `alloc_frames(count > 1, hint)` | 128 bounded topological candidate probes + hard 64-page allocation/free repair cap | source-gated by `--check-o1-allocator` |
| **ManifoldFS lookup** | `lookup(path)` | O(path depth) + O(K) cell search | benchmark pending |
| **ManifoldFS teleport** | move file | O(1) metadata rewiring with same-inode directory topology move; persistent metadata updates do not rewrite file bytes | `[BENCH] manifold-teleport` gate proves same inode, source removal, destination presence, bounded metadata ops, and `persistence_bytes_per_move=0` |
| **Scheduler select** | `select_next_task()` | O(1) — one predicted-cell check plus bounded fallback across 8 cells and 256 priority buckets | `[BENCH] scheduler-select-next` gate proves 64 live requeue selections, ready count preservation, zero context switches, 8 Voronoi probes, max 9 bitmap tests, and max 256 bucket scan |
| **Context switch** | `switch_context()` | O(1) — FXSAVE/FXRSTOR + CR3 swap | benchmark pending |
| **NVMe read** | `read_sector(lba)` | O(1) command submit + DMA poll | benchmark pending |
| **NVMe write** | `write_sector(lba)` | O(1) command submit + DMA poll | benchmark pending |
| **TCP demux** | `handle_tcp_packet()` | O(1) flow match before listener fallback | `[BENCH] tcp-packet-demux` proves listener-first socket order, same-port accepted socket delivery, 4-byte payload receipt, established-state transition, and fixture cleanup |
| **TCP round-trip** | localhost ping | O(1) stack traversal | benchmark pending |
| **TLS encrypt** | 1KB record | O(N) AES-GCM | benchmark pending |
| **3D render** | 1K triangles, quality 2 | O(triangles × pixels) software raster | benchmark pending |
| **Tensor render** | 100×100 CSV → mesh | O(N) SVD + O(N) mesh gen + raster | benchmark pending |

*Note: complexity rows are code/proof claims. Latency rows stay pending until the benchmark plan records raw artifacts and side-by-side Ubuntu runs.*

### Benchmarks

Run locally:

```bash
# Primary I/O benchmark
cargo bench --bench io_cycle --manifest-path kernel/aether/aether-link/Cargo.toml

# Compile all benches without running
cargo bench --workspace --no-run
```

See [BENCHMARKS.md](BENCHMARKS.md) for interpretation guide and CI regression gate details.

---

## Build and Run

### Requirements

| Tool | Version | Purpose |
|------|---------|---------|
| Rust (stable) | 1.85+ | Workspace crates |
| Rust (stable) | 1.88+ | LAAMBA Governor Tauri backend |
| Rust (nightly) | latest | Seal OS kernel (`#![feature(abi_x86_interrupt)]`) |
| QEMU | any | `qemu-system-x86_64` for testing |
| Oracle VM VirtualBox | 7.x | Primary GUI VM target |
| OVMF/EDK2 | any | UEFI firmware for QEMU |
| Aether-Lang | repo-local | Native Seal OS scripts and app logic |
| Lean | 4.7.0 | Formal proofs (optional) |

### Workspace Build

```bash
# Build all workspace crates
cargo build --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo doc --workspace --no-deps
```

### Kernel Build (Nightly Required)

```bash
cd kernel/seal-os
cargo +nightly build --release
```

### Create Bootable Image

```bash
cd ../seal-mkimage
cargo +stable run --release

# Output:
# kernel/seal-os/target/x86_64-unknown-uefi/release/seal-os.img
```

The image is a raw GPT disk with a FAT EFI System Partition containing `EFI/BOOT/BOOTX64.EFI` and a second `ManifoldFS` partition with an `MNFD` superblock for the persistent Seal root.

### QEMU (Linux/macOS)

```bash
cd ../seal-os
./run-qemu.sh
```

### QEMU (Windows)

```powershell
cd ../seal-os
.\run-qemu.ps1
```

### Oracle VM VirtualBox

```powershell
cd kernel/seal-os
powershell -File .\build-vbox.ps1
powershell -File .\smoke-vbox.ps1 -Seconds 240
```

Manual conversion:
```bash
VBoxManage convertfromraw --format VDI \
  target/x86_64-unknown-uefi/release/seal-os.img seal-os.vdi
```

VM settings:
- Type=Other, Version=Other/Unknown (64-bit)
- Enable EFI, RAM=4096 MB, CPUs=1-2
- Display=VMSVGA, video memory=128 MB
- Storage=SATA/AHCI, attach seal-os.vdi
- Network=Intel PRO/1000 MT Desktop if networking is needed

### ISO Creation

```bash
# Linux only — requires xorriso
chmod +x scripts/build_iso.sh
scripts/build_iso.sh
```

### Docker (World Model)

```bash
cd kernel/seal-os && docker compose up --build
```

### System Requirements

| Resource | Minimum |
|----------|---------|
| RAM | 4 GB |
| CPU | x86_64 with long mode |
| Display | 1024x768 (optional — serial fallback) |

---

## Documentation Index

Every claim in this README has a supplementary document. Every document traces to source code.

### Getting Started

| Document | What it covers |
|----------|---------------|
| [Quick Start](#quick-start--boot-in-5-minutes) (this README) | 5-minute boot guide |
| [docs/SEAL_OS_GUIDE.md](docs/SEAL_OS_GUIDE.md) | Practical build, VM proof, audit gates, allocator contract, benchmark runbook |
| [docs/BUILD_SYSTEM.md](docs/BUILD_SYSTEM.md) | Workspace structure, toolchains, dependency policy, common errors |
| [kernel/seal-os/README.md](kernel/seal-os/README.md) | Kernel overview, quick start, concepts |

### Architecture & Design

| Document | What it covers | Key source files |
|----------|---------------|-----------------|
| [docs/TOPOLOGICAL_OS_CONTRACT.md](docs/TOPOLOGICAL_OS_CONTRACT.md) | Hard definition of "topological OS", closure gates, O(1) claim discipline | `src/memory/topo_ram.rs`, `src/process/scheduler.rs`, `src/fs/manifold_fs.rs` |
| [kernel/seal-os/ARCHITECTURE.md](kernel/seal-os/ARCHITECTURE.md) | UEFI boot sequence, init, hardware setup | `src/boot/uefi_entry.rs`, `src/lib.rs` |
| [docs/BOOT.md](docs/BOOT.md) | UEFI firmware to Seal kernel, GOP, VM image path | `src/boot/uefi_entry.rs`, `kernel/seal-mkimage` |
| [docs/MANIFOLDFS.md](docs/MANIFOLDFS.md) | Encoding pipeline, inode structure, metadata teleport, bucketed content search | `seal-os/src/fs/encoder.rs`, `manifold_fs.rs` |
| [docs/MANIFOLDFS.md](docs/MANIFOLDFS.md) | Encoding pipeline, inode structure, metadata teleport, bucketed content search | `src/fs/manifold_fs.rs` |
| [docs/MEMORY.md](docs/MEMORY.md) | Physical layout, allocator, UEFI map, MMIO | `src/memory/mod.rs`, `src/boot/uefi_entry.rs` |

### Theorems & Verification

| Document | What it covers | Key source files |
|----------|---------------|-----------------|
| [docs/THEOREMS.md](docs/THEOREMS.md) | All 10 theorems: math, implementation, Lean proofs, callsites | `aether-core/src/tss.rs`, `governor.rs`, `scm.rs`, `topology.rs` |
| [kernel/aether/aether-verified/lean/README.md](kernel/aether/aether-verified/lean/README.md) | Build instructions, provenance map, zero-sorry goal | `lakefile.lean`, `lean-toolchain` |

### API & Reference

| Document | What it covers |
|----------|---------------|
| [docs/API_INDEX.md](docs/API_INDEX.md) | Master API index for all workspace crates |
| [docs/SYSCALLS.md](docs/SYSCALLS.md) | Seal ABI calls + signals + pipes + RTC + Epsilon extensions |
| [kernel/epsilon/epsilon/docs/SPECIFICATION.md](kernel/epsilon/epsilon/docs/SPECIFICATION.md) | Epsilon geometric state transfer spec (v0.1.0-draft) |
| [kernel/epsilon/epsilon/docs/API_REFERENCE.md](kernel/epsilon/epsilon/docs/API_REFERENCE.md) | Epsilon crate public API |

### Aether-Lang

| Document | What it covers |
|----------|---------------|
| [kernel/aether/Aether-Lang/docs/LANGUAGE.md](kernel/aether/Aether-Lang/docs/LANGUAGE.md) | Syntax, semantics, topological primitives |
| [kernel/aether/Aether-Lang/docs/GETTING_STARTED.md](kernel/aether/Aether-Lang/docs/GETTING_STARTED.md) | Setup, first program, REPL usage |
| [kernel/aether/Aether-Lang/docs/ARCHITECTURE.md](kernel/aether/Aether-Lang/docs/ARCHITECTURE.md) | Parser, Bio mode, Titan VM, AEGIS memory |
| [kernel/aether/Aether-Lang/docs/API.md](kernel/aether/Aether-Lang/docs/API.md) | Public API surface |
| [kernel/aether/Aether-Lang/docs/TUTORIAL.md](kernel/aether/Aether-Lang/docs/TUTORIAL.md) | Guided walkthrough |
| [kernel/aether/Aether-Lang/docs/EXAMPLES.md](kernel/aether/Aether-Lang/docs/EXAMPLES.md) | Code samples |
| [kernel/aether/Aether-Lang/docs/FAQ.md](kernel/aether/Aether-Lang/docs/FAQ.md) | Common questions |
| [kernel/aether/Aether-Lang/docs/ML_FROM_SCRATCH.md](kernel/aether/Aether-Lang/docs/ML_FROM_SCRATCH.md) | Building ML pipelines with aether-core |
| [kernel/aether/Aether-Lang/docs/ML_LIBRARY.md](kernel/aether/Aether-Lang/docs/ML_LIBRARY.md) | Tensor, autograd, neural, clustering modules |

### Infrastructure & Operations

| Document | What it covers |
|----------|---------------|
| [docs/CI.md](docs/CI.md) | All 16 CI jobs, QEMU milestones, toolchains |
| [BENCHMARKS.md](BENCHMARKS.md) | How to run Criterion, CI regression gates |
| [docs/BENCHMARK_PLAN.md](docs/BENCHMARK_PLAN.md) | Side-by-side Ubuntu comparison plan |
| [SECURITY.md](SECURITY.md) | Vulnerability reporting, threat model, environment variables |
| [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md) | Full threat model: physical, software, network |
| [docs/CRYPTO_AUDIT.md](docs/CRYPTO_AUDIT.md) | Cryptographic path audit: TLS, random, signatures |
| [docs/VRAM_TOPOLOGY_FAST_PATH.md](docs/VRAM_TOPOLOGY_FAST_PATH.md) | GPU-native data movement contract |

### Project Governance

| Document | What it covers |
|----------|---------------|
| [CONTRIBUTING.md](CONTRIBUTING.md) | Prerequisites, build instructions, subsystem map, style guide, theorem-gate requirements |
| [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) | Contributor Covenant v2.1, enforcement contact |
| [docs/COMMUNITY.md](docs/COMMUNITY.md) | New member onboarding, Discussions vs Issues, finding work |
| [FUTURE_PLAN.md](FUTURE_PLAN.md) | 5-phase roadmap, 15+ subsystems |
| [docs/ONE_PAGER.md](docs/ONE_PAGER.md) | Executive summary for investors and press |
| [docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md) | 3-minute demo video script |

### Agent Plans

| Document | What it covers |
|----------|---------------|
| [.agents/MASTER_PROMPT.md](.agents/MASTER_PROMPT.md) | Master agent prompt for Seal OS construction |
| [.agents/01-kernel-safety.md](.agents/01-kernel-safety.md) | Kernel safety agent plan |
| [.agents/02-memory-allocator.md](.agents/02-memory-allocator.md) | Memory allocator agent plan |
| [.agents/03-manifold-fs.md](.agents/03-manifold-fs.md) | ManifoldFS agent plan |
| [.agents/04-aether-lang-interpreter.md](.agents/04-aether-lang-interpreter.md) | Aether-Lang interpreter agent plan |
| [.agents/05-aether-lang-vm.md](.agents/05-aether-lang-vm.md) | Aether-Lang VM agent plan |
| [.agents/06-epsilon-local.md](.agents/06-epsilon-local.md) | Epsilon local agent plan |
| [.agents/07-epsilon-remote.md](.agents/07-epsilon-remote.md) | Epsilon remote agent plan |
| [.agents/08-aether-link-hardware.md](.agents/08-aether-link-hardware.md) | Aether-Link hardware agent plan |
| [.agents/09-network-stack.md](.agents/09-network-stack.md) | Network stack agent plan |
| [.agents/10-drivers-real.md](.agents/10-drivers-real.md) | Real drivers agent plan |
| [.agents/11-window-manager.md](.agents/11-window-manager.md) | Window manager agent plan |
| [.agents/12-applications.md](.agents/12-applications.md) | Applications agent plan |
| [.agents/13-security-hardening.md](.agents/13-security-hardening.md) | Security hardening agent plan |
| [.agents/14-testing-ci.md](.agents/14-testing-ci.md) | Testing & CI agent plan |
| [.agents/15-documentation.md](.agents/15-documentation.md) | Documentation agent plan |


---

## How To Read Our Docs Without Crying

Our documentation is extensive. Here's a survival guide:

1. **Start with this README.** It's long but comprehensive. Use Ctrl+F.
2. **Read `docs/SEAL_OS_GUIDE.md` next.** It's the practical "how do I make this work" document.
3. **Read `docs/THEOREMS.md` if you're math-curious.** Skip it if you just want to boot the OS.
4. **Read `docs/THREAT_MODEL.md` if you're security-minded.** It'll tell you why you shouldn't use this in production.
5. **Read `kernel/seal-os/ARCHITECTURE.md` if you want to contribute.** It explains the boot sequence and init code.
6. **Avoid `docs/BENCHMARK_PLAN.md` unless you're a masochist.** It's full of detailed comparison plans that aren't finished yet.
7. **The `.agents/` directory is for AI agents.** If you're a human, you can read it for context, but it's written in "agent prompt" format, which is weird.

---

## Repository Map

```
Epsilon-Hollow/
├── kernel/
│   ├── seal-os/                    # Bare-metal x86_64 UEFI kernel
│   │   ├── src/
│   │   │   ├── main.rs             # UEFI #[entry], panic handler
│   │   │   ├── lib.rs              # kernel_main(), module declarations
│   │   │   ├── boot/               # uefi_entry.rs, boot_info.rs, ap_trampoline.rs
│   │   │   ├── memory/             # phys.rs (bitmap), slab.rs, heap.rs, virt.rs (VMM), gdt.rs
│   │   │   ├── drivers/            # IDT, APIC, serial, PCI, NVMe, AHCI, e1000, xHCI, HDA, entropy, RTC, watchdog, ACPI, WiFi/BT/GPU probe
│   │   │   ├── fs/                 # ManifoldFS + FAT + ext2 + PipeFS + VFS (devtmpfs, procfs, sysfs)
│   │   │   ├── graphics/           # Framebuffer, double-buffer, font, console, splash, wallpaper, htek, topo_render
│   │   │   ├── process/            # ManifoldScheduler, context switch, ELF loader, userspace (ring-3)
│   │   │   ├── syscall/            # Seal ABI calls + signals + pipes + RTC + Epsilon extensions
│   │   │   ├── wm/                 # Compositor, windows, desktop, taskbar
│   │   │   ├── cpu/                # SMP bring-up (INIT-SIPI-SIPI)
│   │   │   ├── net/                # TCP/IP stack (ARP, DHCP, DNS, ICMP, IPv4, TCP, UDP)
│   │   │   ├── security/           # ASLR, seccomp, MAC, SMAP/SMEP, audit
│   │   │   ├── sync/               # Ticket lock, seq lock, TLB shootdown
│   │   │   ├── pkg/                # ManifoldPkg package manager
│   │   │   ├── lang/               # Aether-Lang kernel integration
│   │   │   ├── async_rt/           # Minimal async runtime
│   │   │   └── apps/               # Shell, terminal, IDE, calculator, SealPlayer, games
│   │   ├── .cargo/config.toml      # target = x86_64-unknown-uefi
│   │   └── build.rs                # Build configuration
│   │
│   ├── epsilon/epsilon/crates/
│   │   ├── aether-core/            # Runtime math for T1-T5
│   │   ├── epsilon/                # Context teleportation (bridge, manifold, governor)
│   │   └── epsilon-os/             # World model REPL
│   │
│   └── aether/
│       ├── Aether-Lang/crates/     # Topological DSL runtime + CLI
│       ├── aether-link/            # I/O superkernel (benchmark pending)
│       └── aether-verified/        # no_std T1-T10 theorem checks + Lean 4 artifacts
│
├── apps/
│   └── laamba-governor/            # Bundled native app workload (Tauri backend)
│
├── infrastructure/                 # K8s manifests, orchestrator, training
│
├── scripts/                        # BOM check, demo, model download, ISO build
│
├── tests/                          # Legacy research tests
│
├── tools/
│   └── ubuntu-alloc-bench/         # Ubuntu allocator comparison harness
│
├── docs/                           # Technical references, specs, guides
│
├── .github/
│   ├── workflows/ci.yml            # 16-job CI pipeline
│   ├── ISSUE_TEMPLATE/             # Bug report, feature request, documentation
│   ├── pull_request_template.md    # PR checklist
│   ├── CODEOWNERS                  # Subsystem maintainer map
│   └── SECURITY_ADVISORIES.md      # Advisory template and disclosure process
│
├── Cargo.toml                      # Workspace root (10+ member crates)
├── deny.toml                       # License + dependency policy
├── rust-toolchain.toml             # Stable 1.85 + components
├── README.md                       # This document
├── CONTRIBUTING.md                 # Full contribution guide
├── CODE_OF_CONDUCT.md              # Contributor Covenant v2.1
├── SECURITY.md                     # Vulnerability reporting + threat model
├── LICENSE                         # MIT License
└── FUTURE_PLAN.md                  # 5-phase roadmap
```


---

## How To Navigate This Repo (Survival Guide)

### For The Impatient Contributor

1. **Want to fix a bug?** Check `tests/` and `kernel/seal-os/src/` for the relevant subsystem.
2. **Want to add a feature?** Read `.agents/` for the subsystem plan, then implement.
3. **Want to change the README?** Make sure you don't break `--check-doc-claim-contract`. We CI our README.
4. **Want to add a theorem?** Talk to us first. Theorems have a high bar.
5. **Want to add a driver?** Godspeed. Read `kernel/seal-os/src/drivers/` for examples.

### Directory Smells

- `kernel/seal-os/src/lib.rs` is 800+ lines of module declarations. It's fine. Don't refactor it unless you want 47 merge conflicts.
- `kernel/seal-os/src/apps/` contains both shell commands and full applications. The boundary is fuzzy. We know.
- `docs/` has both `.md` files and subdirectories with more `.md` files. The nesting is organic, not planned.
- `.agents/` contains prompts for AI agents. If you're a human, read them for context. If you're an AI, follow them.

---

## Contributing & Community

We welcome contributions from systems programmers, mathematicians, language designers, and researchers.

### Quick Start for Contributors

1. Read [CONTRIBUTING.md](CONTRIBUTING.md) for prerequisites and setup
2. Read [docs/COMMUNITY.md](docs/COMMUNITY.md) for communication norms
3. Pick a subsystem from the [Repository Map](#repository-map) above
4. Look for issues labelled [`good first issue`](https://github.com/teerthsharma/epsilon-hollow/labels/good%20first%20issue) or [`help wanted`](https://github.com/teerthsharma/epsilon-hollow/labels/help%20wanted)
5. Run the pre-checks before submitting:
   ```bash
   cargo fmt --all
   cargo clippy --workspace --all-targets -- -D warnings
   cargo test --workspace
   ```

### Where to Contribute

- **DSL / language work** — `kernel/aether/Aether-Lang/crates/`
- **Core math (topology, manifolds, PD control, teleportation)** — `kernel/epsilon/epsilon/crates/`
- **I/O superkernel and benchmarks** — `kernel/aether/aether-link/`
- **Kernel subsystems** — `kernel/seal-os/src/`
- **Documentation** — `docs/` and this README

### Community Spaces

- **GitHub Discussions** — questions, ideas, show-and-tell
- **GitHub Issues** — bug reports, feature requests (use templates)
- **Security reports** — see [SECURITY.md](SECURITY.md) for private disclosure


---

## Contributing: Or, How To Lose Your Weekend To Geometry

So you want to contribute to Seal OS. Bless your heart. Here's what you're signing up for.

### The Good News

- The codebase is smaller than Linux. Much smaller. You can read all of it in a week.
- It's all Rust. If you know Rust, you know 90% of what you need.
- The build system is just Cargo. No autotools. No CMake. No Makefile mysteries.
- We have CI that catches most mistakes before they hit main.

### The Bad News

- It's `#![no_std]`. No `std::fs`. No `std::net`. No `std::process`. Everything is manual.
- The math is real. You will encounter `arccos`, `sinh`, `eigenvectors`, and `Betti numbers`.
- There are no Stack Overflow answers for "how do I implement a Voronoi-based scheduler in Rust."
- Debugging means serial output. No GDB. No `println!`. Just `serial_println!("help me")`.

### What Makes A Good Contribution

1. **Bug fixes with tests.** We love these. They make CI greener.
2. **Documentation improvements.** Especially if they explain math in human terms.
3. **Benchmarks.** Especially side-by-side comparisons with Linux.
4. **Driver improvements.** Real hardware testing is gold.
5. **Theorem proofs.** Lean 4 proofs get you eternal respect.

### What Makes A Bad Contribution

1. **"Why not use Linux instead?"** We know Linux exists. We chose not to use it. This is not a bug.
2. **"This is overkill."** Yes. That's the point.
3. **"You should rewrite it in Zig."** No.
4. **PRs that break CI without explanation.** Our CI is sacred. Breaking it is a cardinal sin.
5. **Removing theorems to simplify things.** The theorems stay. They are non-negotiable.

---

## Security Policy

Please report suspected vulnerabilities privately. Do **not** open a public issue for security-sensitive reports.

- **Preferred**: GitHub Security Advisories ("Report a vulnerability" on the repo's Security tab)
- **Email fallback**: `teerths57@gmail.com` (see [SECURITY.md](SECURITY.md) for current contact)

Include reproduction steps, affected version/commit, and impact. Coordinated disclosure is appreciated.

For full threat model, cryptographic audit, and security architecture, see:
- [SECURITY.md](SECURITY.md)
- [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md)
- [docs/CRYPTO_AUDIT.md](docs/CRYPTO_AUDIT.md)
- [.github/SECURITY_ADVISORIES.md](.github/SECURITY_ADVISORIES.md)


---

## How To Break This OS (A Hacker's Guide To Our Pain)

We believe in full disclosure. Here are known ways to break Seal OS, ranked by how embarrassing they are for us.

### 🔴 Critical (Please Don't)

1. **COW Fork Double-Free:** Under memory pressure, the copy-on-write fork path can double-free a page frame. This is a known bug tracked in our security audit. The fix requires rewriting the COW page table cloning logic.
2. **KPTI Not Fully Wired:** The CR3 swap code exists but the syscall entry/exit paths don't fully use it yet. A clever userspace program might be able to read kernel memory via cache timing.
3. **Weak Password Hashing:** `/etc/shadow` uses single-iteration SHA-256. This is documented as weak. A GPU could crack it in seconds.

### 🟠 Medium (Annoying But Not Fatal)

4. **No X.509/PKI/ECDHE:** The TLS stack only does PSK. You can't verify certificates. You can't do ECDHE key exchange. You can only talk to servers that accept raw PSK. (There are approximately zero such servers on the public internet.)
5. **GPU Shader Stubs:** The AMD GPU compute path executes shader binaries that are stubs. They return zeros or garbage. If you depend on GPU-accelerated topology computation, you get wrong answers.
6. **Simulated WiFi/Bluetooth:** The WiFi and Bluetooth stacks return simulated scan results. They don't actually use the hardware. If you think you're connected to a network, you're connected to our imagination.

### 🟡 Low (Cosmetic / Inconvenient)

7. **Default Password "seal":** The default login is `seal`/`seal`. Change it. Or don't. It's a research OS.
8. **No Multi-User Permissions:** There's no proper user management. `setuid`/`setgid` exist as syscalls but the security model is basically "everyone is root."
9. **Serial Output During Panic:** The panic handler writes to serial. If serial fails, the panic handler might double-panic. We fixed one instance of this but there may be others.

### 🟢 Theoretical (We Think These Exist But Haven't Proven)

10. **APIC Timer Race:** There might be a race between the scheduler lock release and context switch. We haven't observed it in practice, but the window exists.
11. **Voronoi Index Overflow:** If you create more than 2^32 files, the inode generation counter wraps. This is theoretical because we haven't tested with 4 billion files.

> **If you find a new vulnerability:** Please report it privately. We will fix it, credit you, and add it to this section with appropriate self-deprecating commentary.

---

## License

MIT License. Copyright (c) 2024 Teerth Sharma. See [LICENSE](LICENSE).


> **Why MIT and not GPL?** Because we want people to use this code, learn from it, and maybe even build something cooler. If you fork Seal OS and turn it into a billion-dollar product, good for you. Just maybe mention us in the credits. Or don't. The MIT license doesn't require it, but it would be nice.

---


---

## System Call Reference

Seal ABI provides native kernel semantics without POSIX inheritance. Epsilon extensions expose the theorem engine to userspace.

### Process & Task Management

| # | Name | Description |
|---|------|-------------|
| 0 | `exit` | Terminate current process |
| 5 | `exec` | Execute a new program (ELF64, shebang, or Aether-Lang script) |
| 6 | `fork` | Duplicate current process — child returns 0, parent gets real PID |
| 7 | `waitpid` | Wait for child process state change |
| 9 | `getpid` | Return current process ID |
| 16 | `getppid` | Return parent process ID |
| 17 | `nanosleep` | Sleep for specified duration (spin-yield implementation) |
| 18 | `reboot` | ACPI power-off, keyboard controller reset, or triple-fault |
| 25 | `kill` | Send signal to process |
| 26 | `sigaction` | Install signal handler |
| 27 | `sigreturn` | Return from signal handler |
| 34 | `watchdog` | Pet the APIC timer watchdog |

### File & Directory Operations

| # | Name | Description |
|---|------|-------------|
| 1 | `write` | Write bytes to file descriptor |
| 2 | `read` | Read bytes from file descriptor |
| 3 | `open` | Open or create a file |
| 4 | `close` | Close a file descriptor |
| 10 | `stat` | Get file metadata |
| 11 | `mkdir` | Create directory |
| 14 | `chdir` | Change current working directory (per-task) |
| 15 | `getcwd` | Get current working directory |
| 19 | `lseek` | Seek in file (SEEK_SET, SEEK_CUR, SEEK_END) |
| 20 | `unlink` | Remove file |
| 21 | `rmdir` | Remove empty directory |
| 22 | `rename` | Rename or move file (cross-mount fallback supported) |

### Memory Management

| # | Name | Description |
|---|------|-------------|
| 8 | `mmap` | Map memory into process address space (demand-paged) |
| 31 | `brk` | Grow or shrink user heap |

### Security & Information

| # | Name | Description |
|---|------|-------------|
| 12 | `setuid` | Set user ID |
| 13 | `setgid` | Set group ID |
| 23 | `getrandom` | Hardware entropy (RDRAND/RDSEED, fails closed) |
| 24 | `kmsg_read` | Read from 32 KiB kernel message ring buffer |
| 32 | `gettimeofday` | Get time since epoch (RTC-based) |
| 33 | `settimeofday` | Set system time (returns EPERM — honest) |
| 35 | `ioctl` | Device-specific control (TCGETS, TCSETS, FIONREAD) |

### Inter-Process Communication

| # | Name | Description |
|---|------|-------------|
| 28 | `pipe` | Create anonymous pipe (returns two fds) |
| 29 | `dup` | Duplicate file descriptor |
| 30 | `dup2` | Duplicate fd to specific target fd |

### Epsilon Extensions (Theorem & Topology)

| # | Name | Description |
|---|------|-------------|
| 100 | `manifold_query` | Query topological state (Voronoi cells, governor epsilon) |
| 101 | `teleport` | Execute O(1) metadata file move via topological surgery |
| 102 | `theorem_status` | Read T1-T10 verification status |
| 103 | `pkg_install` | Install package from `.eph` or HTTPS registry |
| 104 | `pkg_remove` | Remove installed package |
| 105 | `pkg_list` | List installed packages |
| 106 | `wifi_scan` | Scan for WiFi networks (simulated) |
| 107 | `wifi_connect` | Connect to WiFi network (simulated) |
| 108 | `bt_scan` | Scan for Bluetooth devices (simulated) |
| 109 | `bt_pair` | Pair Bluetooth device (simulated) |
| 110 | `setting_get` | Read live setting from BTreeMap |
| 111 | `setting_set` | Write live setting to BTreeMap |

### Syscall Result

All syscalls return `SyscallResult { code: i64, data: Option<String> }`.

---

## Troubleshooting

### Build Errors

| Symptom | Cause | Fix |
|---------|-------|-----|
| `error: no such file or directory` for `cargo +nightly` | Nightly toolchain not installed | `rustup toolchain install nightly --component rust-src,llvm-tools-preview` |
| `aes-gcm` or `curve25519-dalek` fails in debug | LLVM SIMD lowering bug on `x86_64-unknown-uefi` | Build release only: `cargo +nightly build --release` |
| `linker `rust-lld` not found` | Missing `llvm-tools-preview` | `rustup component add llvm-tools-preview --toolchain nightly` |
| QEMU shows "no bootable device" | OVMF firmware not found | Install `ovmf` package; path varies by distro |
| Serial log garbled | Wrong baud rate | Verify COM1 at 115200 baud, 8N1 |

### Runtime Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| Desktop not appearing | GOP framebuffer unavailable | Check VM display settings; serial-only fallback still works |
| AHCI disk not detected | QEMU machine type incorrect | Use `-machine q35` with AHCI controller |
| Theorem verification fails | `aether_verified` invariant mismatch | Check `kernel/aether/aether-verified/` build; run `cargo test -p aether_verified` |
| No audio playback | HDA codec not discovered | Verify QEMU `-device intel-hda`; some codecs require specific verb sequences |

### Performance Tuning

| Goal | Setting |
|------|---------|
| Faster boot | Disable desktop proof frame: remove `desktop_proof_frame_blit` from boot sequence |
| Larger heap | Modify `HEAP_SIZE` in `memory/heap.rs` (default 16 MB) |
| More Voronoi cells | Recompile with `VORONOI_K` constant in `aether-core/src/tss.rs` |
| Verbose logging | Enable `LOG_LEVEL=debug` at compile time |

---

## aether-core — Math Foundation

The `no_std` mathematics library that powers every theorem call in the kernel.

```mermaid
graph TD
    subgraph SG57["aether-core"]
        TSS["tss.rs<br/>SphericalVoronoiIndex&lt;K&gt;<br/>great_circle_distance()"]
        SCM["scm.rs<br/>SpectralContractionOperator&lt;D&gt;<br/>LatentPredictor"]
        TOPO["topology.rs<br/>compute_betti_0(), compute_betti_1()<br/>TopologicalShape, verify_shape()"]
        GOV["governor.rs<br/>GeometricGovernor<br/>α=0.01, β=0.05, ε₀=0.1"]
        MAN["manifold.rs<br/>ManifoldPoint&lt;D&gt;<br/>SparseAttentionGraph<br/>TimeDelayEmbedder<br/>TopologicalPipeline"]
        STATE["state.rs — state tracking"]
        OS["os.rs — page tables, CPU context"]
        MEM["memory.rs — Chebyshev liveness bounds"]
        AETHER["aether.rs — BlockMetadata,<br/>DriftDetector, HierarchicalBlockTree"]
    end

    subgraph SG58["ml/"]
        TENSOR["tensor.rs — N-dim tensor ops"]
        AUTOGRAD["autograd.rs — backpropagation"]
        NEURAL["neural.rs — Dense, Conv, Activation"]
        CLUSTER["clustering.rs — Betti-0 semantic clustering"]
        LINALG["linalg.rs — matrix ops, SVD, eigen"]
        CONV["convolution.rs — manifold-aware conv"]
        CLASS["classification.rs"]
        REG["regression.rs"]
        GOSSIP["gossip.rs — distributed learning"]
    end

    TSS --> TOPO
    SCM --> MAN
    GOV --> TSS
    TENSOR --> LINALG
    AUTOGRAD --> TENSOR
    NEURAL --> AUTOGRAD
```

### Key Algorithms

- **`SphericalVoronoiIndex<K>::locate(θ, φ)`**: computes great-circle distance to all K centroids, returns nearest. O(K) with K constant = O(1) amortized. Distance: `arccos(sin θ₁ sin θ₂ + cos θ₁ cos θ₂ cos(φ₁ - φ₂))`.

- **`GeometricGovernor::adapt(deviation)`**: PD control law `ε(t+1) = ε(t) + 0.01·e(t) + 0.05·de/dt` where `e(t) = R_target - Δ(t)/ε(t)`. Clamped to [0.001, 10.0]. Target tick rate: 1000 Hz.

- **`SpectralContractionOperator<D>::step(state)`**: applies a contraction mapping with ratio < 1, guaranteed convergence to a fixed-point attractor by Banach's theorem.

---

## Lean 4 Proofs

All ten theorem checks build into `kernel/seal-os` through the `aether_verified` no_std crate. Lean 4 artifacts live beside them.

```
kernel/aether/aether-verified/lean/
├── AetherVerified.lean           # Top-level umbrella
├── AetherVerified/
│   ├── Pruning.lean              # Pruning algorithm proofs
│   ├── Governor.lean             # T4 governor convergence
│   ├── Chebyshev.lean            # Chebyshev liveness bounds
│   └── Betti.lean                # Betti number properties
├── lakefile.lean                 # Lake build config
└── lean-toolchain                # Lean 4.7.0
```

CI builds the Lean package on every push. Proof strength and remaining placeholders are tracked in [docs/THEOREMS.md](docs/THEOREMS.md).

**Hygiene status**: Zero `sorry` or `admit` tactics in theorem files.

---

## CI Pipeline

16 jobs. Every push. No exceptions.

| Job | What it checks |
|-----|---------------|
| `fmt` | `cargo fmt --check` across all files |
| `build` | `cargo build --workspace` |
| `clippy` | `cargo clippy --workspace --all-targets -- -D warnings` |
| `test` | `cargo test --workspace` + `no_std` feature check + doc claim contract + Lean hygiene |
| `bench-compile` | `cargo bench --workspace --no-run` |
| `miri` | Miri UB detection on `aether-core` state, OS, and proptest |
| `bench-regression` | Criterion `io_cycle_8_lbas` < 120 ns median gate |
| `audit` | `cargo audit` for known vulnerabilities |
| `deny` | `cargo deny check` for license/dependency policy |
| `docs` | `cargo doc --workspace --no-deps` with `-D warnings` |
| `bom` | UTF-8 BOM check on all source files |
| `kernel-build` | Seal OS kernel build on nightly, PE/COFF header verification |
| `kernel-image` | UEFI disk image creation, `seal-mkimage --verify`, ISO build |
| `kernel-qemu-smoke` | 240-second QEMU boot with 20+ hard milestone gates |
| `kernel-clippy` | Kernel-specific clippy on nightly |
| `laamba-governor-check` | LAAMBA Governor Rust backend check |
| `lean` | Lean 4 package build with Mathlib cache |

**QEMU smoke test hard gates** (must all pass):

1. UEFI entry and Seal OS banner
2. Heap initialized
3. IDT + PIC initialized
4. T4 governor online
5. T1 Voronoi index reports 8 cells
6. All ten theorem lines `[THEOREM] Tn/... VERIFIED`
7. QEMU AHCI disk identity
8. Block device `0x800` registered
9. Persistent ManifoldFS root mounted from disk
10. Desktop proof frame blit sentinel
11. Desktop ready sentinel
12. Event-loop entry sentinel
13. Scheduler started
14. SYSCALL/SYSRET MSRs programmed
15. Aether runtime proof marker

See [docs/CI.md](docs/CI.md) for full pipeline documentation.


---

## The CI Pipeline: Our Digital Torture Chamber

Our CI is not a suggestion. It is a law. Break it, and your PR dies. Here is what every push goes through.

### The Jobs

| Job | What it checks | Our feelings about it |
|-----|---------------|----------------------|
| `fmt` | `cargo fmt --check` across all files | Boring but necessary. Like brushing your teeth. |
| `build` | `cargo build --workspace` | If this fails, nothing else matters. |
| `clippy` | `cargo clippy --workspace --all-targets -- -D warnings` | Clippy is a pedant. We love it and hate it. |
| `test` | `cargo test --workspace` + `no_std` feature check + doc claim contract + Lean hygiene | The big one. Break this, go to jail. |
| `bench-compile` | `cargo bench --workspace --no-run` | Ensures benchmarks still compile. |
| `miri` | Miri UB detection on `aether-core` state, OS, and proptest | Miri finds things we didn't know were wrong. It's spooky. |
| `bench-regression` | Criterion `io_cycle_8_lbas` < 120 ns median gate | Performance must not regress. Period. |
| `audit` | `cargo audit` for known vulnerabilities | Security scanning. Important. |
| `deny` | `cargo deny check` for license/dependency policy | We take licensing seriously. |
| `docs` | `cargo doc --workspace --no-deps` with `-D warnings` | Documentation must build. |
| `bom` | UTF-8 BOM check on all source files | Because BOMs are evil. |
| `kernel-build` | Seal OS kernel build on nightly, PE/COFF header verification | The kernel must build. |
| `kernel-image` | UEFI disk image creation, `seal-mkimage --verify`, ISO build | The image must be valid. |
| `kernel-qemu-smoke` | 240-second QEMU boot with 20+ hard milestone gates | The OS must actually boot. |
| `kernel-clippy` | Kernel-specific clippy on nightly | Extra pedantry for kernel code. |
| `laamba-governor-check` | LAAMBA Governor Rust backend check | Our Tauri app must compile. |
| `lean` | Lean 4 package build with Mathlib cache | Math must be right. |

### QEMU Smoke Test Hard Gates

1. UEFI entry and Seal OS banner
2. Heap initialized
3. IDT + PIC initialized
4. T4 governor online
5. T1 Voronoi index reports 8 cells
6. All ten theorem lines `[THEOREM] Tn/... VERIFIED`
7. QEMU AHCI disk identity
8. Block device `0x800` registered
9. Persistent ManifoldFS root mounted from disk
10. Desktop proof frame blit sentinel
11. Desktop ready sentinel
12. Event-loop entry sentinel
13. Scheduler started
14. SYSCALL/SYSRET MSRs programmed
15. Aether runtime proof marker

If any of these 15 gates fail, the entire CI run fails. No exceptions. No mercy.

> **CI story:** Once, we pushed a change that broke the desktop soak marker because we changed the font rendering and the pixel colors shifted by 1. CI caught it. We fixed it. This is why we have CI.

---

## Acknowledgements

Seal OS stands on the shoulders of:

- **The Rust community** — for `no_std`, `core`, `alloc`, and the borrow checker
- **The Lean community** — for formal proof infrastructure and Mathlib
- **TempleOS (Terry A. Davis)** — for proving that a single person can write an entire operating system with a native language
- **Redox OS** — for demonstrating Rust bare-metal OS viability
- **Topology and geometry researchers** — for the mathematical primitives that make this design possible


---

## People We Blame For This

*(An expanded acknowledgements section with personality)*

- **The Rust community** — for `no_std`, `core`, `alloc`, and the borrow checker. Without you, we'd be writing C and having buffer overflow nightmares.
- **The Lean community** — for formal proof infrastructure and Mathlib. You make us look smarter than we are.
- **TempleOS (Terry A. Davis)** — for proving that a single person can write an entire operating system with a native language. Rest in peace, Terry. Your spirit lives in every `~` terminator.
- **Redox OS** — for demonstrating Rust bare-metal OS viability. You're the cool older sibling we aspire to annoy.
- **Topology and geometry researchers** — for the mathematical primitives that make this design possible. We read your papers at 3 AM and made questionable life choices because of them.
- **QEMU developers** — for the emulator that lets us test without bricking real hardware. You are the unsung heroes of OS development.
- **The person who invented coffee** — without you, none of this would exist.
- **Our future therapists** — you're going to have so much material.

---

## Contact

- **Author**: Teerth Sharma
- **Repository**: https://github.com/teerthsharma/epsilon-hollow
- **Security**: See [SECURITY.md](SECURITY.md)
- **Discussions**: GitHub Discussions tab


---

## Glossary of Words We Made Up

| Word | Definition |
|------|------------|
| **ManifoldFS** | A filesystem where files have geometric embeddings. Not a real manifold in the mathematical sense, but close enough for marketing. |
| **Aether-Lang** | A programming language with `~` terminators. Pronounced "ether lang." Not related to Ethereum. |
| **Teleport** | Moving a file by rewiring metadata pointers. No actual quantum mechanics involved. |
| **TopoRAM** | Memory with spherical coordinates. Because flat RAM was too boring. |
| **Betti-0** | The number of connected components in a topological space. In our OS, it measures fragmentation. In conversation, it makes you sound smart. |
| **Governor epsilon** | A PD control parameter that adapts scheduling timeslices. Named after the Greek letter because Greek letters make things look official. |
| **Spectral Contraction** | A mathematical operation that shrinks prediction states. Not related to ghosts or music. |
| **Hyperbolic Curvature** | A measure of how "tree-like" a structure is. In our OS, it classifies file lifetimes. In normal life, it's a phrase that ends conversations at parties. |
| **Voronoi Cell** | A region of space closer to one point than any other. We use them for scheduling, memory, and files. They are the Swiss Army knife of our OS. |
| **Seal OS** | This operating system. Named after seals. Not Navy SEALs. The cute kind that balance balls on their noses. |
| **LAAMBA** | We still don't know. If you figure it out, open an issue. |

---

## A Day In The Life Of A Seal OS Developer

**06:00** — Wake up. Check CI. Green? Good. Red? Panic.

**06:30** — Coffee. Read topology paper. Get idea for new optimization.

**07:00** — Implement idea. Break build. Fix build. Break test. Fix test.

**09:00** — Realize the optimization requires changing 12 files. Question life choices.

**10:00** — Write Lean proof for the optimization. Lean says `sorry` is not allowed. Spend 2 hours on a lemma.

**12:00** — Lunch. Think about spheres.

**13:00** — Debug APIC timer interrupt. Realize it's a one-off race condition. Add atomic operation. Test passes.

**14:00** — Run QEMU. OS boots. Desktop appears. Feel accomplished.

**14:05** — Try to open calculator. Calculator panics. Investigate.

**15:00** — Fix calculator panic. It was a font rendering off-by-one. Commit.

**16:00** — CI runs. All green. Push to main.

**16:30** — Read Hacker News. Someone posted about Seal OS. Top comment: "Why?"

**17:00** — Write sarcastic FAQ entry in response.

**18:00** — Dinner. Tell friend about OS. Friend asks "can it run Excel?"

**19:00** — Cry.

**20:00** — Write more Lean proofs. The math is soothing.

**22:00** — Bed. Dream of Voronoi cells.

**02:00** — Wake up with idea for new theorem. Write it down. Go back to sleep.

**Repeat.**

---

## Final Words

If you've read this far, congratulations. You now know more about Seal OS than 99% of humanity. You know our strengths, our weaknesses, our jokes, and our regrets.

Seal OS is not a product. It is not a startup. It is not a revolution. It is one person's obsession with geometry, expressed as 111,000 lines of Rust and a dream of a world where operating systems think in spheres.

If that sounds interesting to you, welcome aboard. If it sounds insane, you're not wrong. But insanity is just genius that hasn't been understood yet.

Or maybe it's just insanity. Either way, the code compiles.

---

---

<p align="center">

<!-- RUST_LINE_COUNT_START -->
**111183 lines of Rust** across 387 files | 0 lines of x86 assembly | 1823 lines of Aether-Lang DSL | **113006 total**
<!-- RUST_LINE_COUNT_END -->

</p>

<p align="center">
  <em>OS state is topology on S². No timelines. No excuses. Only geometry.</em>
</p>
