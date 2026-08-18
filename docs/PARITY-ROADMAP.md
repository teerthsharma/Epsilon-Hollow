# Parity Roadmap — Seal OS toward a self-contained desktop

**Status date:** 2026-08-18. Written against commit `684a89d`, branch `ralph/graph-round-1`,
after a full QEMU boot on Windows 11 with QEMU 11.1.0 and OVMF `edk2-x86_64-code.fd`.

Two targets, and they pull in different directions:

- **Ubuntu v1 parity** — the capability set of Ubuntu 4.10 (Warty, October 2004): boot to a
  desktop, a terminal that accepts input, a file manager, a text editor, install packages,
  reach the network, install to disk and survive a reboot, log in as a user.
- **HolyOS character** — self-contained, one coherent system, its own language and shell,
  no POSIX debt. `FUTURE_PLAN.md` already states this: POSIX, libc and GRUB compatibility
  are rejected except as labelled legacy host interop.

The tension is real and this document resolves it one way: **deliver Ubuntu-level capability
through the Seal ABI, never by adopting POSIX.** Every gate below is written as a capability
the user gets, not as a Linux subsystem to clone.

---

## What the boot actually showed

A complete boot happened. The serial log is not ambiguous:

```
[BOOT] GOP mode set to 1024x768 (area=786432)
[BOOT] Framebuffer: 1024x768x32bpp @ 0x80000000, pitch=4096
[KASLR] alias_base=0xffffffffb1c00000 entropy=rdseed bits=30
[SMP] BSP per-CPU data initialized
[SealShell] Loaded
[Desktop] 12 windows active (Terminal, IDE, Files, Theorems, Calculator, SealPlayer,
          Snake, Breakout, Warp Racer, Tensor Viewer, LAAMBA Governor, Aether App)
[EVENT] Entering real event loop — keyboard and mouse active
```

Memory, KASLR, ACPI, SMP init, VFS, ext2, the shell, the compositor and twelve windows all
came up. That is a lot of working operating system. The three complaints — slow, bad
graphics, terminal takes no input — have three separate causes, and they are not the same
kind of problem.

### Why it was slow

**Cause 1 — it ran under software emulation.** The launch passed no `-accel` flag, so QEMU
used TCG, interpreting every guest instruction. `qemu-system-x86_64 -accel help` on this
machine reports `tcg` and `whpx`. This is the dominant term and it is a launch-flag defect,
not a kernel defect.

**Cause 2 — one core.** No `-smp` was passed, and the log confirms `[SMP] Only BSP present
(0 CPUs total)`. The kernel has real AP bootstrap, real per-CPU runqueues and a real
reschedule IPI, all verified in source. None of it was exercised.

**Cause 3 — the compositor redraws everything, every frame.** This one is real and lives in
the kernel:

```
[GFX] desktop-soak frames=24 p50_cycles=41166182 p95_cycles=45680154 dirty_px_max=786432
```

`dirty_px_max=786432` is exactly `1024 × 768` — the entire framebuffer is marked dirty on
every frame. There is no damage-rectangle tracking, so every frame is a full back-buffer
composite plus a full blit across an emulated MMIO aperture. Under TCG, where each
framebuffer write traps, this is close to the worst possible combination. Fixing cause 3
matters even after causes 1 and 2 are gone.

### Why the terminal took no input

Input is reaching the desktop. The boot proof shows a click routed all the way through:

```
[GFX] desktop-live-proof route=desktop_handle_input action=desktop_icon_launch
      app=Files app_id=3 events=2 handled=1 icon_hit=1 launched_app_id=3 result=pass
```

So the event loop, hit-testing and window focus all work. The gap is narrower than it
appeared: the Terminal window has no keystroke handler that feeds the shell. That is a
bounded defect, not a missing subsystem.

### The thing that reframes everything

The boot log line `[execve] '/bin/init' not found; continuing with kernel desktop` reads
like "there is no userspace." It is not. `Process::execve`
(`kernel/seal-os/src/process/mod.rs:30`) genuinely reads the file, calls
`scheduler::spawn_user`, ELF-loads it, and the task's first scheduled instruction is
`enter_userspace_trampoline`, which reaches ring 3 through `iretq`
(`kernel/seal-os/src/process/userspace.rs:46`). There is even an embedded
`EMERGENCY_SHELL_ELF` fallback.

**The machinery is built. No `/bin/init` binary is produced or placed on the disk image**, so
`lib.rs:591`'s VFS lookup fails and `execve` is never called. Everything you saw — shell,
windows, apps — is kernel-side code.

That is the single highest-leverage fact in this document. The distance to a real userspace
is one binary and its plumbing, not a rewrite.

---

## Gate structure

Six gates. Each has an **exit proof**, because this project already has the right mechanism
for that: `seal-mkimage --check-*-proof` verifies serial markers from a QEMU boot, and
`.github/workflows/kernel-tests.yml` runs them. Every gate below ends in a new marker. A gate
is not done because it looks done; it is done when a boot proof asserts it.

---

## Gate 0 — Make the development loop honest

Nothing else is measurable until this is true. Days, not weeks.

### G0.1 — SMP initialises before ACPI, so the kernel never sees a second CPU

This is the highest-leverage single fix in this document and it is an ordering bug.

`cpu::smp::smp_init()` runs at `kernel/seal-os/src/lib.rs:240`. Its first act is to ask ACPI
for the topology:

```rust
let cpu_count = crate::drivers::acpi::cpu_count();
let apic_ids  = crate::drivers::acpi::apic_ids();
if cpu_count <= 1 {
    serial_println!("[SMP] Only BSP present ({} CPUs total)", cpu_count);
    return;
}
```

But `drivers::acpi::init(info)` — which parses the MADT and populates those tables — does not
run until `lib.rs:269`, twenty-nine lines later. `cpu_count()` is therefore read before it is
ever written, `smp_init` takes the early return, and no AP is ever started.

Three independent confirmations:

1. **Source order.** `lib.rs:240` (smp_init) precedes `lib.rs:269` (acpi::init).
2. **Serial log order.** `[SMP] Only BSP present (0 CPUs total)` prints *before*
   `[ACPI] MADT found @ 0x7FB78000`. The consumer runs before the producer.
3. **It reproduces under load.** Booting with `-smp 4` still yields
   `[SMP] Only BSP present (0 CPUs total)`.

The consequence is large. Every SMP feature in this tree — AP trampoline through long mode,
per-CPU `ManifoldScheduler` runqueues, idle-triggered work stealing, the reschedule IPI, the
TLB shootdown IPI — is implemented, unit-covered, and **has never executed once**. Phase 4 in
`FUTURE_PLAN.md` is honestly ticked as built, and every one of those ticks describes code that
is dead at runtime.

**The ordering bug was masking a second one.** `smp_init` is now split: `smp_init_bsp()` keeps
the per-CPU/GS-base setup at `lib.rs:240`, where the idle-stack switch immediately after it
needs it, and `smp_start_aps()` holds the AP bring-up and is placed after the scheduler's first
yield, which is the earliest point where ACPI, the local APIC, the scheduler and a live
`interrupts::ticks()` all hold.

Arming it deadlocks the boot. Measured with `-smp 4` on QEMU 11.1.0 with OVMF: the boot reaches
`[BOOT] Scheduler first yield returned` and stops. The tick spin is not the cause — the local
APIC timer is confirmed live by two `[THERMAL]` lines in the same log. The mechanism is that
`timer_handler_apic` calls `thermal_governor_step`, which `serial_println!`s under a global
lock; once an AP is ticking, both CPUs take that lock from interrupt context.

So the call ships commented out with a `ponytail:` marker naming the ceiling and this upgrade
path: make interrupt-context logging lock-free or drop it on contention with `try_lock`, audit
every `serial_println!` reachable from an interrupt handler for the same pattern, then arm the
call behind a boot proof asserting `[SMP] N CPUs online` with N > 1.

Two bugs, then, not one — and the second is the reason to expect more. No line of AP code in
this tree has ever run.

### G0.2 — Everything else in the loop

| Item | Detail |
|---|---|
| Hardware acceleration | Add `-accel whpx` (Windows) / `-accel kvm` (Linux) with TCG fallback to the QEMU runner scripts. **Caveat measured 2026-08-18:** `-accel whpx -cpu max` makes OVMF itself `#GP` in `PlatformPei` on every AP before Seal OS loads; dropping `-cpu max` clears that, but the boot then stalls inside AHCI port probing (`[disk::ahci] Port N SSTS=0x0 ... SIG=0xffffffff` on empty ports) and does not reach the desktop within 150 s, where the same image under TCG reaches it comfortably. AHCI has no interrupt-driven I/O and its retry path has no backoff, so probe timeouts that were free under emulated timing become real wall-clock under acceleration. Acceleration is not currently usable and needs this investigated first. |
| Multi-core | Pass `-smp 4` — but it does nothing until G0.1 lands. |
| CI reaches the kernel | `kernel/seal-os` is in `exclude` in the root `Cargo.toml`, so `cargo test --workspace` and `cargo clippy --workspace` never touch it. Its 552 in-kernel `register_test` cases only run under QEMU. Wire the documented invocation (`cargo +nightly clippy --release --target x86_64-unknown-uefi`, from inside the crate, without `--all-targets`) into CI. |
| Frame timing that means something | `missed_16ms=unscaled` in the soak proof means frame budget is not being measured. Make it a real number under acceleration. |

**Exit proof:** a boot proof recording accelerator in use, CPU count > 1, and a frame-time
p95 in milliseconds rather than raw cycles.

---

## Gate 1 — It responds

The "barely working" gate. This is what turns a demo into something you can sit in front of.

| Item | Detail |
|---|---|
| Terminal keystroke path | Route key events from the event loop through the focused window into `SealShell`. Everything upstream already works. |
| Damage tracking | Track dirty rectangles per window; composite and blit only what changed. `dirty_px_max` must stop being 786432 on an idle desktop. |
| Shell exit status | `Shell::dispatch` returns `String`, so a failing command is indistinguishable from output — a sourced script that fails carries on (`kernel/seal-os/src/apps/shell.rs:1929`). Introduce an exit status. |
| Shell tokenizer | No quoting exists: `|`, `<`, `>` are metacharacters everywhere, so `write note a > b` redirects instead of writing `a > b` (`kernel/seal-os/src/apps/shell.rs:986`). |

**Exit proof:** a boot proof that types a command into the Terminal window, asserts the shell
saw it, asserts a nonzero exit status propagates, and asserts idle-frame dirty pixels are
under a threshold.

---

## Gate 2 — Real userspace

The spine. Everything after this depends on it.

| Item | Detail |
|---|---|
| Ship `/bin/init` | Build a real ELF init and place it on the image so `lib.rs:591` finds it. The loader and ring-3 entry already work. |
| `execve` argv/envp | Currently `_argv` and `_envp` are ignored, and the task name is a hardcoded match on `"/bin/init"` (`process/mod.rs:46`). |
| Real PIDs | `process::create` hardcodes `pid: 1` (`process/mod.rs:22`). |
| `fork` inheritance | A forked child inherits no descriptors and no registered regions. The upgrade path is already written down: `syscall::table::inherit_fds` plus `memory::mmap::inherit_regions` (`process/scheduler.rs:938`). |
| `SYS_MUNMAP` | Does not exist. User address space only grows; `munmap_user` has no callers and the VA allocator has no free list (`memory/mmap.rs:59`). |
| Process identity | `stat` reports every file as root-owned because no task holds identity (`process/scheduler.rs:1514`), which makes the security distance check treat every file as sensitive. |

**Exit proof:** a boot proof showing `/bin/init` spawned as a ring-3 task with a real pid,
forking a child that inherits an open descriptor, mapping and then unmapping a region, and
exiting with a status the parent reads.

---

## Gate 3 — Userspace can reach the machine

Right now a userspace process, once it exists, can barely do anything.

| Item | Detail |
|---|---|
| Socket syscalls | **Zero exist.** TCP and UDP have a complete in-kernel Rust API used by `http.rs`, `dhcp.rs` and `tls_socket.rs`, but `syscall/table.rs` has no socket, bind, listen, accept, connect, send or recv entry. Userspace cannot touch the network at all. This is the largest single capability gap in the tree. |
| ext2 `truncate` | No truncate operation exists; only whole-file unlink frees blocks (`fs/ext2.rs:1510`). A text editor needs this. |
| Terminal as a process | Once init exists, move the terminal off the kernel side onto a real process with a tty. |
| Buffer cache flush | Mark-dirty is real, but nothing flushes periodically — only explicit callers (`fs/buffer_cache.rs:89`). Unflushed writes are lost data. |
| Block layer unification | virtio-blk defines its own `BlockDevice` trait and is never registered with `register_block_device`, so it is not part of the block layer (`drivers/block/virtio_blk.rs:7`). |

**Exit proof:** a boot proof where a ring-3 process opens a socket through a syscall, completes
a loopback TCP exchange, writes a file, truncates it, and the data survives a buffer-cache
flush and re-read.

---

## Gate 4 — Ubuntu v1 parity

Warty's actual user story, delivered through the Seal ABI. Much of this exists as kernel-side
code and needs to become userspace-reachable rather than being written from scratch.

| Capability | Standing today |
|---|---|
| Boot to desktop | Works. Twelve windows compose and render. |
| Terminal you can type in | Gate 1. |
| File manager | A Files app exists and launches; needs the userspace VFS path. |
| Text editor | Needs Gate 3's truncate and write path. |
| Install packages | ManifoldPkg exists with a verified remote channel and a rollback floor. Note the floor key ships in the boot image, so it stops an attacker who can write the data partition but not one who can also read the ESP (`pkg/channel.rs:63`) — closing that needs a TPM NV counter or an authenticated UEFI variable, which is hardware this kernel does not have. |
| Reach the network | Gate 3's socket syscalls. Also needs a routing table — today there is a single subnet/gateway check with no longest-prefix match — and DHCP renew/rebind, which do not exist (`net/dhcp.rs:171`). |
| Install to disk, survive reboot | The installer does real GPT partitioning and ext2 format against a guarded target. Needs `/bin/init` on the installed volume, and ext2 formatting past its current single-block-group 8 MiB ceiling (`fs/ext2_format.rs:72`). |
| Log in as a user | Needs Gate 2's process identity, plus ManifoldFS per-node ownership that survives a remount — ownership currently lives in a side table, not on the inode (`fs/manifold_fs.rs:95`). |

**Exit proof:** an install-to-disk boot proof — install from the live image to a second disk,
reboot into it, log in as a non-root user, open a terminal, edit and save a file, install a
package, and fetch a URL, all from ring 3.

---

## Gate 5 — HolyOS character

This is what makes it yours rather than a small Linux. It should be pursued alongside Gate 4,
not after it.

- **Aether-Lang as the system language.** The runtime already proves out at boot
  (`parser=ok interpreter=ok app_host=ok`). The goal is that every system surface is scriptable
  in it, and that the shell and it are the same language rather than two.
- **The topology as the interface, not a subsystem.** T1–T5 already drive real decisions —
  Voronoi cells for allocation, spectral contraction for prefetch, the governor for adaptation.
  Parity work must not bolt a conventional kernel beside them.
- **Adopt the primitives already built.** `TicketLock` and `SeqLock` are implemented and unit
  tested and have **zero callers** outside `sync/`; roughly 80 sites still use `spin::Mutex`,
  and the tick counter is a plain `AtomicU64`. Building a primitive and not adopting it is the
  most common way this tree accrues debt.
- **Keep the refusals.** A topological password construction was measured and deliberately not
  shipped. That judgment is an asset; parity pressure must not erode it.

---

## Sequencing

Gates 0 and 1 are days and unblock every judgment about performance. Gate 2 is the structural
one and everything downstream waits on it. Gate 3 is the widest gap. Gates 4 and 5 run in
parallel once 3 lands.

Do not start Gate 4 before Gate 2's exit proof passes. A parity checklist pursued while
everything still runs in kernel space produces a demo that cannot be finished.

---

## Standing debt

Deliberate shortcuts are inventoried in [`PONYTAIL-DEBT.md`](../PONYTAIL-DEBT.md): 43 markers,
each naming its ceiling and the trigger to revisit it. Most are correctly sized and should not
be touched. The ones that intersect this roadmap are cited inline above.

Per-item build status for Phases 1–4 lives in [`FUTURE_PLAN.md`](../FUTURE_PLAN.md), where
ticked items carry a `path:line` citation and unticked items are either unbuilt or annotated
with the specific gap that keeps them unticked.
