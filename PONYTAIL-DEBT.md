# Ponytail debt ledger

This file is generated from `ponytail:` comments in source. Where this table and a source comment disagree, the source comment is authoritative.

## boot

| Location | Simplified | Ceiling | Upgrade trigger |
|---|---|---|---|
| [boot/uefi_entry.rs:264](kernel/seal-os/src/boot/uefi_entry.rs:264) | GOP framebuffer pitch/height overflow is reported only, not enforced | "no OVMF build was available to confirm the value it reports, and refusing the mode on an unverified number would risk the boot path" | "Once one boot log shows this line absent and `needed <= fb_size` holds on real firmware, make this a `return None` alongside the other rejections above." |
| [drivers/acpi/rsdp.rs:126](kernel/seal-os/src/drivers/acpi/rsdp.rs:126) | ACPI table length ceiling is a heuristic byte count, not derived from the firmware memory map | fixed at `16 * 1024 * 1024` (16 MiB) | "Tighten by threading BootInfo's ACPI-reclaim region length through drivers::acpi::mod::init if a real machine ever needs more." |

## fs

| Location | Simplified | Ceiling | Upgrade trigger |
|---|---|---|---|
| [fs/ext2.rs:1364](kernel/seal-os/src/fs/ext2.rs:1364) | `i_size`/`i_blocks` (`u32`) directory-size wraparound left unguarded | reachable only past a 4 GiB volume, itself gated behind a heap allocation this `no_std` kernel has no `#[alloc_error_handler]` for; "no harness in this tree can construct the case" | "Upgrade path when a 4 GiB image fixture exists: `i_size.checked_add(self.block_size).ok_or(VfsError::IoError)?`, same for `i_blocks`, and a test that drives `add_dir_entry` on a directory at `u32::MAX - block_size`." |
| [fs/ext2_format.rs:72](kernel/seal-os/src/fs/ext2_format.rs:72) | `format_ext2` supports a single block group | "`BLOCKS_PER_GROUP` blocks (8 MiB at 1 KiB blocks)" | "Multi-group needs a BGD entry and a bitmap pair per group plus superblock backups; add it when a target larger than 8 MiB has to be formatted." |
| [fs/fat.rs:879](kernel/seal-os/src/fs/fat.rs:879) | FAT directory cluster lookup uses a linear scan | none stated beyond the scan itself | "a bitmap over clusters if directory counts ever make this show up in a profile." |
| [fs/manifold_fs.rs:95](kernel/seal-os/src/fs/manifold_fs.rs:95) | Inode ownership (`owners`) kept in a side table rather than a field on `InodeMetadata` | "ownership does not survive a remount" | "The upgrade path is `uid`/`gid` on `InodeMetadata` written into the eight bytes `InodeRecord::to_record` leaves zero at offset 248..256, which existing disks already read back as root and so needs no superblock version bump." |
| [fs/prefetch.rs:100](kernel/seal-os/src/fs/prefetch.rs:100) | `record_lba` shifts the 64-entry LBA history 512 bytes once full | "fine while nothing calls this more than once per block request" | "A caller on a hot path wants the ring back plus a copy-out that rotates it into arrival order." |
| [fs/vfs.rs:594](kernel/seal-os/src/fs/vfs.rs:594) | Test VFS is initialised on first use rather than explicitly at the top of `test_main` | "`testing/runner.rs` is owned by another change this round" | "Upgrade path: call `fs::init_vfs()` (or this function) from `test_main` before `register_all`, then delete this." |

## net

| Location | Simplified | Ceiling | Upgrade trigger |
|---|---|---|---|
| [net/dns.rs:18](kernel/seal-os/src/net/dns.rs:18) | Each DNS query allocates a fresh `udp::socket()` rather than reusing one shared socket | "UDP_SOCKETS grows without bound from DNS traffic again" | "A shared socket was tried and reverted: `rebind` and `sendto` are separate UDP_SOCKETS lock acquisitions, and a second task's concurrent query could rebind the shared socket in the gap between them... upgrade to a per-query-checked-out-and-returned socket (or a real close/free-list on UDP_SOCKETS) if the growth itself becomes the problem." |
| [net/ipv6.rs:404](kernel/seal-os/src/net/ipv6.rs:404) | Neighbor advertisement parsing only reads the option at offset 24 | an advertisement putting another option first is dropped rather than parsed ("fails closed") | "Walk the option chain by its length fields if a peer is seen to send one." |
| [net/tcp.rs:1129](kernel/seal-os/src/net/tcp.rs:1129) | Socket generation counter | "it wraps after 2^32 sockets on a 64-bit target -- 50 days at one socket per millisecond -- and a handle that old would alias again" | "Past that needs a handle wider than `usize`." |
| [net/tcp.rs:1170](kernel/seal-os/src/net/tcp.rs:1170) | `insert_socket` finds a free slot by linear scan | table "no longer grows without bound now that it is reaped" | "A free list when either stops holding single digits." |
| [net/tcp.rs:1265](kernel/seal-os/src/net/tcp.rs:1265) | `alloc_ephemeral_port` in-use test is a linear scan, held across `TCP_SOCKETS` | "O(sockets) per candidate on a table that holds single digits in every observed exchange" | "A port bitmap if a host ever holds thousands of sockets." |
| [net/tcp.rs:457](kernel/seal-os/src/net/tcp.rs:457) | `retransmit_queue.remove(pos)` | "O(n) on a queue that holds single digits, the same ceiling the flush loop already carries" | none stated |
| [net/tcp.rs:561](kernel/seal-os/src/net/tcp.rs:561) | `TCP_TX_QUEUE.remove(0)` | "O(n) on a queue that holds single digits in every observed exchange" | "A ring buffer if that ever stops being true." |

## ml_engine

| Location | Simplified | Ceiling | Upgrade trigger |
|---|---|---|---|
| [ml_engine/foliation.rs:71](kernel/seal-os/src/ml_engine/foliation.rs:71) | Fixed fan-out with a linear child scan | "32 distinct continuations per prefix; past that `descend` refuses to share and reports `children_full`" | "Upgrade path is an open-addressed key->child map per leaf, which trades 3x metadata for unbounded fan-out." |
| [ml_engine/foliation.rs:615](kernel/seal-os/src/ml_engine/foliation.rs:615) | Free-slot lookup in the leaf arena is a linear scan | "bounded by the construction-time arena size" | "Upgrade path is an explicit free list, which is 8 bytes per leaf and O(1) — not worth it until the arena outgrows a few hundred entries." |
| [ml_engine/foliation.rs:727](kernel/seal-os/src/ml_engine/foliation.rs:727) | `pick_victim` eviction is an O(pool_blocks) linear scan of resident plaques | "pool_blocks is fixed at construction, so this does not grow with sequences, tokens, or RAM" | "Upgrade path is a bucketed priority queue keyed on (entrants, depth) — both are small integers — giving O(1) pop at the cost of maintaining bucket membership on every refcount change." |
| [ml_engine/foliation.rs:779](kernel/seal-os/src/ml_engine/foliation.rs:779) | `next_use` (Belady oracle) is an O(remaining trace) forward scan | "used only inside the boot benchmark to bound how much of the LRU gap a realizable policy could close; it is never on a runtime path" | none stated |
| [ml_engine/stratum.rs:473](kernel/seal-os/src/ml_engine/stratum.rs:473) | `PREFETCH_OVERRIDE` is one global rather than one per stream | "with two workloads registered the later `SYS_FIT_REGIME` caller overwrites the earlier one's threshold, and [`unregister`] clears the value for both" | "Move it into [`FitStream`] and key [`training_prefetch_epsilon`] by task id when a second training workload can actually exist." |

## security

| Location | Simplified | Ceiling | Upgrade trigger |
|---|---|---|---|
| [sandbox.rs:452](kernel/seal-os/src/sandbox.rs:452) | `mst_edges` (2D) is a near-copy of the Prim loop in `ml_engine/stratum.rs` | that one "is private, hard-typed to `ManifoldPoint<3>` and `STRATUM_WINDOW`, and returns only `(max, median)` rather than the spectrum this needs" | "Upgrade path is to lift one generic `mst_edges` into a shared geometry module and have `stratum` derive its two statistics from it; that is a two-file change and this module owns neither file today." |
| [security/perm_field.rs:136](kernel/seal-os/src/security/perm_field.rs:136) | Permission-denial check is a quadratic scan re-scanning the observation set for a blocking grant | "128² · 8 ≈ 131k component comparisons, once, off any hot path" | "Upgrade path is sorting observations by path and walking ancestors with a stack, which is O(n log n) and about three times the code." |
| [security/topo_key.rs:446](kernel/seal-os/src/security/topo_key.rs:446) | The stuck-source arm of `fresh_salt` is not covered by any registered in-kernel test | "nothing in the kernel can decide what RDRAND returns; the host harness covers it by stubbing `getrandom`" | "To cover it in-kernel, `getrandom` would need a seam that feeds it bytes — which belongs in `drivers/entropy.rs`, not here." |

## pkg

| Location | Simplified | Ceiling | Upgrade trigger |
|---|---|---|---|
| [pkg/channel.rs:63](kernel/seal-os/src/pkg/channel.rs:63) | `FLOOR_RECORD_KEY` ships embedded in the boot image | "stops an attacker who can write the data partition (`/packages` on ext2) without reading the image on the ESP — it does not stop one who has both" | "Closing that, and closing the delete-and-replay hole `load_floor` documents, needs monotonic storage the kernel does not have yet: a TPM NV counter or a UEFI authenticated variable." |
| [pkg/mod.rs:49](kernel/seal-os/src/pkg/mod.rs:49) | `INSTALL_ROOTS` is a flat two-entry allowlist | "because two roots exist" | "If packages ever need to own more of the tree, this becomes a per-package prefix granted at install time rather than a constant." |

## process

| Location | Simplified | Ceiling | Upgrade trigger |
|---|---|---|---|
| [process/scheduler.rs:938](kernel/seal-os/src/process/scheduler.rs:938) | `fork_current` gives the child no inherited descriptors and no registered regions | none stated beyond the omission itself | "Upgrade path: `syscall::table::inherit_fds(parent_id, child_id)` and `memory::mmap::inherit_regions(parent_pt, child_pt)`, both invoked from the free `fork_current()` below *after* its scheduler guard is dropped, and neither consulting the scheduler itself. `fd_lookup` already takes `FILE_TABLE` and then `scheduler_lock`... so reaching either from in here, under the guard, inverts that order on a non-reentrant `spin::Mutex`." |
| [process/scheduler.rs:1514](kernel/seal-os/src/process/scheduler.rs:1514) | `current_uid` reports identity as root because there is no task to hold it | "there is no task to hold it" | "Upgrade path, in order: give `ManifoldFS::stat` real per-node ownership so `t5_check_distance` stops treating every file as sensitive, then make `schedule()` reachable so the boot thread has a task at all, then let this read the login identity." |

## apps/shell

| Location | Simplified | Ceiling | Upgrade trigger |
|---|---|---|---|
| [apps/shell.rs:270](kernel/seal-os/src/apps/shell.rs:270) | `grep_matches` case-fold lowercases the pattern and every tested line, one allocation per line | none stated beyond the allocation cost | "Upgrade path if a large stream ever makes that show up: compare `char` iterators with `to_lowercase` applied per character." |
| [apps/shell.rs:533](kernel/seal-os/src/apps/shell.rs:533) | `tr` supports no ranges (`a-z`) and no classes (`[:upper:]`); both arguments must be exactly one `char` | "`tr a-z A-Z` asks for something this does not do and must say so instead of translating `a` to `A` and dropping the rest" | "Upgrade path: expand a `x-y` argument into a character set and map by position." |
| [apps/shell.rs:730](kernel/seal-os/src/apps/shell.rs:730) | `cmd_export` marks a var exported but no command actually receives it | "Seal OS commands are `Shell` methods rather than processes... so there is no environment block to put anything in; inventing one would mean a process boundary that does not exist" | "Upgrade path: when `run` or `install` spawns a real task, hand the exported pairs to whatever builds its address space." |
| [apps/shell.rs:986](kernel/seal-os/src/apps/shell.rs:986) | `run_line` has no quoting; `\|`, `<`, `>` are metacharacters everywhere | "`write note a > b` redirects instead of writing `a > b`" | "Upgrade path: tokenize once into `Vec<String>` with `'`/`\"` handling, split the pipeline on unquoted `\|`, and hand each arm its argument vector." |
| [apps/shell.rs:1443](kernel/seal-os/src/apps/shell.rs:1443) | `search` is not filtered by the access policy | "a name and a byte count from a directory the caller may not read can still leak" | "Upgrade path: carry the resolved path on `FindResult` and filter here — that field lives in fs/manifold_fs.rs, outside this unit's scope." |
| [apps/shell.rs:1929](kernel/seal-os/src/apps/shell.rs:1929) | `cmd_source` reports a failing sourced script as ordinary output, so a script that sources a failing script carries on | "`dispatch` returns a `String`" | "Upgrade path is the same one the `Shell::run_line` note names: an exit status for the command table, at which point this returns it." |

## graphics/wm

| Location | Simplified | Ceiling | Upgrade trigger |
|---|---|---|---|
| [graphics/topo_render.rs:772](kernel/seal-os/src/graphics/topo_render.rs:772) | Zero-length edge lengths (`len0`/`1`/`2`) are divided into without a guard | "a degenerate triangle renders fully transparent at quality 4 instead of being skipped" (f32 division yields inf/NaN, saturating cast turns it into alpha 0, never a `#DE`) | "Upgrade by rejecting it alongside the `area <= 0.0` test above." |
| [wm/compositor.rs:644](kernel/seal-os/src/wm/compositor.rs:644) | Window `z_order` saturates at `u32::MAX` | "reaching that takes 4.29e9 such clicks" | "renumber the stack by sort rank if it ever matters." |

## memory

| Location | Simplified | Ceiling | Upgrade trigger |
|---|---|---|---|
| [memory/mmap.rs:59](kernel/seal-os/src/memory/mmap.rs:59) | `USER_VIRT_BUMP` has no free list | "`munmap_user` has no callers and no `SYS_MUNMAP` exists, so a free list would be unreachable code with an O(n) scan on the reservation path" | "Upgrade path, in order: wire `SYS_MUNMAP` in `syscall/table.rs`, then have `munmap_user` push `(start, pages)` onto a coalescing free list that `mmap_user` first-fits before advancing the bump." |

## atlas

| Location | Simplified | Ceiling | Upgrade trigger |
|---|---|---|---|
| [atlas/relobj.rs:420](kernel/seal-os/src/atlas/relobj.rs:420) | On unload, frames are returned but the virtual range is not | "the kernel VA bump allocator has no free list" | "Add one if chart churn matters." |

## aether-core

| Location | Simplified | Ceiling | Upgrade trigger |
|---|---|---|---|
| [aether-core/src/attention.rs:496](kernel/epsilon/epsilon/crates/aether-core/src/attention.rs:496) | `single_linkage_clusters` uses O(n²) edge enumeration plus a sort | "`sparse_attention` takes a dense `[seq, seq]` mask, so the kernel around this call is already Theta(seq^2) in both time and memory, and a neighbour graph here would save nothing" | "Trigger: revisit when the mask stops being materialised dense (a block or index list instead of `[seq, seq]` bools), at which point the quadratic edge scan becomes the ceiling and a k-NN graph over `prepared` is the upgrade." |
| [aether-core/src/diagram.rs:130](kernel/epsilon/epsilon/crates/aether-core/src/diagram.rs:130) | `wasserstein_distance` uses a cubic Hungarian solver | "Bar count is bounded by `PersistenceConfig`'s `max_points`... 512 in the widest preset (`h0_only`), 128 in `h1_dense` — so `(n + m)` stays in the hundreds" | "Trigger: any preset's `max_points` raised past 2048, or a caller assembling diagrams outside those caps. Then swap in an auction or Sinkhorn solver — the tests here pin the answer either way." |
| [aether-core/src/ml/dispatch.rs:61](kernel/epsilon/epsilon/crates/aether-core/src/ml/dispatch.rs:61) | `Signal::effective` uses linear decay | none stated | "swap for exp if staleness curve matters." |
| [aether-core/src/ml/dispatch.rs:145](kernel/epsilon/epsilon/crates/aether-core/src/ml/dispatch.rs:145) | Queue pop rebuilds the whole heap, O(n) per pop | none stated | "use a decay-aware indexed heap when queue grows." |
| [aether-core/src/ml/neural.rs:481](kernel/epsilon/epsilon/crates/aether-core/src/ml/neural.rs:481) | `result.converged` reports true on a finite stall even at a bad loss (e.g. a saturated sigmoid pinned at 0.5) | "separating \"stopped\" from \"stopped somewhere useful\" needs a problem-scale reference this signature does not have" | none stated |

## seal-graph

| Location | Simplified | Ceiling | Upgrade trigger |
|---|---|---|---|
| [seal-graph/src/exec.rs:568](kernel/seal-graph/src/exec.rs:568) | `flat` takes one copy per operand | "keeps every kernel below a plain slice loop and sidesteps deadlocking on a graph that feeds the same value into two operands of one op (`spin::Mutex` is not reentrant)" | "Operate on the guards directly if a profile ever shows the copies mattering." |

## No stated trigger

- [aether-core/src/ml/neural.rs:481](kernel/epsilon/epsilon/crates/aether-core/src/ml/neural.rs:481) — names no upgrade path or condition; the comment explains why callers must pair the flag with `final_loss` instead.
- [seal-os/src/ml_engine/foliation.rs:779](kernel/seal-os/src/ml_engine/foliation.rs:779) — names no upgrade path; the comment states the scan is only ever run offline inside a boot benchmark, never on a runtime path.
- [seal-os/src/net/tcp.rs:457](kernel/seal-os/src/net/tcp.rs:457) — names no upgrade path or condition, only that it shares the same ceiling as the flush loop's own `remove(0)`.

The controller's candidate list also named `seal-os/src/fs/ext2.rs:1364`. That comment does state an upgrade trigger — "Upgrade path when a 4 GiB image fixture exists: `i_size.checked_add(self.block_size).ok_or(VfsError::IoError)?`..." — so it is excluded here and listed instead in the `fs` table above.

43 markers, 3 with no stated trigger.
