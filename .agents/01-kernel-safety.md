# 01 — Kernel Safety: Fix IRQ Deadlocks, Static Mut, Unwrap Panics

## Goal

Eliminate every path that can deadlock, triple-fault, or invoke undefined behavior in the kernel.

## Current State

### IRQ handler deadlocks (`kernel/seal-os/src/drivers/interrupts.rs`)
- **Line ~253**: Timer handler acquires `TICKS` spinlock. If mainline code holds `TICKS` when the PIT fires, the handler spins forever → deadlock → system hang.
- **Lines ~258-268**: Keyboard handler acquires `EVENT_QUEUE` lock. Same deadlock if mainline is mid-poll.
- **Lines ~271-300**: Mouse handler acquires **three** locks sequentially (`MOUSE_STATE`, `EVENT_QUEUE`, `LAST_BUTTONS`). Worst-case deadlock surface in the entire kernel.

### `static mut FRAMEBUFFER` (`kernel/seal-os/src/main.rs:34`)
- Two `unsafe` blocks read/write a `static mut` with no synchronization. The compiler is free to reorder or duplicate accesses → undefined behavior under optimizations.

### `unwrap()` panics (`kernel/seal-os/src/fs/manifold_fs.rs`)
- 8 `unwrap()` calls on `BTreeMap::get_mut()` after `contains_key()` — safe in single-threaded code but fragile.
- **Line ~277**: `partial_cmp().unwrap()` on `f64` — panics on NaN. Similarity scores can produce NaN from 0/0 division.

### `static mut HEAP` (`kernel/seal-os/src/memory/mod.rs:15`)
- `dealloc` is a silent no-op. Any `Box::drop` or `Vec` resize leaks permanently. Acceptable for Phase 1 but must be documented.

## Gap Analysis

| Issue | Severity | Can crash? |
|-------|----------|------------|
| IRQ spinlock deadlock | P0 | Yes — infinite hang |
| NaN `unwrap()` | P0 | Yes — kernel panic → triple fault |
| `static mut` UB | P1 | Possible under LTO/optimization |
| BTreeMap `unwrap()` | P1 | Yes on edge cases |
| Silent dealloc leak | P3 | OOM after heavy use |

## Implementation Steps

1. **Replace `TICKS` with `AtomicU64`**
   - In `interrupts.rs`, change `static TICKS: Mutex<u64>` to `static TICKS: AtomicU64`
   - Timer handler uses `TICKS.fetch_add(1, Ordering::Relaxed)`
   - `ticks()` function uses `TICKS.load(Ordering::Relaxed)`
   - No lock needed, no deadlock possible

2. **Use `try_lock()` in keyboard and mouse IRQ handlers**
   - Replace `EVENT_QUEUE.lock()` with `EVENT_QUEUE.try_lock()`
   - If contention → drop the event (acceptable: user presses key again)
   - Same for `MOUSE_STATE` and `LAST_BUTTONS`
   - Alternative: wrap all mainline `EVENT_QUEUE.lock()` calls with `x86_64::instructions::interrupts::without_interrupts(|| { ... })`

3. **Replace `static mut FRAMEBUFFER` with `spin::Once<Framebuffer>`**
   - `static FRAMEBUFFER: spin::Once<Framebuffer> = spin::Once::new();`
   - Init: `FRAMEBUFFER.call_once(|| Framebuffer::new(...))`
   - Access: `FRAMEBUFFER.get().unwrap()` (safe after boot init)
   - Remove both `unsafe` blocks in `main.rs`

4. **Fix NaN panic in ManifoldFS**
   - `manifold_fs.rs:277`: Replace `partial_cmp().unwrap()` with `total_cmp()` (stable since Rust 1.62)
   - Or use `partial_cmp().unwrap_or(core::cmp::Ordering::Equal)`

5. **Replace check-then-unwrap with `if let` / `entry()` API**
   - All 8 `contains_key()` + `get_mut().unwrap()` patterns → `if let Some(v) = map.get_mut(&key)`
   - For insert-or-update patterns → `map.entry(key).or_insert_with(|| ...)`

6. **Add `// SAFETY:` comments to all `unsafe impl`**
   - `framebuffer.rs:14-15`: Document that the raw pointer is a memory-mapped I/O region, single-writer (kernel main thread only), never deallocated.

7. **Add dealloc leak tracking (optional, P3)**
   - In `memory/mod.rs`, add `static LEAKED_BYTES: AtomicUsize` incremented in `dealloc()`
   - Expose `pub fn leaked_bytes() -> usize` for diagnostics

## Dependencies

None — this is Phase 1, no other plans depend on it completing first, but all other plans assume kernel stability.

## Acceptance Criteria

- [ ] `cargo +nightly build --release` in `kernel/seal-os/` passes
- [ ] No `static mut` in `main.rs`
- [ ] Zero `unwrap()` calls in `manifold_fs.rs` (except documented-safe ones)
- [ ] `TICKS` is `AtomicU64`, not `Mutex<u64>`
- [ ] IRQ handlers use `try_lock()` or mainline disables interrupts around lock acquisition
- [ ] All `unsafe impl Send/Sync` have `// SAFETY:` comments
- [ ] QEMU smoke test passes all 20 boot milestones

## Files to Modify

- `kernel/seal-os/src/drivers/interrupts.rs`
- `kernel/seal-os/src/main.rs`
- `kernel/seal-os/src/fs/manifold_fs.rs`
- `kernel/seal-os/src/graphics/framebuffer.rs`
- `kernel/seal-os/src/memory/mod.rs`
