# 02 — Memory Allocator: Replace Bump Allocator with Free-List

## Goal

Replace the current bump-only allocator (which never frees memory) with a linked-list free-list allocator that supports real `dealloc`, enabling long-running kernel operations without OOM.

## Current State

`kernel/seal-os/src/memory/mod.rs` — 54 lines total:
- 16 MB static `HeapStorage` array, page-aligned
- `BumpAllocator` with a single `next: usize` pointer
- `alloc()` bumps forward, aligns, bounds-checks
- `dealloc()` is literally empty — every allocation is permanent
- `init_heap()` is a no-op marker function

ManifoldFS (`kernel/seal-os/src/fs/manifold_fs.rs`) creates `String`, `Vec<f64>`, `BTreeMap` entries on every file operation. Under sustained use, the 16 MB heap will exhaust.

## Gap Analysis

- No deallocation → every `Vec::drop`, `String::drop`, `BTreeMap::remove` leaks
- No fragmentation handling
- No OOM reporting (returns `null_mut()` silently)
- No heap usage metrics

## Implementation Steps

1. **Implement a linked-list free-list allocator**
   - On `dealloc()`: insert the freed block into a sorted linked list
   - On `alloc()`: walk the free list for first-fit (or best-fit) block ≥ requested size
   - If no free block fits, fall back to bumping from the remaining heap
   - Split oversized free blocks to reduce internal fragmentation
   - Merge adjacent free blocks on dealloc (coalescing)

2. **Data structures**
   ```
   struct FreeBlock {
       size: usize,
       next: Option<NonNull<FreeBlock>>,
   }
   ```
   - Minimum block size = `size_of::<FreeBlock>()` (16 bytes on x86_64)
   - Free list head stored in a `Mutex<Option<NonNull<FreeBlock>>>`

3. **Keep bump allocator as fallback**
   - Free list is checked first
   - If no suitable block, bump from `next` pointer (existing logic)
   - This means the allocator starts as a bump allocator and gains free-list behavior as blocks are freed

4. **Add heap diagnostics**
   - `pub fn heap_used() -> usize` — bytes allocated (bump pointer position minus free list total)
   - `pub fn heap_free() -> usize` — HEAP_SIZE minus heap_used
   - `pub fn heap_fragmentation() -> usize` — number of free blocks (lower = less fragmented)

5. **OOM handling**
   - When both free-list and bump are exhausted, log via `serial_println!` before returning `null_mut()`
   - Consider a kernel OOM handler that dumps heap stats

6. **Testing**
   - Unit tests (in `std` test harness): alloc/dealloc cycles, fragmentation recovery, alignment correctness
   - Stress test: allocate and free 10,000 blocks of random sizes

## Dependencies

- **01-kernel-safety** must complete first (the allocator lock must not deadlock with IRQ handlers)

## Acceptance Criteria

- [ ] `dealloc()` actually frees memory (re-usable by subsequent `alloc()`)
- [ ] Coalescing works: alloc A, alloc B, free B, free A → single contiguous free block
- [ ] Alignment guarantees preserved (4096-byte page alignment for large allocs)
- [ ] Heap diagnostics available via `memory::heap_used()` etc.
- [ ] ManifoldFS can create and delete 1000 files without OOM
- [ ] `cargo +nightly build --release` passes
- [ ] Existing QEMU smoke test passes

## Files to Modify

- `kernel/seal-os/src/memory/mod.rs` (major rewrite)

## Files to Create

- `kernel/seal-os/src/memory/free_list.rs` (optional: separate free-list logic)
