# Buffer Cache — Design Document

> **Phase 4 Task**: BC-001
> **Priority**: MEDIUM (performance; required before large-file workloads)
> **Estimated Effort**: 3 weeks
> **Blocked by**: Block device abstraction, physical allocator for DMA, interrupt framework
> **Blocks**: Large file performance, write-back persistence, read-ahead, mmap file backing

---

## Overview

Raw AHCI/virtio reads every block from disk. A buffer cache amortizes disk I/O by keeping recently accessed blocks in RAM, batching writes, and pre-fetching sequential streams. This is the difference between 5 MB/s and 500 MB/s on rotating media.

---

## Architecture

### 1. Buffer Head (`struct Buffer`)

```rust
pub struct Buffer {
    dev: DeviceId,           // Major:minor
    block: u64,              // Block number on device
    data: DmaBuffer,         // 4 KiB page, DMA-coherent
    
    state: BufferState,      // Clean, Dirty, Locked, Invalid
    ref_count: AtomicU32,
    
    // LRU list
    lru_prev: *mut Buffer,
    lru_next: *mut Buffer,
    
    // Hash table chain
    hash_next: *mut Buffer,
}

enum BufferState {
    Invalid,    // Allocated but no valid data
    Clean,      // Matches disk
    Dirty,      // Modified, not yet written
    Locked,     // I/O in progress
}
```

- [ ] Define `Buffer` struct with all fields
- [ ] Define `BufferState` enum
- [ ] Define `BufferRef` RAII guard (increments ref_count on clone, decrements on drop)
- [ ] Allocate buffer metadata from slab (≤ 128 bytes)
- [ ] Allocate buffer data as 4 KiB DMA pages
- [ ] Implement `Buffer::data()` returning `&[u8]` / `&mut [u8]`
- [ ] Implement `Buffer::mark_dirty()`
- [ ] Implement `Buffer::mark_clean()`
- [ ] Implement `Buffer::lock()` / `Buffer::unlock()` for I/O serialization

### 2. Hash Table

```rust
const HASH_BUCKETS: usize = 1024;
static HASH_TABLE: [Mutex<*mut Buffer>; HASH_BUCKETS];

fn hash(dev: DeviceId, block: u64) -> usize {
    fxhash64((dev.as_u64() << 32) | block) as usize % HASH_BUCKETS
}
```

- [ ] Define `HASH_BUCKETS = 1024`
- [ ] Allocate hash table array
- [ ] Implement `hash(dev, block)` function
- [ ] Implement `bfind(dev, block)` — lookup with ref_count increment
- [ ] Implement `binsert(buf)` — add to hash chain
- [ ] Implement `bremove(dev, block)` — remove from hash chain
- [ ] Protect hash bucket operations with spinlock
- [ ] Measure hash collision rate under load

### 3. LRU Eviction

```rust
static LRU_HEAD: Mutex<*mut Buffer>;
static LRU_TAIL: Mutex<*mut Buffer>;
static FREE_COUNT: AtomicUsize;
```

- [ ] Define intrusive doubly-linked LRU list
- [ ] Implement `lru_add_head(buf)` — most recently used
- [ ] Implement `lru_move_to_head(buf)` — on access
- [ ] Implement `lru_remove(buf)` — on eviction
- [ ] Implement `alloc_buffer()` — pop from free list or evict LRU tail
- [ ] Eviction policy: prefer `Clean` buffers with `ref_count == 0`
- [ ] If no clean buffers, synchronous write-back of oldest `Dirty` buffer
- [ ] Track `FREE_COUNT` (buffers available without eviction)
- [ ] Implement `brelease(buf)` — decrement ref_count, move to LRU tail if 0

### 4. Read Path

```rust
pub fn bread(dev: DeviceId, block: u64) -> Result<BufferRef, Error> {
    if let Some(buf) = bfind(dev, block) {
        return Ok(BufferRef(buf));  // Cache hit
    }
    
    let buf = alloc_buffer()?;
    buf.dev = dev;
    buf.block = block;
    buf.state = BufferState::Locked;
    hash_insert(buf);
    
    // Submit async read
    let bd = block_dev_lookup(dev)?;
    bd.read_sectors(block * 8, buf.data.as_mut())?;
    
    buf.state = BufferState::Clean;
    Ok(BufferRef(buf))
}
```

- [ ] Implement `bread(dev, block)` — cache hit fast path
- [ ] Implement cache miss allocation + I/O submission
- [ ] Synchronous interface: spin/sleep until `state != Locked`
- [ ] Return `BufferRef` so caller can access data
- [ ] Async interface `bread_a(dev, block, callback)` for v2
- [ ] Count cache hits vs misses for telemetry

### 5. Write Path

```rust
pub fn bwrite(buf: &mut Buffer) {
    buf.state = BufferState::Dirty;
    // Does NOT write immediately
}

pub fn bflush(dev: Option<DeviceId>) {
    // Walk all buffers; for each Dirty buffer matching dev:
    //   state = Locked
    //   submit write
    //   on completion: state = Clean
}
```

- [ ] Implement `bwrite(buf)` — mark dirty, no immediate I/O
- [ ] Implement `bflush(dev)` — write all dirty buffers for device
- [ ] Implement `bflush_all()` — write all dirty buffers globally
- [ ] Implement `bsync(buf)` — synchronous write of single buffer
- [ ] Track dirty buffer count for telemetry

### 6. Read-Ahead

```rust
fn maybe_readahead(dev: DeviceId, block: u64, access_pattern: &Pattern) {
    if access_pattern.is_sequential() {
        let ahead = min(block + 1 + READAHEAD_WINDOW, dev.max_block);
        for b in (block + 1)..ahead {
            if bfind(dev, b).is_none() {
                let buf = alloc_buffer();
                if buf.is_ok() {
                    submit_async_read(dev, b, buf);
                }
            }
        }
    }
}
```

- [ ] Implement per-file-descriptor `AccessPattern` tracking
- [ ] Detect sequential access (`block == last_block + 1` for N consecutive reads)
- [ ] Define `READAHEAD_WINDOW` (default 32 blocks = 128 KiB)
- [ ] Submit read-ahead requests without blocking caller
- [ ] Do NOT trigger read-ahead for random access patterns
- [ ] Cap total read-ahead in-flight per device
- [ ] Measure read-ahead hit rate

### 7. Write-Back vs Write-Through

| Mode | Latency | Durability | Use Case |
|---|---|---|---|
| Write-back | µs | 5-second window | Default for data blocks |
| Write-through | ms | Immediate | Metadata blocks (inode, superblock), `O_SYNC` files |
| Barrier | ms | Ordered | `fsync()`, journal commits |

- [ ] Default all regular blocks to write-back
- [ ] Mark metadata blocks as write-through
- [ ] Honor `O_SYNC` flag on file open
- [ ] Implement `fsync()` as full flush + barrier
- [ ] Implement periodic flush timer (5-second interval)
- [ ] Implement low-memory-triggered flush
- [ ] Add `sync_required` flag to `Buffer` for prioritized flush

### 8. mmap File Backing

For `mmap(MAP_SHARED)` of a file:
```rust
fn file_mmap(vma: &mut VmArea, file: &File, offset: u64) {
    vma.backing = Backing::BufferCache {
        dev: file.inode.dev,
        block_start: offset / BLOCK_SIZE,
        block_count: (vma.len + BLOCK_SIZE - 1) / BLOCK_SIZE,
    };
}
```

- [ ] Implement `file_mmap()` creating buffer-cache-backed VMA
- [ ] Page fault on file-backed VMA → `bread(dev, block)`
- [ ] Map buffer's physical page directly into process page table (read-only for MAP_SHARED)
- [ ] Write to MAP_SHARED page → COW page fault → mark buffer Dirty
- [ ] Implement `msync()` flushing dirty file-backed pages
- [ ] Implement `munmap()` releasing buffer references

---

## Verification

| Test | What it proves | Status |
|---|---|---|
| `test_bread_cache_hit` | Second `bread(dev, block)` returns without disk I/O | [ ] |
| `test_bread_cache_miss` | First read goes to disk, returns correct data | [ ] |
| `test_bwrite_dirty` | `bwrite()` marks dirty, data not on disk yet | [ ] |
| `test_bflush_persist` | After `bflush()`, power-loss simulation reads correct data | [ ] |
| `test_lru_eviction` | Fill cache + 1; oldest clean block evicted, not dirty | [ ] |
| `test_lru_dirty_stall` | All buffers dirty + alloc → synchronous write-back, no data loss | [ ] |
| `test_readahead_sequential` | Sequential read of 1 MiB file issues ≤ 10 disk commands | [ ] |
| `test_readahead_random` | Random read pattern does NOT trigger read-ahead | [ ] |
| `test_mmap_file_backing` | `mmap` file, read via pointer → no `read()` syscall, data correct | [ ] |
| `test_mmap_shared_write` | Write via mmap pointer, `msync()` → disk contains written data | [ ] |

---

*Design document produced by Phase 4 Planning*
*2026-05-19*
