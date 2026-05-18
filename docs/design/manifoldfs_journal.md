# ManifoldFS Journaling & Persistence Design Document

> **Phase X Task**: CX-007
> **Priority**: HIGH (data integrity)
> **Estimated Effort**: 1 month

---

## Overview

ManifoldFS is currently in-memory. This design adds block-device persistence with write-ahead logging (journaling).

## On-Disk Layout

```
LBA 0: Boot sector (MBR-compatible, magic 0x55AA at offset 510)
LBA 1-7: Superblock
  - magic: "MANI"
  - version: 1
  - block_size: 4096
  - inode_count
  - free_block_bitmap_start
  - journal_start
  - journal_size
LBA 8-39: Journal (circular log)
  - Each entry: opcode + inode_id + old_value + new_value + checksum
LBA 40-1031: Inode table (fixed size, 1 inode per 64 bytes)
LBA 1032+: Data blocks
  - File contents
  - Directory entries (BTreeMap serialization)
  - Sphere point clouds
```

## Journaling Protocol

### Transaction Begin
1. Write `TX_START` to journal
2. Write all modified inode records
3. Write `TX_COMMIT` to journal
4. Flush journal to disk (AHCI cache flush)
5. Write inodes to inode table
6. Write data blocks
7. Write `TX_DONE` to journal

### Recovery
1. Scan journal from last known checkpoint
2. If `TX_COMMIT` found without `TX_DONE`, replay transaction
3. If `TX_START` without `TX_COMMIT`, discard (partial transaction)

## Write Ordering

Use AHCI write cache flush after each critical step. If NCQ available, use FUA (Force Unit Access) bits.

## Verification

- `test_crash_recovery`: Write file, simulate power loss in QEMU (`stop` + `savevm`), reboot, verify file exists
- `test_fsck`: Corrupt random LBAs, run `fsck.manifoldfs`, verify repair

---

*Design document produced by Phase X Planning*
*2026-05-18*
