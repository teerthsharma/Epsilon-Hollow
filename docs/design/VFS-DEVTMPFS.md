# VFS, /dev Filesystem, tmpfs, and initrd — Design Document

> **Phase 3 Task**: VFS-001
> **Priority**: HIGH (required for Seal ABI device access and early userspace)
> **Estimated Effort**: 4 weeks
> **Blocked by**: Syscall dispatcher, block device abstraction, physical allocator
> **Blocks**: SealShell, Aether runtime, device drivers (console, null, zero, random), package manager

---

## Overview

Current filesystem is a single `ManifoldFS` instance. Seal OS uses a VFS layer to multiplex across native filesystem types: rootfs (ManifoldFS or ext2 image interop), `/dev` (devtmpfs), `/tmp` (tmpfs), `/proc` (procfs), and initrd (ramdisk). This design covers the VFS abstraction, devtmpfs, tmpfs, initrd loading, and mount/umount semantics without adopting a Unix contract.

---

## Architecture

### 1. VFS Layer

```rust
pub struct Inode {
    pub ino: u64,
    pub mode: u16,        // S_IFREG, S_IFDIR, S_IFCHR, S_IFBLK, S_IFLNK, S_IFIFO, S_IFSOCK
    pub nlink: u32,
    pub uid: u32,
    pub gid: u32,
    pub size: u64,
    pub atime: Timespec,
    pub mtime: Timespec,
    pub ctime: Timespec,
    pub blksize: u32,
    pub blocks: u64,
    pub rdev: u32,        // for S_IFCHR / S_IFBLK
    
    // VFS-internal
    pub fs: &'static dyn FileSystem,
    pub private: *mut c_void,  // FS-specific data (ManifoldFS inode id, etc.)
}

pub struct Dentry {
    pub name: String,
    pub parent: *mut Dentry,
    pub inode: Option<Arc<Inode>>,
    pub children: HashMap<String, Arc<Dentry>>,
    pub mounted_here: Option<Arc<VfsMount>>,  // mount point
}

pub struct VfsMount {
    pub mnt_root: Arc<Dentry>,      // root dentry of mounted FS
    pub mnt_parent: Option<Arc<VfsMount>>,
    pub mnt_mountpoint: Option<Arc<Dentry>>,  // where we're mounted
    pub fs: &'static dyn FileSystem,
    pub flags: MountFlags,
}
```

- [x] Define `Inode` struct with Seal metadata fields
- [x] Define `Timespec` struct
- [x] Define `Dentry` struct with mount point handling
- [x] Define `VfsMount` struct
- [x] Define `MountFlags` bitflags (MS_RDONLY, MS_NOSUID, MS_NODEV, MS_NOEXEC, MS_SYNCHRONOUS, MS_REMOUNT, MS_BIND, MS_MOVE)
- [x] Implement `lookup_path(path, follow_symlinks)` — full path resolution
- [x] Handle `.` and `..` components correctly
- [x] Handle mount point crossing (`..` from mount root goes to parent mount)
- [x] Implement symlink following (with loop detection, max 40 hops)
- [x] Implement dcache (dentry cache) for fast repeated lookups
- [x] Implement `nameidata` structure for path walk state
- [x] Handle absolute vs relative paths

### 2. FileSystem Trait

```rust
pub trait FileSystem: Send + Sync {
    fn name(&self) -> &'static str;
    
    // Inode operations
    fn lookup(&self, dir: &Inode, name: &str) -> Result<Arc<Inode>, Errno>;
    fn create(&self, dir: &mut Inode, name: &str, mode: u16) -> Result<Arc<Inode>, Errno>;
    fn mkdir(&self, dir: &mut Inode, name: &str, mode: u16) -> Result<Arc<Inode>, Errno>;
    fn unlink(&self, dir: &mut Inode, name: &str) -> Result<(), Errno>;
    fn rmdir(&self, dir: &mut Inode, name: &str) -> Result<(), Errno>;
    fn symlink(&self, dir: &mut Inode, name: &str, target: &str) -> Result<Arc<Inode>, Errno>;
    fn link(&self, old: &Inode, dir: &mut Inode, name: &str) -> Result<(), Errno>;
    fn rename(&self, old_dir: &mut Inode, old_name: &str, new_dir: &mut Inode, new_name: &str) -> Result<(), Errno>;
    
    // File operations
    fn read(&self, inode: &Inode, buf: &mut [u8], offset: u64) -> Result<usize, Errno>;
    fn write(&self, inode: &mut Inode, buf: &[u8], offset: u64) -> Result<usize, Errno>;
    fn truncate(&self, inode: &mut Inode, size: u64) -> Result<(), Errno>;
    
    // Directory operations
    fn readdir(&self, inode: &Inode, offset: u64, entries: &mut [DirEntry]) -> Result<usize, Errno>;
    
    // Metadata
    fn getattr(&self, inode: &Inode) -> Result<Stat, Errno>;
    fn setattr(&self, inode: &mut Inode, attr: &SetAttr) -> Result<(), Errno>;
    
    // Sync
    fn sync(&self) -> Result<(), Errno>;
}
```

- [x] Define `FileSystem` trait with all methods
- [x] Define `Errno` enum with all standard errno values
- [x] Define `Stat` struct for Seal ABI metadata
- [x] Define `SetAttr` struct for setattr
- [x] Define `DirEntry` struct for readdir
- [x] Implement `FileOperations` trait for per-file ops
- [x] Implement `File` struct wrapping dentry + position + flags
- [x] Implement `FdTable` for per-process file descriptors
- [x] Implement fd allocation (lowest available)

### 3. devtmpfs

Auto-populated `/dev` filesystem. Created at boot, before userspace init.

```rust
pub struct DevTmpfs;

impl DevTmpfs {
    pub fn populate() {
        self.mknod("null",  S_IFCHR | 0o666, makedev(1, 3));
        self.mknod("zero",  S_IFCHR | 0o666, makedev(1, 5));
        self.mknod("full",  S_IFCHR | 0o666, makedev(1, 7));
        self.mknod("random", S_IFCHR | 0o666, makedev(1, 8));
        self.mknod("urandom", S_IFCHR | 0o666, makedev(1, 9));
        self.mknod("tty",   S_IFCHR | 0o666, makedev(5, 0));
        self.mknod("console", S_IFCHR | 0o600, makedev(5, 1));
        self.mknod("pts",   S_IFDIR | 0o755, 0);
        
        // Block devices discovered at boot
        for (major, minor, name) in block_dev_list() {
            self.mknod(name, S_IFBLK | 0o660, makedev(major, minor));
        }
        
        // Network interfaces
        for iface in net_ifaces() {
            self.mknod(&format!("net/{}", iface.name()), S_IFCHR | 0o600, makedev(10, 200));
        }
    }
}
```

- [x] Implement `DevTmpfs` struct implementing `FileSystem`
- [x] Implement `mknod()` for device files
- [x] Implement `mkdir()` for directories
- [x] Populate `/dev/null` (major 1, minor 3)
- [x] Populate `/dev/zero` (major 1, minor 5)
- [x] Populate `/dev/full` (major 1, minor 7)
- [x] Populate `/dev/random` (major 1, minor 8)
- [x] Populate `/dev/urandom` (major 1, minor 9)
- [x] Populate `/dev/tty` (major 5, minor 0)
- [x] Populate `/dev/console` (major 5, minor 1)
- [x] Create `/dev/pts` directory
- [x] Auto-populate block devices on driver registration
- [x] Auto-populate network interfaces on ifup

**Device file operations:**
```rust
pub trait CharDevice: Send + Sync {
    fn read(&self, buf: &mut [u8]) -> Result<usize, Errno>;
    fn write(&self, buf: &[u8]) -> Result<usize, Errno>;
    fn ioctl(&self, cmd: u32, arg: u64) -> Result<i64, Errno>;
    fn poll(&self) -> PollFlags;
}
```

- [x] Define `CharDevice` trait
- [x] Implement `NullDevice` — read returns 0, write discards
- [x] Implement `ZeroDevice` — read fills with 0x00
- [x] Implement `FullDevice` — write returns -ENOSPC
- [x] Implement `RandomDevice` — blocks until entropy, uses RDRAND + jitter
- [x] Implement `UrandomDevice` — never blocks, pseudo-random fallback
- [x] Implement `TtyDevice` — serial console I/O
- [x] Implement `ConsoleDevice` — framebuffer or serial console
- [x] Register all char devices in `CHAR_DEVS` hash map
- [x] Implement `makedev(major, minor)` and `major(dev)`, `minor(dev)` macros

### 4. tmpfs

RAM-backed filesystem for `/tmp`, `/run`, `/var/tmp`.

```rust
pub struct Tmpfs {
    inodes: Mutex<HashMap<u64, TmpfsInode>>,
    next_ino: AtomicU64,
    total_pages: AtomicUsize,
    max_pages: usize,  // Configurable; default = total_ram / 4
}

struct TmpfsInode {
    metadata: Inode,
    data: Option<Vec<u8>>,      // For regular files
    children: Option<HashMap<String, u64>>,  // For directories
    symlink_target: Option<String>,  // For symlinks
}
```

- [x] Implement `Tmpfs` struct implementing `FileSystem`
- [x] Implement file create/read/write/truncate
- [x] Implement directory create/lookup/remove
- [x] Implement symlink create/readlink
- [x] Implement hard link
- [x] Implement rename
- [x] Enforce `max_pages` limit (return -ENOSPC)
- [x] Track `total_pages` accurately
- [x] No persistence across reboot
- [x] Swappable pages (v2)

### 5. procfs

Pseudo-filesystem exposing kernel state. Minimal v1:

```
/proc/
├── cpuinfo          # CPU model, cores, features
├── meminfo          # Total/used/free RAM
├── uptime           # Seconds since boot
├── version          # Kernel version string
├── cmdline          # Kernel command line
├── self/            # Symlink to current PID
│   ├── exe          # Symlink to executable
│   ├── maps         # Memory mappings
│   ├── status       # PID, state, RSS, threads
│   ├── fd/          # Symlinks to open files
│   └── ...
├── sys/
│   ├── kernel/
│   │   └── hostname
│   └── net/
│       └── ipv4/ip_forward
```

- [x] Implement `ProcFs` struct implementing `FileSystem`
- [x] Generate `/proc/cpuinfo` from CPUID
- [x] Generate `/proc/meminfo` from allocator stats
- [x] Generate `/proc/uptime` from `ticks()`
- [x] Generate `/proc/version` string
- [x] Generate `/proc/cmdline` from boot args
- [x] Implement `/proc/self` symlink to current PID
- [x] Implement `/proc/<pid>/status` with PID, state, RSS
- [x] Implement `/proc/<pid>/maps` with memory mappings
- [x] Implement `/proc/<pid>/fd/` directory with symlinks
- [x] Implement `/proc/sys/kernel/hostname`
- [x] Implement `/proc/sys/net/ipv4/ip_forward`
- [x] All procfs files generated on-the-fly (no storage)

### 6. initrd / initramfs

**Format:** CPIO archive (newc format), optionally gzipped. Embedded in kernel image or loaded by bootloader.

```rust
fn load_initrd() -> Result<Tmpfs, Error> {
    let initrd_start = &__initrd_start as *const u8;
    let initrd_end = &__initrd_end as *const u8;
    let len = initrd_end as usize - initrd_start as usize;
    
    let decompressed = if is_gzip(initrd_start, len) {
        gzip_decompress(initrd_start, len)?
    } else {
        slice::from_raw_parts(initrd_start, len)
    };
    
    let fs = Tmpfs::new();
    for entry in cpio::parse(decompressed) {
        match entry.typ {
            CpioType::Dir => fs.mkdir(entry.path, entry.mode)?,
            CpioType::File => fs.write_file(entry.path, entry.data, entry.mode)?,
            CpioType::Symlink => fs.symlink(entry.path, entry.target)?,
            CpioType::Node => fs.mknod(entry.path, entry.mode, entry.rdev)?,
        }
    }
    Ok(fs)
}
```

- [x] Define `__initrd_start` / `__initrd_end` linker symbols
- [x] Detect gzip compression (magic bytes 0x1f 0x8b)
- [x] Implement gzip decompression (or use static decompression library)
- [x] Implement CPIO newc format parser
- [x] Handle directories, regular files, symlinks, device nodes
- [x] Mount initrd as root filesystem if present
- [x] Run `/init` from initrd
- [x] Implement `pivot_root` for switching to real root
- [x] If no initrd, mount ManifoldFS directly as root

### 7. Mount Table

```rust
static MOUNT_TABLE: RwLock<Vec<VfsMount>>;
```

- [x] Implement `mount(source, target, fstype, flags, data)` syscall
- [x] Implement `umount2(target, flags)` syscall
- [x] Support `MS_RDONLY`, `MS_NOSUID`, `MS_NODEV`, `MS_NOEXEC`, `MS_SYNCHRONOUS`
- [x] Support `MS_REMOUNT` for changing flags
- [x] Support `MS_BIND` for bind mounts (v2)
- [x] Support `MS_MOVE` for moving mount points (v2)
- [x] Verify mount target exists and is a directory
- [x] Update `MOUNT_TABLE` on mount/umount
- [x] Handle mount propagation (shared/subtree mounts v2)

---

## Verification

| Test | What it proves | Status |
|---|---|---|
| `test_vfs_lookup` | `lookup_path("/dev/null")` returns correct inode with mode S_IFCHR | [x] |
| `test_vfs_cross_mount` | `lookup_path("/mnt/manifold/file.txt")` crosses mount boundary correctly | [x] |
| `test_devtmpfs_read_null` | `read("/dev/null", buf, 1024)` returns 0 | [x] |
| `test_devtmpfs_write_null` | `write("/dev/null", buf, 1024)` returns 1024, data discarded | [x] |
| `test_devtmpfs_read_zero` | `read("/dev/zero", buf, 1024)` returns 1024, all zeros | [x] |
| `test_devtmpfs_random_entropy` | `read("/dev/random", buf, 8)` returns 8, bytes vary across calls | [x] |
| `test_tmpfs_create_write_read` | Create file in /tmp, write, read back, data matches | [x] |
| `test_tmpfs_enospc` | Fill /tmp to max_pages, next write returns -ENOSPC | [x] |
| `test_proc_uptime` | `read("/proc/uptime")` returns positive number, increases across reads | [x] |
| `test_proc_self_pid` | `readlink("/proc/self")` returns current PID | [x] |
| `test_initrd_load` | CPIO archive loaded, /init exists and is executable | [x] |
| `test_mount_tmpfs` | `mount("tmpfs", "/mnt", "tmpfs", 0, None)` succeeds, files visible | [x] |
| `test_umount` | `umount("/mnt")` succeeds, files no longer visible | [x] |

---

*Design document produced by Phase 3 Planning*
*2026-05-19*
