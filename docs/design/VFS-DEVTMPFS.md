# VFS, /dev Filesystem, tmpfs, and initrd — Design Document

> **Phase 3 Task**: VFS-001
> **Priority**: HIGH (required for POSIX compatibility, device access, early userspace)
> **Estimated Effort**: 4 weeks
> **Blocked by**: Syscall dispatcher, block device abstraction, physical allocator
> **Blocks**: Shell, libc, device drivers (console, null, zero, random), package manager

---

## Overview

Current filesystem is a single `ManifoldFS` instance. POSIX requires a VFS layer that multiplexes across multiple filesystem types: rootfs (ManifoldFS or ext2), `/dev` (devtmpfs), `/tmp` (tmpfs), `/proc` (procfs), and initrd (ramdisk). This design covers the VFS abstraction, devtmpfs, tmpfs, initrd loading, and mount/umount semantics.

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

- [ ] Define `Inode` struct with all POSIX metadata fields
- [ ] Define `Timespec` struct
- [ ] Define `Dentry` struct with mount point handling
- [ ] Define `VfsMount` struct
- [ ] Define `MountFlags` bitflags (MS_RDONLY, MS_NOSUID, MS_NODEV, MS_NOEXEC, MS_SYNCHRONOUS, MS_REMOUNT, MS_BIND, MS_MOVE)
- [ ] Implement `lookup_path(path, follow_symlinks)` — full path resolution
- [ ] Handle `.` and `..` components correctly
- [ ] Handle mount point crossing (`..` from mount root goes to parent mount)
- [ ] Implement symlink following (with loop detection, max 40 hops)
- [ ] Implement dcache (dentry cache) for fast repeated lookups
- [ ] Implement `nameidata` structure for path walk state
- [ ] Handle absolute vs relative paths

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

- [ ] Define `FileSystem` trait with all methods
- [ ] Define `Errno` enum with all standard errno values
- [ ] Define `Stat` struct matching Linux `struct stat`
- [ ] Define `SetAttr` struct for setattr
- [ ] Define `DirEntry` struct for readdir
- [ ] Implement `FileOperations` trait for per-file ops
- [ ] Implement `File` struct wrapping dentry + position + flags
- [ ] Implement `FdTable` for per-process file descriptors
- [ ] Implement fd allocation (lowest available)

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

- [ ] Implement `DevTmpfs` struct implementing `FileSystem`
- [ ] Implement `mknod()` for device files
- [ ] Implement `mkdir()` for directories
- [ ] Populate `/dev/null` (major 1, minor 3)
- [ ] Populate `/dev/zero` (major 1, minor 5)
- [ ] Populate `/dev/full` (major 1, minor 7)
- [ ] Populate `/dev/random` (major 1, minor 8)
- [ ] Populate `/dev/urandom` (major 1, minor 9)
- [ ] Populate `/dev/tty` (major 5, minor 0)
- [ ] Populate `/dev/console` (major 5, minor 1)
- [ ] Create `/dev/pts` directory
- [ ] Auto-populate block devices on driver registration
- [ ] Auto-populate network interfaces on ifup

**Device file operations:**
```rust
pub trait CharDevice: Send + Sync {
    fn read(&self, buf: &mut [u8]) -> Result<usize, Errno>;
    fn write(&self, buf: &[u8]) -> Result<usize, Errno>;
    fn ioctl(&self, cmd: u32, arg: u64) -> Result<i64, Errno>;
    fn poll(&self) -> PollFlags;
}
```

- [ ] Define `CharDevice` trait
- [ ] Implement `NullDevice` — read returns 0, write discards
- [ ] Implement `ZeroDevice` — read fills with 0x00
- [ ] Implement `FullDevice` — write returns -ENOSPC
- [ ] Implement `RandomDevice` — blocks until entropy, uses RDRAND + jitter
- [ ] Implement `UrandomDevice` — never blocks, pseudo-random fallback
- [ ] Implement `TtyDevice` — serial console I/O
- [ ] Implement `ConsoleDevice` — framebuffer or serial console
- [ ] Register all char devices in `CHAR_DEVS` hash map
- [ ] Implement `makedev(major, minor)` and `major(dev)`, `minor(dev)` macros

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

- [ ] Implement `Tmpfs` struct implementing `FileSystem`
- [ ] Implement file create/read/write/truncate
- [ ] Implement directory create/lookup/remove
- [ ] Implement symlink create/readlink
- [ ] Implement hard link
- [ ] Implement rename
- [ ] Enforce `max_pages` limit (return -ENOSPC)
- [ ] Track `total_pages` accurately
- [ ] No persistence across reboot
- [ ] Swappable pages (v2)

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

- [ ] Implement `ProcFs` struct implementing `FileSystem`
- [ ] Generate `/proc/cpuinfo` from CPUID
- [ ] Generate `/proc/meminfo` from allocator stats
- [ ] Generate `/proc/uptime` from `ticks()`
- [ ] Generate `/proc/version` string
- [ ] Generate `/proc/cmdline` from boot args
- [ ] Implement `/proc/self` symlink to current PID
- [ ] Implement `/proc/<pid>/status` with PID, state, RSS
- [ ] Implement `/proc/<pid>/maps` with memory mappings
- [ ] Implement `/proc/<pid>/fd/` directory with symlinks
- [ ] Implement `/proc/sys/kernel/hostname`
- [ ] Implement `/proc/sys/net/ipv4/ip_forward`
- [ ] All procfs files generated on-the-fly (no storage)

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

- [ ] Define `__initrd_start` / `__initrd_end` linker symbols
- [ ] Detect gzip compression (magic bytes 0x1f 0x8b)
- [ ] Implement gzip decompression (or use static decompression library)
- [ ] Implement CPIO newc format parser
- [ ] Handle directories, regular files, symlinks, device nodes
- [ ] Mount initrd as root filesystem if present
- [ ] Run `/init` from initrd
- [ ] Implement `pivot_root` for switching to real root
- [ ] If no initrd, mount ManifoldFS directly as root

### 7. Mount Table

```rust
static MOUNT_TABLE: RwLock<Vec<VfsMount>>;
```

- [ ] Implement `mount(source, target, fstype, flags, data)` syscall
- [ ] Implement `umount2(target, flags)` syscall
- [ ] Support `MS_RDONLY`, `MS_NOSUID`, `MS_NODEV`, `MS_NOEXEC`, `MS_SYNCHRONOUS`
- [ ] Support `MS_REMOUNT` for changing flags
- [ ] Support `MS_BIND` for bind mounts (v2)
- [ ] Support `MS_MOVE` for moving mount points (v2)
- [ ] Verify mount target exists and is a directory
- [ ] Update `MOUNT_TABLE` on mount/umount
- [ ] Handle mount propagation (shared/subtree mounts v2)

---

## Verification

| Test | What it proves | Status |
|---|---|---|
| `test_vfs_lookup` | `lookup_path("/dev/null")` returns correct inode with mode S_IFCHR | [ ] |
| `test_vfs_cross_mount` | `lookup_path("/mnt/manifold/file.txt")` crosses mount boundary correctly | [ ] |
| `test_devtmpfs_read_null` | `read("/dev/null", buf, 1024)` returns 0 | [ ] |
| `test_devtmpfs_write_null` | `write("/dev/null", buf, 1024)` returns 1024, data discarded | [ ] |
| `test_devtmpfs_read_zero` | `read("/dev/zero", buf, 1024)` returns 1024, all zeros | [ ] |
| `test_devtmpfs_random_entropy` | `read("/dev/random", buf, 8)` returns 8, bytes vary across calls | [ ] |
| `test_tmpfs_create_write_read` | Create file in /tmp, write, read back, data matches | [ ] |
| `test_tmpfs_enospc` | Fill /tmp to max_pages, next write returns -ENOSPC | [ ] |
| `test_proc_uptime` | `read("/proc/uptime")` returns positive number, increases across reads | [ ] |
| `test_proc_self_pid` | `readlink("/proc/self")` returns current PID | [ ] |
| `test_initrd_load` | CPIO archive loaded, /init exists and is executable | [ ] |
| `test_mount_tmpfs` | `mount("tmpfs", "/mnt", "tmpfs", 0, None)` succeeds, files visible | [ ] |
| `test_umount` | `umount("/mnt")` succeeds, files no longer visible | [ ] |

---

*Design document produced by Phase 3 Planning*
*2026-05-19*
