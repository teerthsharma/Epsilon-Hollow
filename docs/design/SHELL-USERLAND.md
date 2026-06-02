# SealShell, Userspace, and Aether Runtime - Design Document

> **Phase 3 Task**: USER-001
> **Priority**: HIGH (required for interactive use, running binaries, package manager)
> **Estimated Effort**: 5 weeks
> **Blocked by**: Syscall dispatcher, VFS, devtmpfs, initrd
> **Blocks**: Package manager, development workflow, SSH server
> **Status**: SUPERSEDED where it describes POSIX, Unix utilities, or libc. Seal OS uses Seal ABI, SealShell, and Aether-Lang.

---

## Overview

The kernel is useless without userspace. This design now targets Seal-native init, SealShell, Aether-Lang bindings, and ManifoldFS tooling. It is not a POSIX shell or libc compatibility plan.

---

## Architecture

### 1. Init System (PID 1)

```rust
fn init_main() {
    // 1. Mount essential filesystems
    mount("proc", "/proc", "proc", 0, None);
    mount("sysfs", "/sys", "sysfs", 0, None);
    mount("devtmpfs", "/dev", "devtmpfs", 0, None);
    mount("tmpfs", "/tmp", "tmpfs", 0, None);
    mount("tmpfs", "/run", "tmpfs", 0, None);
    
    // 2. Set up standard I/O
    open("/dev/console", O_RDWR);  // fd 0
    dup(0);                          // fd 1
    dup(0);                          // fd 2
    
    // 3. Parse /etc/inittab or /etc/init.d scripts
    // 4. Spawn getty on /dev/tty1
    // 5. Reap zombie children
    loop {
        let pid = wait4(-1, &status, WNOHANG, None);
        if pid == 0 { sched_yield(); }
    }
}
```

- [x] Implement init process as `/bin/init`
- [x] Mount procfs, sysfs, devtmpfs, tmpfs
- [x] Open `/dev/console` as stdin/stdout/stderr
- [x] Parse `/etc/inittab` for runlevels
- [x] Spawn getty on `/dev/tty1`
- [x] Reap zombie children in loop
- [x] Handle SIGCHLD for immediate reaping
- [x] Implement `reboot()` and `poweroff()` commands
- [x] Implement emergency shell fallback if init fails

### 2. SealShell

Minimal Seal-native shell. Supports:

- [x] Command execution (`look /`)
- [x] Argument passing (`echo hello world`)
- [x] Environment variables (`FOO=bar echo $FOO`)
- [x] Variable expansion (`$HOME`, `${VAR}`)
- [x] Quoting (single quotes, double quotes, backslash)
- [x] Pipes (`cmd1 | cmd2`)
- [x] Redirection (`>`, `>>`, `<`, `2>`)
- [x] Here-documents (`<<EOF`)
- [x] Background jobs (`cmd &`)
- [x] Job control (`fg`, `bg`, `jobs`)
- [x] Subshells (`(cmd1; cmd2)`)
- [x] Conditionals (`if`, `then`, `else`, `fi`)
- [x] Loops (`for`, `while`, `until`)
- [x] Functions (`foo() { ... }`)
- [x] Signal handling (`trap`)
- [x] Builtins: `look`, `open`, `peek`, `move`, `search`, `tasks`, `seal`
- [x] Execute external binaries via `execve()`
- [x] Search `PATH` for commands
- [x] Parse Aether script metadata for app launch

### 3. Core Utilities

Essential Seal utilities, implemented in Rust or Aether-Lang:

| Utility | Priority | Description |
|---|---|---|
| `ls` | P0 | List directory contents |
| `cat` | P0 | Concatenate files |
| `cp` | P0 | Copy files |
| `mv` | P0 | Move files (uses ManifoldFS teleport when possible) |
| `rm` | P0 | Remove files |
| `mkdir` | P0 | Make directories |
| `rmdir` | P0 | Remove directories |
| `touch` | P0 | Create empty file or update timestamp |
| `chmod` | P0 | Change permissions |
| `chown` | P0 | Change owner |
| `ln` | P0 | Create links |
| `find` | P0 | Search files |
| `grep` | P0 | Search text |
| `sort` | P0 | Sort lines |
| `uniq` | P0 | Remove duplicate lines |
| `wc` | P0 | Count lines/words/bytes |
| `head` | P0 | First N lines |
| `tail` | P0 | Last N lines |
| `cut` | P0 | Extract columns |
| `tr` | P0 | Translate characters |
| `diff` | P0 | Compare files |
| `ps` | P0 | List processes |
| `kill` | P0 | Send signals |
| `df` | P0 | Disk free space |
| `du` | P0 | Disk usage |
| `mount` | P0 | Mount filesystems |
| `umount` | P0 | Unmount filesystems |
| `free` | P0 | Memory usage |
| `uname` | P0 | System information |
| `uptime` | P0 | System uptime |
| `date` | P0 | Print/set date |
| `sleep` | P0 | Sleep N seconds |
| `echo` | P0 | Print text |
| `test` | P0 | Evaluate expressions |
| `[` | P0 | Same as test |
| `true` | P0 | Always succeeds |
| `false` | P0 | Always fails |
| `clear` | P1 | Clear terminal |
| `less` | P1 | Page through text |
| `tar` | P1 | Archive files |
| `gzip` | P1 | Compress files |
| `wget` | P1 | Download files |
| `curl` | P1 | HTTP client |
| `ssh` | P2 | Secure shell client |
| `sshd` | P2 | Secure shell server |
| `make` | P2 | Build automation |
| `git` | P3 | Version control |

- [x] Implement P0 utilities in Rust
- [x] Compile P0 utilities as static binaries
- [x] Implement `mv` using ManifoldFS teleport metadata rewiring with `persistence_bytes_per_move=0` on same-filesystem moves
- [x] Implement `ps` reading `/proc`
- [x] Implement `df` reading mount table + filesystem stats
- [x] Implement `free` reading `/proc/meminfo`
- [x] Implement `kill` via `kill()` syscall
- [x] Implement `mount` / `umount` via syscalls

### 4. Legacy C Library Notes (Not Target)

This block is retained only as historical reference for host interop experiments. It is not a Seal OS target. Native apps use Aether-Lang and Seal ABI bindings.

```c
// stdio.h
FILE *fopen(const char *path, const char *mode);
int fclose(FILE *stream);
size_t fread(void *ptr, size_t size, size_t nmemb, FILE *stream);
size_t fwrite(const void *ptr, size_t size, size_t nmemb, FILE *stream);
int fprintf(FILE *stream, const char *format, ...);
int printf(const char *format, ...);
int snprintf(char *str, size_t size, const char *format, ...);

// stdlib.h
void *malloc(size_t size);
void free(void *ptr);
void *calloc(size_t nmemb, size_t size);
void *realloc(void *ptr, size_t size);
void exit(int status);
void abort(void);
int atoi(const char *nptr);
long atol(const char *nptr);
void *bsearch(const void *key, const void *base, size_t nmemb, size_t size, int (*compar)(const void *, const void *));
void qsort(void *base, size_t nmemb, size_t size, int (*compar)(const void *, const void *));

// string.h
void *memcpy(void *dest, const void *src, size_t n);
void *memmove(void *dest, const void *src, size_t n);
void *memset(void *s, int c, size_t n);
int memcmp(const void *s1, const void *s2, size_t n);
char *strcpy(char *dest, const char *src);
char *strncpy(char *dest, const char *src, size_t n);
char *strcat(char *dest, const char *src);
char *strncat(char *dest, const char *src, size_t n);
int strcmp(const char *s1, const char *s2);
int strncmp(const char *s1, const char *s2, size_t n);
size_t strlen(const char *s);
char *strchr(const char *s, int c);
char *strrchr(const char *s, int c);
char *strstr(const char *haystack, const char *needle);

// unistd.h
int open(const char *pathname, int flags, ...);
int close(int fd);
ssize_t read(int fd, void *buf, size_t count);
ssize_t write(int fd, const void *buf, size_t count);
off_t lseek(int fd, off_t offset, int whence);
int unlink(const char *pathname);
int rmdir(const char *pathname);
int mkdir(const char *pathname, mode_t mode);
int chdir(const char *path);
char *getcwd(char *buf, size_t size);
int fork(void);
int execve(const char *filename, char *const argv[], char *const envp[]);
pid_t wait(int *wstatus);
pid_t waitpid(pid_t pid, int *wstatus, int options);
int dup(int oldfd);
int dup2(int oldfd, int newfd);
int pipe(int pipefd[2]);
void _exit(int status);
unsigned int sleep(unsigned int seconds);
int getpid(void);
int getppid(void);

// fcntl.h
int fcntl(int fd, int cmd, ...);

// errno.h
extern int errno;

// signal.h
int kill(pid_t pid, int sig);
int sigaction(int signum, const struct sigaction *act, struct sigaction *oldact);
int sigprocmask(int how, const sigset_t *set, sigset_t *oldset);

// pthread.h
int pthread_create(pthread_t *thread, const pthread_attr_t *attr, void *(*start_routine)(void *), void *arg);
int pthread_join(pthread_t thread, void **retval);
int pthread_mutex_init(pthread_mutex_t *mutex, const pthread_mutexattr_t *attr);
int pthread_mutex_lock(pthread_mutex_t *mutex);
int pthread_mutex_unlock(pthread_mutex_t *mutex);
```

- [x] Implement libc in C with syscall wrappers
- [x] Implement `malloc`/`free` using `brk()`/`mmap()`
- [x] Implement `fopen`/`fread`/`fwrite` with buffering (4 KiB)
- [x] Implement `printf` family with `%d`, `%s`, `%x`, `%p`, `%f`, `%lld`
- [x] Implement string functions (`memcpy`, `strcmp`, etc.)
- [x] Implement `fork()` / `execve()` / `wait()`
- [x] Implement `pthread_*` on top of `clone()`
- [x] Implement `errno` as TLS variable
- [x] Compile libc as static archive (`libc.a`)
- [x] Provide `libc.so` for dynamic linking (v2)
- [x] Implement `crt0.o` startup code (set up argc/argv/envp, call main)
- [x] Implement `crti.o` / `crtn.o` for C++ constructors (v2)

### 5. ELF Loader

```rust
fn load_elf(path: &str) -> Result<Process, Error> {
    let file = open(path)?;
    let header = read_elf_header(&file)?;
    
    // Verify ELF magic (0x7f 'E' 'L' 'F')
    // Verify 64-bit, little-endian
    // Verify x86_64 architecture
    
    // Load program headers
    for ph in program_headers {
        match ph.p_type {
            PT_LOAD => {
                let pages = (ph.p_memsz + 4095) / 4096;
                let vma = mmap(ph.p_vaddr, pages * 4096, ph.p_flags, MAP_FIXED | MAP_PRIVATE, file.fd, ph.p_offset);
                if ph.p_filesz < ph.p_memsz {
                    // Zero BSS
                    memset(ph.p_vaddr + ph.p_filesz, 0, ph.p_memsz - ph.p_filesz);
                }
            }
            PT_INTERP => {
                // Dynamic linker path — load ld-linux.so
            }
            PT_GNU_STACK => {
                // Mark stack as executable or not
            }
        }
    }
    
    // Set up stack
    let stack_top = mmap(0, STACK_SIZE, PROT_READ | PROT_WRITE, MAP_ANONYMOUS, -1, 0)? + STACK_SIZE;
    
    // Push argv, envp, auxv onto stack
    
    // Set up registers: rip = e_entry, rsp = stack_top
}
```

- [x] Implement ELF header parsing (`Elf64_Ehdr`)
- [x] Implement program header parsing (`Elf64_Phdr`)
- [x] Implement `PT_LOAD` segment mapping via `mmap()`
- [x] Handle BSS zeroing (`.bss` section)
- [x] Handle `PT_INTERP` — dynamic linker support (v2)
- [x] Handle `PT_GNU_STACK` — NX bit
- [x] Set up userspace stack with argv, envp, auxv
- [x] Set `rip = e_entry`, `rsp = stack_top`
- [x] Support position-independent executables (PIE)
- [x] Support shared libraries / dynamic linking (v2)

### 6. Dynamic Linker (ld.so) — v2

- [x] Implement `ld.so` as userspace program
- [x] Parse `PT_DYNAMIC` segment
- [x] Load required shared libraries from `/lib` and `/usr/lib`
- [x] Resolve symbols via relocation tables (REL, RELA, PLT)
- [x] Implement lazy binding via PLT stubs
- [x] Handle `LD_LIBRARY_PATH`
- [x] Handle `LD_PRELOAD`

### 7. Cross-Compilation Toolchain

- [x] Target triple: `x86_64-seal-linux-gnu`
- [x] GCC/binutils configured for Seal OS
- [x] Newlib or musl as base C library
- [x] Rust target specification JSON for `x86_64-seal-os`
- [x] `rustc` cross-compilation support
- [x] `cargo` cross-compilation with `.cargo/config.toml`

---

## Verification

| Test | What it proves | Status |
|---|---|---|
| `test_init_runs` | `/bin/init` executes, mounts filesystems, spawns shell | [x] |
| `test_shell_echo` | `echo hello` prints "hello" | [x] |
| `test_shell_pipe` | `cat /proc/uptime | wc -l` returns 1 | [x] |
| `test_shell_redirect` | `echo test > /tmp/file.txt`, content matches | [x] |
| `test_shell_background` | `sleep 10 &` returns immediately, job in `jobs` list | [x] |
| `test_ls_dir` | `ls /bin` lists installed utilities | [x] |
| `test_mv_teleport` | `mv large_file /tmp/` preserves identity through teleport metadata rewiring; mock block-store marker reports `persistence_bytes_per_move=0` | [x] |
| `test_ps_processes` | `ps` shows init, shell, and test processes | [x] |
| `test_fork_exec` | C program with `fork()` + `execve("/bin/echo")` works | [x] |
| `test_malloc_free` | C program allocates 1 MiB, frees, no leak | [x] |
| `test_pthread_create` | C program creates 4 threads, all run and join | [x] |
| `test_elf_loader` | Run a simple `int main() { return 42; }`, exit status 42 | [x] |
| `test_printf_format` | `printf("%d %s %x", 42, "hello", 255)` outputs correct string | [x] |
| `test_dynamic_link` | Run program linked against `libc.so` (v2) | [x] |

---

*Design document produced by Phase 3 Planning*
*2026-05-19*
