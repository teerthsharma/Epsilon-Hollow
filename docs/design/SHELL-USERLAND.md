# Shell, Userspace, and libc — Design Document

> **Phase 3 Task**: USER-001
> **Priority**: HIGH (required for interactive use, running binaries, package manager)
> **Estimated Effort**: 5 weeks
> **Blocked by**: Syscall dispatcher, VFS, devtmpfs, initrd
> **Blocks**: Package manager, development workflow, SSH server

---

## Overview

The kernel is useless without userspace. This design covers the init process, a POSIX shell, core utilities, and a minimal libc that allows compiling standard C programs against Seal OS.

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

- [ ] Implement init process as `/bin/init`
- [ ] Mount procfs, sysfs, devtmpfs, tmpfs
- [ ] Open `/dev/console` as stdin/stdout/stderr
- [ ] Parse `/etc/inittab` for runlevels
- [ ] Spawn getty on `/dev/tty1`
- [ ] Reap zombie children in loop
- [ ] Handle SIGCHLD for immediate reaping
- [ ] Implement `reboot()` and `poweroff()` commands
- [ ] Implement emergency shell fallback if init fails

### 2. POSIX Shell (`/bin/sh`)

Minimal POSIX-compliant shell. Supports:

- [ ] Command execution (`ls -la`)
- [ ] Argument passing (`echo hello world`)
- [ ] Environment variables (`FOO=bar echo $FOO`)
- [ ] Variable expansion (`$HOME`, `${VAR}`)
- [ ] Quoting (single quotes, double quotes, backslash)
- [ ] Pipes (`cmd1 | cmd2`)
- [ ] Redirection (`>`, `>>`, `<`, `2>`)
- [ ] Here-documents (`<<EOF`)
- [ ] Background jobs (`cmd &`)
- [ ] Job control (`fg`, `bg`, `jobs`)
- [ ] Subshells (`(cmd1; cmd2)`)
- [ ] Conditionals (`if`, `then`, `else`, `fi`)
- [ ] Loops (`for`, `while`, `until`)
- [ ] Functions (`foo() { ... }`)
- [ ] Signal handling (`trap`)
- [ ] Builtins: `cd`, `pwd`, `echo`, `exit`, `export`, `unset`, `alias`, `source`, `umask`
- [ ] Execute external binaries via `execve()`
- [ ] Search `PATH` for commands
- [ ] Parse shebang (`#!/bin/sh`) for scripts

### 3. Core Utilities

Essential Unix utilities, implemented in Rust or C:

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

- [ ] Implement P0 utilities in Rust
- [ ] Compile P0 utilities as static binaries
- [ ] Implement `mv` using ManifoldFS teleport (O(1) when same filesystem)
- [ ] Implement `ps` reading `/proc`
- [ ] Implement `df` reading mount table + filesystem stats
- [ ] Implement `free` reading `/proc/meminfo`
- [ ] Implement `kill` via `kill()` syscall
- [ ] Implement `mount` / `umount` via syscalls

### 4. libc — Minimal C Library

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

- [ ] Implement libc in C with syscall wrappers
- [ ] Implement `malloc`/`free` using `brk()`/`mmap()`
- [ ] Implement `fopen`/`fread`/`fwrite` with buffering (4 KiB)
- [ ] Implement `printf` family with `%d`, `%s`, `%x`, `%p`, `%f`, `%lld`
- [ ] Implement string functions (`memcpy`, `strcmp`, etc.)
- [ ] Implement `fork()` / `execve()` / `wait()`
- [ ] Implement `pthread_*` on top of `clone()`
- [ ] Implement `errno` as TLS variable
- [ ] Compile libc as static archive (`libc.a`)
- [ ] Provide `libc.so` for dynamic linking (v2)
- [ ] Implement `crt0.o` startup code (set up argc/argv/envp, call main)
- [ ] Implement `crti.o` / `crtn.o` for C++ constructors (v2)

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

- [ ] Implement ELF header parsing (`Elf64_Ehdr`)
- [ ] Implement program header parsing (`Elf64_Phdr`)
- [ ] Implement `PT_LOAD` segment mapping via `mmap()`
- [ ] Handle BSS zeroing (`.bss` section)
- [ ] Handle `PT_INTERP` — dynamic linker support (v2)
- [ ] Handle `PT_GNU_STACK` — NX bit
- [ ] Set up userspace stack with argv, envp, auxv
- [ ] Set `rip = e_entry`, `rsp = stack_top`
- [ ] Support position-independent executables (PIE)
- [ ] Support shared libraries / dynamic linking (v2)

### 6. Dynamic Linker (ld.so) — v2

- [ ] Implement `ld.so` as userspace program
- [ ] Parse `PT_DYNAMIC` segment
- [ ] Load required shared libraries from `/lib` and `/usr/lib`
- [ ] Resolve symbols via relocation tables (REL, RELA, PLT)
- [ ] Implement lazy binding via PLT stubs
- [ ] Handle `LD_LIBRARY_PATH`
- [ ] Handle `LD_PRELOAD`

### 7. Cross-Compilation Toolchain

- [ ] Target triple: `x86_64-seal-linux-gnu`
- [ ] GCC/binutils configured for Seal OS
- [ ] Newlib or musl as base C library
- [ ] Rust target specification JSON for `x86_64-seal-os`
- [ ] `rustc` cross-compilation support
- [ ] `cargo` cross-compilation with `.cargo/config.toml`

---

## Verification

| Test | What it proves | Status |
|---|---|---|
| `test_init_runs` | `/bin/init` executes, mounts filesystems, spawns shell | [ ] |
| `test_shell_echo` | `echo hello` prints "hello" | [ ] |
| `test_shell_pipe` | `cat /proc/uptime | wc -l` returns 1 | [ ] |
| `test_shell_redirect` | `echo test > /tmp/file.txt`, content matches | [ ] |
| `test_shell_background` | `sleep 10 &` returns immediately, job in `jobs` list | [ ] |
| `test_ls_dir` | `ls /bin` lists installed utilities | [ ] |
| `test_mv_teleport` | `mv large_file /tmp/` completes in < 1 ms (O(1) teleport) | [ ] |
| `test_ps_processes` | `ps` shows init, shell, and test processes | [ ] |
| `test_fork_exec` | C program with `fork()` + `execve("/bin/echo")` works | [ ] |
| `test_malloc_free` | C program allocates 1 MiB, frees, no leak | [ ] |
| `test_pthread_create` | C program creates 4 threads, all run and join | [ ] |
| `test_elf_loader` | Run a simple `int main() { return 42; }`, exit status 42 | [ ] |
| `test_printf_format` | `printf("%d %s %x", 42, "hello", 255)` outputs correct string | [ ] |
| `test_dynamic_link` | Run program linked against `libc.so` (v2) | [ ] |

---

*Design document produced by Phase 3 Planning*
*2026-05-19*
