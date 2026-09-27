#!/usr/bin/env bash
# Measure, on a real Linux distro (run inside WSL or any x86_64 Linux), the two
# facts the "replace the distro's kernel" proposal depends on:
#   1. which Linux syscall numbers unmodified distro binaries actually issue
#      (ptrace tracer, every thread and child, counted at syscall entry);
#   2. what the distro's own kernel modules (drivers) import and are licensed as.
# Writes chase_linux_measured.json next to this script. Stores only facts
# (syscall name/number pairs, symbol names, license tags, counts) -- no Linux
# source or header text is copied.
#
#   wsl -e bash tests/linux_parity/chase_measure_linux.sh
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

cat >"$work/trace.c" <<'EOF'
#define _GNU_SOURCE
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <sys/ptrace.h>
#include <sys/wait.h>
#include <unistd.h>
/* Count syscall entries (by number) of a command, its threads and children. */
static unsigned long counts[1024];
int main(int argc, char **argv) {
    if (argc < 3) return 2; /* trace OUTFILE cmd args... */
    pid_t child = fork();
    if (child == 0) {
        ptrace(PTRACE_TRACEME, 0, 0, 0);
        raise(SIGSTOP);
        execvp(argv[2], argv + 2);
        _exit(127);
    }
    int st;
    waitpid(child, &st, 0);
    ptrace(PTRACE_SETOPTIONS, child, 0,
           PTRACE_O_TRACESYSGOOD | PTRACE_O_TRACEFORK | PTRACE_O_TRACEVFORK |
           PTRACE_O_TRACECLONE | PTRACE_O_TRACEEXEC | PTRACE_O_EXITKILL);
    ptrace(PTRACE_SYSCALL, child, 0, 0);
    for (;;) {
        pid_t pid = waitpid(-1, &st, __WALL);
        if (pid < 0) break; /* ECHILD: every tracee is gone */
        if (!WIFSTOPPED(st)) continue;
        int sig = WSTOPSIG(st), inject = 0;
        if (sig == (SIGTRAP | 0x80)) {
            struct __ptrace_syscall_info info; /* glibc name */
            if (ptrace(PTRACE_GET_SYSCALL_INFO, pid, sizeof info, &info) > 0 &&
                info.op == PTRACE_SYSCALL_INFO_ENTRY && info.entry.nr < 1024)
                counts[info.entry.nr]++;
        } else if (sig != SIGTRAP && sig != SIGSTOP) {
            inject = sig; /* a real signal: deliver it */
        }
        ptrace(PTRACE_SYSCALL, pid, 0, inject);
    }
    FILE *out = fopen(argv[1], "w");
    if (!out) return 3;
    for (int i = 0; i < 1024; i++)
        if (counts[i]) fprintf(out, "%d %lu\n", i, counts[i]);
    return fclose(out) != 0;
}
EOF
cat >"$work/hello.c" <<'EOF'
#include <stdio.h>
int main(void) { puts("hello"); return 0; }
EOF
gcc -O2 -o "$work/trace" "$work/trace.c"
gcc -O2 -static -o "$work/hello-static" "$work/hello.c"

trace() { # label, command...
    local label=$1; shift
    "$work/trace" "$work/t.$label" "$@" >/dev/null 2>&1 </dev/null
    echo "$label" >>"$work/labels"
}
: >"$work/labels"
trace static-hello "$work/hello-static"
trace true /usr/bin/true
trace cat /usr/bin/cat /etc/os-release
trace sh-pipeline /bin/sh -c 'ls / | wc -l; x=$(uname -r)'
if [ -x /usr/lib/systemd/systemd ]; then
    trace systemd-version /usr/lib/systemd/systemd --version
fi

grep -E '^#define __NR_[a-z0-9_]+ [0-9]+$' /usr/include/x86_64-linux-gnu/asm/unistd_64.h \
    | awk '{sub("__NR_","",$2); print $3, $2}' >"$work/names"

mod_root=/lib/modules/$(uname -r)/kernel
find "$mod_root" -name '*.ko' | sort >"$work/modules"
: >"$work/mods.tsv"
while read -r ko; do
    rel=${ko#"$mod_root"/}
    lic=$(modinfo -F license "$ko" | head -n1)
    vm=$(modinfo -F vermagic "$ko" | head -n1)
    und=$(nm -u "$ko" 2>/dev/null | awk '{print $2}' | tr '\n' ' ')
    nver=$(readelf -S "$ko" 2>/dev/null | grep -c '__versions' || true)
    printf '%s\t%s\t%s\t%s\t%s\n' "$rel" "$lic" "$vm" "$nver" "$und" >>"$work/mods.tsv"
done <"$work/modules"

WORK=$work OUT="$here/chase_linux_measured.json" python3 - <<'EOF'
import json, os, platform, subprocess, datetime
w = os.environ["WORK"]
names = {}
for line in open(f"{w}/names"):
    nr, name = line.split()
    names[int(nr)] = name
programs = {}
for label in open(f"{w}/labels").read().split():
    counts = {}
    for line in open(f"{w}/t.{label}"):
        nr, n = line.split()
        counts[names.get(int(nr), f"nr{nr}")] = {"nr": int(nr), "count": int(n)}
    programs[label] = counts
mods, vermagic, driver_imports = [], {}, set()
for line in open(f"{w}/mods.tsv"):
    rel, lic, vm, nver, und = line.rstrip("\n").split("\t")
    imports = und.split()
    mods.append({"path": rel, "license": lic, "has_versions": nver != "0",
                 "n_imports": len(imports)})
    vermagic[vm] = vermagic.get(vm, 0) + 1
    if rel.startswith("drivers/"):
        driver_imports.update(imports)
os_release = dict(l.rstrip().split("=", 1) for l in open("/etc/os-release") if "=" in l)
out = {
    "provenance": {
        "script": "tests/linux_parity/chase_measure_linux.sh",
        "measured_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "kernel": platform.release(),
        "distro": os_release.get("PRETTY_NAME", "").strip('"'),
        "gcc": subprocess.run(["gcc", "-dumpfullversion"], capture_output=True, text=True).stdout.strip(),
        "glibc": platform.libc_ver()[1],
    },
    "syscalls_by_program": programs,
    "modules": mods,
    "module_vermagic": vermagic,
    "driver_import_union": sorted(driver_imports),
}
json.dump(out, open(os.environ["OUT"], "w"), indent=0, sort_keys=True)
print(f"programs={len(programs)} modules={len(mods)} -> {os.environ['OUT']}")
EOF
