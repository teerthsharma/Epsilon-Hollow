#!/usr/bin/env bash
# Boot the shipped Seal OS image under QEMU with a Linux-made disk on AHCI
# port 0 (the device Seal mounts as its root, 0x800) and check one property.
#
#   chase_boot.sh usermode       A static Linux x86_64 ELF (gcc, 2 PT_LOAD) at
#                                /bin/init on an ext2 root must execute
#                                user-mode code. Its only syscall is nr 1
#                                write(1, msg, len): same number and same
#                                rdi/rsi/rdx on Linux and on the Seal ABI, so
#                                this measures "does a user task run", not ABI.
#   chase_boot.sh usermode-seal  Control: the kernel's own EMERGENCY_SHELL_ELF
#                                (process/userspace.rs, Seal ABI) as /bin/init
#                                must print its greeting.
#   chase_boot.sh ext4           A disk made by the distro's own mkfs.ext4 must
#                                be refused (unknown INCOMPAT features) and left
#                                byte-identical.
#   chase_boot.sh wx             The kernel's own [SECURITY-FEATURES] self-probe
#                                must report no writable+executable kernel page
#                                (wx_violations=0) before any untrusted binary
#                                is let in.
#
# Exit 0 = property holds, 1 = property violated (RED), 2 = setup failure.
#
# Needs the kernel image built first:
#   (kernel/seal-os)      cargo +nightly build --release
#   (kernel/seal-mkimage) cargo +stable run --release
# Linux-side tools (gcc, mke2fs, e2fsck) run natively, or through WSL with
# LINUX="wsl -e env PATH=/usr/sbin:/usr/bin". Windows (Git Bash) example:
#   MSYS2_ARG_CONV_EXCL="PATH=" LINUX="wsl -e env PATH=/usr/sbin:/usr/bin" \
#   QEMU="/c/Program Files/qemu/qemu-system-x86_64.exe" \
#   OVMF="C:/Program Files/qemu/share/edk2-x86_64-code.fd" \
#   bash tests/linux_parity/chase_boot.sh usermode
set -euo pipefail
mode=${1:?usage: chase_boot.sh usermode|usermode-seal|ext4|wx}
root=$(cd "$(dirname "$0")/../.." && pwd)
rel=$root/kernel/seal-os/target/x86_64-unknown-uefi/release
LINUX=${LINUX:-}
QEMU=${QEMU:-qemu-system-x86_64}
OVMF=${OVMF:-/usr/share/OVMF/OVMF_CODE_4M.fd}
MAX_SECS=${MAX_SECS:-420}
MEM=${MEM:-4096}
SETTLE_SECS=${SETTLE_SECS:-30}

setup_fail() { echo "SETUP: $*"; exit 2; }
[ -f "$rel/seal-os.img" ] || setup_fail "missing $rel/seal-os.img (build kernel + seal-mkimage first)"

work=$rel/chase-$mode
rm -rf "$work"
mkdir -p "$work/root/bin" "$work/root/etc"
cd "$work"
printf 'chase canary\n' >root/etc/chase-canary

case $mode in
usermode)
    cat >init.S <<'EOF'
.globl _start
_start:
    mov $1, %eax
    mov $1, %edi
    lea msg(%rip), %rsi
    mov $len, %edx
    syscall
1:  pause
    jmp 1b
msg: .ascii "CHASE-USERMODE-RAN\n"
len = . - msg
EOF
    $LINUX gcc -nostdlib -static -no-pie -o root/bin/init init.S || setup_fail "gcc"
    marker='CHASE-USERMODE-RAN'
    ;;
usermode-seal)
    bytes=$(sed -n '/EMERGENCY_SHELL_ELF: &\[u8\] = &\[/,/^\];/p' \
        "$root/kernel/seal-os/src/process/userspace.rs" | grep -o '0x[0-9a-f][0-9a-f]' | sed 's/0x/\\x/' | tr -d '\n')
    [ -n "$bytes" ] || setup_fail "EMERGENCY_SHELL_ELF not found in userspace.rs"
    printf '%b' "$bytes" >root/bin/init
    marker='Hello from Seal OS userspace!'
    ;;
ext4 | wx) ;;
*) setup_fail "unknown mode $mode" ;;
esac

if [ "$mode" = ext4 ]; then
    $LINUX mke2fs -q -F -t ext4 -d root rootfs.img 64M || setup_fail "mke2fs"
    $LINUX dumpe2fs -h rootfs.img 2>/dev/null | grep -i 'features' >features.txt || true
    $LINUX e2fsck -fn rootfs.img >fsck.before 2>&1 || setup_fail "fresh ext4 not clean: $(cat fsck.before)"
    sha_before=$(sha256sum rootfs.img | cut -d' ' -f1)
    root_opts=""
else
    $LINUX mke2fs -q -F -t ext2 -O none -I 128 -b 1024 -d root rootfs.img 8M 2>/dev/null || setup_fail "mke2fs"
    root_opts=",snapshot=on"
fi

# Boot disk on port 1; the Linux-made disk on port 0 is what Seal mounts.
"$QEMU" -machine q35 -m "$MEM" -cpu qemu64 -smp 2 \
    -drive if=pflash,format=raw,readonly=on,file="$OVMF" \
    -device ahci,id=sata \
    -drive if=none,id=rootd,file=rootfs.img,format=raw,media=disk$root_opts \
    -device ide-hd,drive=rootd,bus=sata.0 \
    -drive if=none,id=boot,file="$rel/seal-os.img",format=raw,media=disk,snapshot=on \
    -device ide-hd,drive=boot,bus=sata.1 \
    -serial file:serial.log -display none -device VGA -no-reboot &
qpid=$!
t=0; settled=-1
while [ $t -lt "$MAX_SECS" ] && kill -0 $qpid 2>/dev/null; do
    sleep 5; t=$((t + 5))
    if grep -qE 'KERNEL PANIC|\[FAULT\]|\[WATCHDOG\]' serial.log 2>/dev/null; then break; fi
    if [ "$mode" = wx ] && grep -q '^\[SECURITY-FEATURES\] proof' serial.log 2>/dev/null; then break; fi
    if [ $settled -lt 0 ] && grep -q 'Entering real event loop' serial.log 2>/dev/null; then settled=$t; fi
    if [ $settled -ge 0 ] && [ $((t - settled)) -ge "$SETTLE_SECS" ]; then break; fi
done
kill $qpid 2>/dev/null || true
wait $qpid 2>/dev/null || true
sleep 2
echo "ran ${t}s; serial.log: $work/serial.log"
grep -E '^\[VFS\]|^\[execve\]|Hello from Seal|CHASE-USERMODE-RAN|Entering real event loop|KERNEL PANIC|\[FAULT\]' serial.log || true

if [ "$mode" = wx ]; then
    line=$(grep -m1 '^\[SECURITY-FEATURES\] proof' serial.log) || setup_fail "no [SECURITY-FEATURES] proof line"
    wxv=$(printf '%s' "$line" | grep -o 'wx_violations=[0-9]*' | cut -d= -f2)
    scanned=$(printf '%s' "$line" | grep -o 'wx_pages_scanned=[0-9]*' | cut -d= -f2)
    echo "$line"
    if [ "$wxv" = 0 ]; then echo "PASS: no writable+executable kernel page"; exit 0; fi
    echo "FAIL: $wxv of $scanned scanned kernel pages are writable+executable (kernel self-probe)"
    exit 1
fi

if [ "$mode" != ext4 ]; then
    grep -q 'Ext2 mounted from AHCI port 0' serial.log || setup_fail "ext2 root was not mounted from port 0"
    grep -q "\[execve\] Loading '/bin/init'" serial.log || setup_fail "kernel did not find /bin/init"
    if grep -q "$marker" serial.log; then
        echo "PASS: /bin/init executed user-mode code"; exit 0
    fi
    spawned=no; grep -q "Spawned 'init' as task" serial.log && spawned=yes
    desktop=no; grep -q 'Entering real event loop' serial.log && desktop=yes
    echo "FAIL: /bin/init executed zero user-mode instructions in ${t}s (task enqueued: $spawned; boot reached desktop event loop: $desktop)"
    exit 1
fi

grep -q '^\[VFS\]' serial.log || setup_fail "boot never reached VFS init"
sha_after=$(sha256sum rootfs.img | cut -d' ' -f1)
$LINUX e2fsck -fn rootfs.img >fsck.after 2>&1 && fsck_rc=0 || fsck_rc=$?
echo "features: $(cat features.txt)"
echo "sha256 before=$sha_before after=$sha_after; e2fsck -fn after rc=$fsck_rc"
fail=0
if grep -q 'Ext2 mounted from AHCI port 0\|Ext2 attached as ManifoldFS' serial.log; then
    echo "FAIL: Seal mounted an ext4 volume whose INCOMPAT features it does not implement"; fail=1
fi
if [ "$sha_before" != "$sha_after" ]; then
    echo "FAIL: Seal wrote to the foreign ext4 disk"; fail=1
    sed -n 1,20p fsck.after
fi
[ $fail -eq 0 ] && echo "PASS: foreign ext4 refused and untouched"
exit $fail
