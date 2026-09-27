#!/usr/bin/env bash
# First-milestone boot checks for the Linux-parity revamp question.
#
# Each case boots the Seal OS kernel under QEMU (UEFI, q35, same flags as
# kernel/seal-os/run-qemu.ps1) and checks ONE property on the serial log.
#
#   ring3-seal   /bin/init is a static ELF speaking the Seal ABI (write=1, exit=0).
#                Property: its marker, printed from ring 3, reaches serial.
#   ring3-linux  /bin/init is an unmodified static Linux ELF, no libc (write=1, exit=60).
#                Property: its marker, printed from ring 3, reaches serial.
#   ring3-glibc  /bin/init is `gcc -static` hello world against glibc (13 distinct syscalls).
#                Property: its marker, printed from ring 3, reaches serial.
#   ahci-root    control: stock seal-os.img on AHCI (the run-qemu.ps1 layout).
#                Property: the root filesystem mounts from disk.
#   virtio-root  the same image on virtio-blk-pci, the only disk a KVM/cloud guest gets.
#                Property: the root filesystem mounts from disk.
#   ext4-root    AHCI port 0 carries an ext4 filesystem (mke2fs -t ext4 defaults: extents,
#                64bit, flex_bg, metadata_csum, journal), i.e. a stock distro root.
#                Property: the ext2 driver refuses it (a driver that ignores INCOMPAT
#                features must not mount, let alone write, a filesystem it cannot parse).
#
# For ring3-* the root filesystem is an ext2 image made by Linux mke2fs with the
# fixture at /bin/init, on AHCI port 0; seal-os.img boots from port 1. Seal's
# init_vfs falls back to ext2 on dev 0x800 when port 0 carries no ManifoldFS
# superblock, and lib.rs Stage 5 execs /bin/init from it.
#
# Run from WSL (needs gcc and mke2fs there; QEMU is the Windows build via interop):
#   bash tests/linux_parity/cameron_qemu_milestone.sh <case>
# Prereq: kernel/seal-os/target/x86_64-unknown-uefi/release/seal-os.img built per
# kernel/seal-os/TESTING.md (cargo +nightly build --release; seal-mkimage).
# Exit: 0 property holds, 1 property fails, 2 setup error (no verdict).
set -u
case_name=${1:?usage: $0 ring3-seal|ring3-linux|ring3-glibc|ahci-root|virtio-root|ext4-root}
repo=$(cd "$(dirname "$0")/../.." && pwd)
rel=$repo/kernel/seal-os/target/x86_64-unknown-uefi/release
work=$rel/linux-parity/$case_name
limit=${CAMERON_BOOT_SECONDS:-300}
mem=${CAMERON_MEM_MB:-4096}   # run-qemu.ps1 uses 4096; lower it when the host is short of RAM
qemu=${QEMU:-"/mnt/c/Program Files/qemu/qemu-system-x86_64.exe"}
ovmf=${OVMF:-"/mnt/c/Program Files/qemu/share/edk2-x86_64-code.fd"}

setup_fail() { echo "SETUP ERROR ($case_name): $*"; exit 2; }
win() { wslpath -w "$1"; }
[ -f "$rel/seal-os.img" ] || setup_fail "missing $rel/seal-os.img"
[ -f "$qemu" ] || setup_fail "no QEMU at $qemu"
[ -f "$ovmf" ] || setup_fail "no OVMF at $ovmf"
rm -rf "$work" && mkdir -p "$work/root/bin"
log=$work/serial.log

# $1 = exit syscall number, $2 = marker. Loops after exit so an unimplemented
# exit cannot fault; the marker is written before exit either way.
raw_init() {
  cat > "$work/init.S" <<EOF
.globl _start
_start:
  mov \$1, %eax
  mov \$1, %edi
  lea msg(%rip), %rsi
  mov \$len, %edx
  syscall
  mov \$$1, %eax
  xor %edi, %edi
  syscall
1: jmp 1b
msg: .ascii "$2\n"
len = . - msg
EOF
  gcc -nostdlib -static -o "$work/root/bin/init" "$work/init.S" || setup_fail "gcc raw init"
}

disks=()
fstype=ext2
case $case_name in
  ring3-seal)  marker=SEAL-ABI-RING3-OK; raw_init 0 "$marker" ;;
  ext4-root)   marker=SEAL-ABI-RING3-OK; raw_init 0 "$marker"; fstype=ext4 ;;
  ring3-linux) marker=LINUX-ABI-RING3-OK; raw_init 60 "$marker" ;;
  ring3-glibc)
    marker=LINUX-GLIBC-RING3-OK
    printf '#include <stdio.h>\nint main(void){puts("%s");return 0;}\n' "$marker" > "$work/hello.c"
    gcc -static -O2 -o "$work/root/bin/init" "$work/hello.c" || setup_fail "gcc glibc init" ;;
  ahci-root)   disks=(-device ahci,id=seal_sata
                      -drive "if=none,id=boot,file=$(win "$rel/seal-os.img"),format=raw,snapshot=on"
                      -device ide-hd,drive=boot,bus=seal_sata.0,unit=0) ;;
  virtio-root) disks=(-drive "if=none,id=boot,file=$(win "$rel/seal-os.img"),format=raw,snapshot=on"
                      -device virtio-blk-pci,drive=boot) ;;
  *) setup_fail "unknown case" ;;
esac

if [[ $case_name == ring3-* || $case_name == ext4-root ]]; then
  chmod 755 "$work/root/bin/init"
  # Control: the fixture is a working program on Linux (the Seal-ABI one is not run here:
  # on Linux its exit number 0 is read(2)).
  if [[ $case_name == ring3-linux || $case_name == ring3-glibc ]]; then
    out=$("$work/root/bin/init") || setup_fail "fixture failed on Linux"
    [[ $out == "$marker" ]] || setup_fail "fixture printed '$out' on Linux"
    echo "control: fixture prints '$out' on Linux $(uname -r)"
  fi
  truncate -s 80M "$work/root.img"
  mke2fs -q -F -t "$fstype" -b 1024 -d "$work/root" "$work/root.img" || setup_fail "mke2fs"
  echo "rootfs: $fstype, features: $(dumpe2fs -h "$work/root.img" 2>/dev/null | sed -n 's/^Filesystem features: *//p')"
  disks=(-device ahci,id=seal_sata
         -drive "if=none,id=rootfs,file=$(win "$work/root.img"),format=raw"
         -device ide-hd,drive=rootfs,bus=seal_sata.0,unit=0
         -drive "if=none,id=boot,file=$(win "$rel/seal-os.img"),format=raw,snapshot=on"
         -device ide-hd,drive=boot,bus=seal_sata.1,unit=0)
fi

"$qemu" -machine q35 -m "$mem" -cpu qemu64 -smp 2 \
  -drive "if=pflash,format=raw,readonly=on,file=$(win "$ovmf")" \
  "${disks[@]}" \
  -serial "file:$(win "$log")" -pidfile "$(win "$work/qemu.pid")" \
  -no-reboot -display none -device VGA &
proxy=$!

has() { grep -qF -- "$1" "$log" 2>/dev/null; }
start=$(date +%s); settled=0
while (( $(date +%s) - start < limit )); do
  sleep 3
  case $case_name in
    ring3-*) has "$marker" && break ;;
    ext4-root) has "[VFS]" && sleep 20 && break ;;   # the verdict is the mount line; 20 s to log what follows
    *) has "[BOOT] All layers initialized." && break ;;
  esac
  # Stage 5 runs just before the desktop; give a spawned init 30 s past the event loop.
  if has "[EVENT] Entering real event loop"; then
    (( settled == 0 )) && settled=$(date +%s)
    (( $(date +%s) - settled > 30 )) && break
  fi
  kill -0 "$proxy" 2>/dev/null || break
done
[ -f "$work/qemu.pid" ] && taskkill.exe /F /PID "$(tr -dc 0-9 < "$work/qemu.pid")" >/dev/null 2>&1
kill "$proxy" 2>/dev/null; wait "$proxy" 2>/dev/null
elapsed=$(( $(date +%s) - start ))

[ -s "$log" ] || setup_fail "empty serial log after ${elapsed}s"
echo "serial: $log (${elapsed}s)"
grep -aE '^\[(VFS|AHCI|execve|userspace)\]|RING3-OK|\[BOOT\] All layers|\[EVENT\] Entering|PANIC|\[FAULT\]' "$log" | head -40

if [[ $case_name == ring3-* ]]; then
  has "[VFS] Ext2 mounted from AHCI port 0" || setup_fail "Linux-made ext2 root did not mount"
  has "[execve] Loading '/bin/init'" || setup_fail "/bin/init not read from the ext2 root"
  if has "$marker"; then echo "GREEN: ring-3 /bin/init printed $marker"; exit 0; fi
  echo "RED: /bin/init was loaded from disk but its ring-3 marker $marker never reached serial in ${elapsed}s"
  exit 1
fi
if [[ $case_name == ext4-root ]]; then
  has "[VFS]" || setup_fail "kernel never reached VFS init"
  if has "[VFS] Ext2 mounted from AHCI port 0"; then
    echo "RED: the ext2 driver mounted an ext4 root as ext2; next: $(grep -aE '^\[(SWAP|execve|BOOT\] All)' "$log" | tr '\n' ' ')"
    exit 1
  fi
  echo "GREEN: the ext4 root was refused: $(grep -aF '[VFS]' "$log" | tr '\n' ' ')"; exit 0
fi
has "[BOOT] All layers initialized." || setup_fail "kernel did not finish init (did firmware boot the disk?)"
if has "[VFS] ManifoldFS mounted from disk"; then echo "GREEN: root filesystem mounted from the $case_name disk"; exit 0; fi
echo "RED: kernel booted from the $case_name disk but mounted no root filesystem from it: $(grep -aF '[VFS]' "$log" | tr '\n' ' ')"
exit 1
