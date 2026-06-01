// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Seal-native signal delivery and handling.
//!
//! Signals are represented as a 64-bit pending bitmap per task.  Delivery
//! happens asynchronously: the sender sets a bit; the handler runs on the
//! next return to userspace (before `iretq`).  Signal frames are built on
//! the user stack so that `sigreturn(2)` can restore the interrupted context.

use core::arch::global_asm;
use core::sync::atomic::Ordering;

use crate::cpu::CPU_COUNT;
use crate::process::task::RestartSyscall;
use crate::process::task::Task;
use crate::process::task::TaskState;
use crate::process::userspace::{SyscallFrame, UserContext};

// ---------------------------------------------------------------------------
// Signal numbers
// ---------------------------------------------------------------------------

pub const SIGHUP: u8 = 1;
pub const SIGINT: u8 = 2; // Ctrl+C
pub const SIGQUIT: u8 = 3;
pub const SIGILL: u8 = 4;
pub const SIGTRAP: u8 = 5;
pub const SIGABRT: u8 = 6;
pub const SIGBUS: u8 = 7;
pub const SIGFPE: u8 = 8;
pub const SIGKILL: u8 = 9; // unblockable
pub const SIGUSR1: u8 = 10;
pub const SIGSEGV: u8 = 11; // segfault
pub const SIGUSR2: u8 = 12;
pub const SIGPIPE: u8 = 13;
pub const SIGALRM: u8 = 14;
pub const SIGTERM: u8 = 15; // polite kill
pub const SIGCHLD: u8 = 17;
pub const SIGCONT: u8 = 18;
pub const SIGSTOP: u8 = 19;
pub const SIGTSTP: u8 = 20;
pub const SIGTTIN: u8 = 21;
pub const SIGTTOU: u8 = 22;

pub const SA_ONSTACK: u64 = 1 << 0;
pub const SA_RESTART: u64 = 1 << 1;

pub const SS_ONSTACK: u64 = 1 << 0;
pub const SS_DISABLE: u64 = 1 << 1;

const MINSIGSTKSZ: u64 = 2048;
const EINTR: i64 = 4;

// ---------------------------------------------------------------------------
// Sigreturn trampoline
// ---------------------------------------------------------------------------

global_asm!(
    ".global sigreturn_trampoline",
    ".p2align 4",
    "sigreturn_trampoline:",
    "mov rax, 27", // SYS_SIGRETURN — must match syscall/signal.rs
    "syscall",
    "ud2",
);

extern "C" {
    fn sigreturn_trampoline();
}

// ---------------------------------------------------------------------------
// Signal delivery
// ---------------------------------------------------------------------------

/// Send signal `sig` to task `target_id`.
///
/// SIGKILL and SIGSTOP are handled immediately (mark target dead).
/// All other signals are added to the target's pending bitmap.
/// If the target is running on another CPU, a reschedule IPI is sent.
pub fn send_signal(target_id: u64, sig: u8) -> bool {
    if sig == 0 || sig >= 32 {
        return false;
    }
    let cpu_count = CPU_COUNT.load(Ordering::SeqCst);
    for i in 0..cpu_count {
        unsafe {
            if let Some(per_cpu) = crate::cpu::per_cpu(i as u32) {
                let _guard = per_cpu.scheduler_lock.lock();
                if let Some(idx) = per_cpu.scheduler.find_task_by_id(target_id) {
                    if let Some(task) = per_cpu.scheduler.task_mut(idx) {
                        if sig == SIGKILL || sig == SIGSTOP {
                            task.state = TaskState::Dead;
                        } else {
                            task.pending_signals |= 1u64 << sig;
                        }
                        let is_running = per_cpu.scheduler.task_is_current(idx);
                        drop(_guard);
                        if is_running && i != crate::cpu::current_cpu_num() as usize {
                            crate::drivers::apic::local_apic().send_ipi(
                                per_cpu.apic_id,
                                0xFE, // VECTOR_IPI_RESCHEDULE
                            );
                        }
                        return true;
                    }
                }
            }
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Signal handling on return to userspace
// ---------------------------------------------------------------------------

/// Check whether the current task has pending, unmasked signals and, if so,
/// build a signal frame on the user stack.
///
/// Must be called with interrupts disabled and with a valid `UserContext`
/// describing the userspace CPU state.
pub fn check_and_handle_signals(ctx: &mut UserContext) {
    let task_ptr = unsafe { crate::cpu::this_cpu().current_task };
    if task_ptr.is_null() {
        return;
    }
    let task = unsafe { &mut *task_ptr };

    let pending = task.pending_signals & !task.signal_mask;
    if pending == 0 {
        return;
    }

    let sig = pending.trailing_zeros() as u8;
    if sig == 0 || sig >= 32 {
        return;
    }
    task.pending_signals &= !(1u64 << sig);

    let handler = task.signal_handlers[sig as usize];

    if handler == 0 {
        // Default action: terminate
        task.state = TaskState::Dead;
        return;
    }
    if handler == 1 {
        // Ignore
        return;
    }

    // Save original context for sigreturn
    task.signal_saved_context = Some(*ctx);

    // Align stack to 16 bytes (SysV AMD64 ABI)
    let mut rsp = signal_stack_pointer(task, sig, ctx.rsp) & !0xF;

    // Push UserContext onto user stack
    let uc_size = core::mem::size_of::<UserContext>() as u64;
    rsp = rsp.saturating_sub(uc_size);
    let uc_bytes = unsafe {
        core::slice::from_raw_parts(ctx as *const UserContext as *const u8, uc_size as usize)
    };
    unsafe {
        if crate::security::smap_smep::copy_to_user(rsp as *mut u8, uc_bytes).is_err() {
            crate::serial_println!("[signal] failed to push UserContext for sig {}", sig);
            task.state = TaskState::Dead;
            return;
        }
    }

    // Push signal number
    rsp = rsp.saturating_sub(8);
    let sig_num = sig as u64;
    unsafe {
        if crate::security::smap_smep::copy_to_user(rsp as *mut u8, &sig_num.to_le_bytes()).is_err()
        {
            crate::serial_println!("[signal] failed to push signum for sig {}", sig);
            task.state = TaskState::Dead;
            return;
        }
    }

    // Push sigreturn trampoline address
    rsp = rsp.saturating_sub(8);
    let tramp = sigreturn_trampoline as *const () as u64;
    unsafe {
        if crate::security::smap_smep::copy_to_user(rsp as *mut u8, &tramp.to_le_bytes()).is_err() {
            crate::serial_println!("[signal] failed to push trampoline for sig {}", sig);
            task.state = TaskState::Dead;
            return;
        }
    }

    ctx.rsp = rsp;
    ctx.rip = handler;
}

fn signal_stack_pointer(task: &mut Task, sig: u8, current_rsp: u64) -> u64 {
    let flags = task.signal_flags[sig as usize];
    let enabled = task.signal_alt_stack_sp != 0
        && task.signal_alt_stack_size >= MINSIGSTKSZ
        && (task.signal_alt_stack_flags & SS_DISABLE) == 0;
    if enabled && (flags & SA_ONSTACK) != 0 {
        task.signal_alt_stack_flags |= SS_ONSTACK;
        task.signal_alt_stack_sp
            .saturating_add(task.signal_alt_stack_size)
    } else {
        current_rsp
    }
}

// ---------------------------------------------------------------------------
// Syscall helpers (exposed for master to wire)
// ---------------------------------------------------------------------------

/// `kill(2)` — send a signal to a task.
pub fn sys_kill(target_id: u64, sig: u8) -> i64 {
    if send_signal(target_id, sig) {
        0
    } else {
        -3 // ESRCH
    }
}

/// `sigaction(2)` — examine and change signal action.
///
/// `act_ptr` points to `{ handler: u64, flags: u64 }` in userspace. For
/// compatibility with the older ABI, the first 8 bytes remain just the handler.
pub fn sys_sigaction(sig: u8, act_ptr: u64, oldact_ptr: u64) -> i64 {
    if sig == 0 || sig >= 32 {
        return -22; // EINVAL
    }
    let task_ptr = unsafe { crate::cpu::this_cpu().current_task };
    if task_ptr.is_null() {
        return -3; // ESRCH
    }
    let task = unsafe { &mut *task_ptr };
    let idx = sig as usize;
    let old = task.signal_handlers[idx];
    let old_flags = task.signal_flags[idx];

    if oldact_ptr != 0 {
        let mut old_action = [0u8; 16];
        old_action[..8].copy_from_slice(&old.to_le_bytes());
        old_action[8..].copy_from_slice(&old_flags.to_le_bytes());
        unsafe {
            if crate::security::smap_smep::copy_to_user(oldact_ptr as *mut u8, &old_action).is_err()
            {
                return -14; // EFAULT
            }
        }
    }

    if act_ptr != 0 {
        let mut new_bytes = [0u8; 8];
        unsafe {
            if crate::security::smap_smep::copy_from_user(&mut new_bytes, act_ptr as *const u8)
                .is_err()
            {
                return -14; // EFAULT
            }
        }
        let new_handler = u64::from_le_bytes(new_bytes);
        task.signal_handlers[idx] = new_handler;

        let mut flags_bytes = [0u8; 8];
        unsafe {
            if crate::security::smap_smep::copy_from_user(
                &mut flags_bytes,
                (act_ptr + 8) as *const u8,
            )
            .is_ok()
            {
                task.signal_flags[idx] = u64::from_le_bytes(flags_bytes);
            }
        }
    }

    0
}

/// `sigaltstack(2)` — install or inspect the alternate signal stack.
///
/// The user structure is `{ sp: u64, flags: u64, size: u64 }`.
pub fn sys_sigaltstack(new_stack_ptr: u64, old_stack_ptr: u64) -> i64 {
    let task_ptr = unsafe { crate::cpu::this_cpu().current_task };
    if task_ptr.is_null() {
        return -3; // ESRCH
    }
    let task = unsafe { &mut *task_ptr };

    if old_stack_ptr != 0 {
        let mut old_stack = [0u8; 24];
        old_stack[..8].copy_from_slice(&task.signal_alt_stack_sp.to_le_bytes());
        old_stack[8..16].copy_from_slice(&task.signal_alt_stack_flags.to_le_bytes());
        old_stack[16..].copy_from_slice(&task.signal_alt_stack_size.to_le_bytes());
        unsafe {
            if crate::security::smap_smep::copy_to_user(old_stack_ptr as *mut u8, &old_stack)
                .is_err()
            {
                return -14; // EFAULT
            }
        }
    }

    if new_stack_ptr != 0 {
        let mut new_stack = [0u8; 24];
        unsafe {
            if crate::security::smap_smep::copy_from_user(
                &mut new_stack,
                new_stack_ptr as *const u8,
            )
            .is_err()
            {
                return -14; // EFAULT
            }
        }
        let sp = u64::from_le_bytes([
            new_stack[0],
            new_stack[1],
            new_stack[2],
            new_stack[3],
            new_stack[4],
            new_stack[5],
            new_stack[6],
            new_stack[7],
        ]);
        let flags = u64::from_le_bytes([
            new_stack[8],
            new_stack[9],
            new_stack[10],
            new_stack[11],
            new_stack[12],
            new_stack[13],
            new_stack[14],
            new_stack[15],
        ]);
        let size = u64::from_le_bytes([
            new_stack[16],
            new_stack[17],
            new_stack[18],
            new_stack[19],
            new_stack[20],
            new_stack[21],
            new_stack[22],
            new_stack[23],
        ]);

        if (task.signal_alt_stack_flags & SS_ONSTACK) != 0 {
            return -1; // EPERM
        }
        if (flags & !SS_DISABLE) != 0 {
            return -22; // EINVAL
        }
        if (flags & SS_DISABLE) == 0 && size < MINSIGSTKSZ {
            return -12; // ENOMEM
        }

        task.signal_alt_stack_sp = sp;
        task.signal_alt_stack_size = size;
        task.signal_alt_stack_flags = flags;
    }

    0
}

fn is_restartable_syscall(number: u64) -> bool {
    matches!(
        number,
        crate::syscall::table::SYS_READ
            | crate::syscall::table::SYS_WAITPID
            | crate::syscall::table::SYS_NANOSLEEP
            | crate::syscall::table::SYS_SLEEP
    )
}

pub fn prepare_syscall_restart(frame: &mut SyscallFrame, number: u64, result: i64) -> bool {
    if result != -EINTR || !is_restartable_syscall(number) {
        return false;
    }

    let task_ptr = unsafe { crate::cpu::this_cpu().current_task };
    if task_ptr.is_null() {
        return false;
    }
    let task = unsafe { &mut *task_ptr };
    let pending = task.pending_signals & !task.signal_mask;
    if pending == 0 {
        return false;
    }
    let sig = pending.trailing_zeros() as usize;
    if sig >= task.signal_flags.len() || (task.signal_flags[sig] & SA_RESTART) == 0 {
        return false;
    }

    let restart_rip = frame.rcx.saturating_sub(2);
    task.restart_syscall = Some(RestartSyscall {
        number,
        arg0: frame.rdi,
        arg1: frame.rsi,
        arg2: frame.rdx,
        rip: restart_rip,
    });
    frame.rax = number;
    frame.rcx = restart_rip;
    true
}

/// `sigreturn(2)` — restore context saved by signal delivery.
///
/// This syscall never returns normally.
pub fn sys_sigreturn() -> ! {
    let task_ptr = unsafe { crate::cpu::this_cpu().current_task };
    if task_ptr.is_null() {
        crate::process::scheduler::yield_current();
        loop {
            x86_64::instructions::hlt();
        }
    }
    let task = unsafe { &mut *task_ptr };
    if let Some(saved) = task.signal_saved_context.take() {
        let mut saved = saved;
        task.signal_alt_stack_flags &= !SS_ONSTACK;
        task.restart_syscall = None;
        unsafe { crate::process::userspace::enter_userspace(&mut saved) }
    } else {
        crate::serial_println!("[signal] sys_sigreturn: no saved context");
        task.state = TaskState::Dead;
        crate::process::scheduler::yield_current();
        loop {
            x86_64::instructions::hlt();
        }
    }
}

/// Wrapper exposed for syscall dispatch wiring.
pub fn sys_sigreturn_call() -> ! {
    sys_sigreturn()
}
