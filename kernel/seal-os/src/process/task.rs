// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Task (process) representation with manifold embedding and CPU context.

use alloc::string::String;
use alloc::vec::Vec;

use super::context_switch::{init_task_context, xsave_area_size, TaskContext, KERNEL_STACK_SIZE};
use super::userspace::UserContext;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TaskState {
    Ready,
    Running,
    Blocked,
    Dead,
}

pub struct Task {
    pub id: u64,
    pub name: String,
    pub state: TaskState,
    pub manifold_embedding: [f64; 8],
    pub voronoi_cell: usize,
    pub priority: u8,
    pub ticks_used: u64,
    pub entry: fn(),
    pub is_userspace: bool,
    pub user_stack: Vec<u8>,
    pub page_table: u64,
    pub brk_end: u64,

    // Real multitasking support
    pub context: TaskContext,
    pub xsave_storage: Vec<u8>,
    pub kernel_stack: Vec<u8>,

    // Security identity
    pub uid: u32,
    pub gid: u32,
    pub euid: u32,
    pub egid: u32,
    pub groups: Vec<u32>,

    // Working directory
    pub cwd: String,

    // Signal state
    pub pending_signals: u64,
    pub signal_mask: u64,
    pub signal_handlers: [u64; 32],
    pub signal_saved_context: Option<UserContext>,

    // T5: Topological process tree
    pub parent_id: Option<u64>,
    pub children: Vec<u64>,

    // Threading
    pub is_thread: bool,
    pub thread_group_leader: u64, // 0 if not a thread

    // TLS — FS base + 64 slots per thread
    pub tls_base: u64,
    pub tls_slots: [u64; 64],

    // T1: Affinity hint for Voronoi cell assignment
    pub affinity_hint: usize,

    // T4: Job object membership
    pub job_id: Option<u64>,
}

impl Task {
    pub fn new(id: u64, name: &str, priority: u8, entry: fn()) -> Self {
        let embedding = Self::compute_embedding(id);
        let mut kernel_stack = vec![0u8; KERNEL_STACK_SIZE];

        let xsave_size = xsave_area_size();
        let xsave_storage = vec![0u8; xsave_size + 64];
        let xsave_ptr = align_up(xsave_storage.as_ptr() as usize, 64) as *mut u8;

        let context = init_task_context(&mut kernel_stack, entry, xsave_ptr);

        Self {
            id,
            name: String::from(name),
            state: TaskState::Ready,
            manifold_embedding: embedding,
            voronoi_cell: (id as usize) % 8,
            priority,
            ticks_used: 0,
            entry,
            is_userspace: false,
            user_stack: Vec::new(),
            page_table: 0,
            brk_end: 0,
            context,
            xsave_storage,
            kernel_stack,
            uid: 0,
            gid: 0,
            euid: 0,
            egid: 0,
            groups: Vec::new(),
            cwd: String::from("/"),
            pending_signals: 0,
            signal_mask: 0,
            signal_handlers: [0; 32],
            signal_saved_context: None,
            parent_id: None,
            children: Vec::new(),
            is_thread: false,
            thread_group_leader: 0,
            tls_base: 0,
            tls_slots: [0; 64],
            affinity_hint: 0,
            job_id: None,
        }
    }

    /// Create a userspace task from a loaded ELF.
    pub fn new_user(
        id: u64,
        name: &str,
        priority: u8,
        entry_point: u64,
        user_stack_top: u64,
        page_table: u64,
    ) -> Self {
        let embedding = Self::compute_embedding(id);
        let kernel_stack = vec![0u8; KERNEL_STACK_SIZE];

        let stack_top = kernel_stack.as_ptr() as u64 + kernel_stack.len() as u64;
        let stack_top = stack_top & !0xF;

        let xsave_size = xsave_area_size();
        let xsave_storage = vec![0u8; xsave_size + 64];
        let xsave_ptr = align_up(xsave_storage.as_ptr() as usize, 64) as *mut u8;

        let mut context = TaskContext::zero();
        // When first scheduled, jump to enter_userspace_trampoline(entry, stack, pt)
        context.rip = super::userspace::enter_userspace_trampoline as *const () as u64;
        context.rdi = entry_point;
        context.rsi = user_stack_top;
        context.rdx = page_table;
        context.rsp = stack_top;
        context.rflags = 0x202; // IF set
        context.xsave_ptr = xsave_ptr;

        Self {
            id,
            name: String::from(name),
            state: TaskState::Ready,
            manifold_embedding: embedding,
            voronoi_cell: (id as usize) % 8,
            priority,
            ticks_used: 0,
            entry: || {}, // dummy — userspace uses entry_point instead
            is_userspace: true,
            user_stack: Vec::new(),
            page_table,
            brk_end: 0x1000_0000,
            context,
            xsave_storage,
            kernel_stack,
            uid: 1000,
            gid: 1000,
            euid: 1000,
            egid: 1000,
            groups: Vec::new(),
            cwd: String::from("/"),
            pending_signals: 0,
            signal_mask: 0,
            signal_handlers: [0; 32],
            signal_saved_context: None,
            parent_id: None,
            children: Vec::new(),
            is_thread: false,
            thread_group_leader: 0,
            tls_base: 0,
            tls_slots: [0; 64],
            affinity_hint: 0,
            job_id: None,
        }
    }

    fn compute_embedding(id: u64) -> [f64; 8] {
        let mut e = [0.0f64; 8];
        let mut hash = id.wrapping_mul(0x517cc1b727220a95);
        for slot in e.iter_mut() {
            hash = hash.wrapping_mul(0x6c62272e07bb0142).wrapping_add(1);
            *slot = ((hash >> 32) as f64) / (u32::MAX as f64);
        }
        let norm = libm::sqrt(e.iter().map(|x| x * x).sum::<f64>());
        if norm > 1e-12 {
            for slot in e.iter_mut() {
                *slot /= norm;
            }
        }
        e
    }
}

pub(crate) fn align_up(addr: usize, align: usize) -> usize {
    (addr + align - 1) & !(align - 1)
}
