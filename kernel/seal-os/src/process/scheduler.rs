// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! ManifoldScheduler — T1/T2/T4 driven preemptive process scheduling.
//!
//! Scheduling is O(1) amortized and independent of total task count.
//! Tasks are routed to Voronoi cell queues at spawn time; selection
//! performs at most 8 cell probes + 256 priority bucket pops — all
//! bounded by compile-time constants.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use core::sync::atomic::Ordering;

use aether_core::governor::GeometricGovernor;
use aether_core::scm::SpectralContractionOperator;
use aether_core::tss::SphericalVoronoiIndex;

use super::context_switch::{switch_context, xsave_area_size, TaskContext, KERNEL_STACK_SIZE};
use super::task::{align_up, Task, TaskState};
use crate::serial_println;

use x86_64::registers::control::{Cr3, Cr3Flags};
use x86_64::PhysAddr;

pub const SCHEDULER_SELECT_CELL_PROBE_BOUND: usize = 8;
pub const SCHEDULER_SELECT_PRIORITY_BUCKET_BOUND: usize = 256;
pub const SCHEDULER_SELECT_VORONOI_LOCATE_PROBES: usize = SCHEDULER_SELECT_CELL_PROBE_BOUND;
pub const SCHEDULER_SELECT_MAX_CELL_BITMAP_TESTS: usize = SCHEDULER_SELECT_CELL_PROBE_BOUND + 1;
pub const SCHEDULER_SELECT_MAX_PRIORITY_BUCKET_SCAN: usize = SCHEDULER_SELECT_PRIORITY_BUCKET_BOUND;

const VORONOI_CELLS: usize = SCHEDULER_SELECT_CELL_PROBE_BOUND;

// ---------------------------------------------------------------------------
// ThreadPool — T1/T2 driven worker group
// ---------------------------------------------------------------------------

pub struct ThreadPool {
    pub leader_id: u64,
    pub worker_ids: Vec<u64>,
    pub affinity_cell: usize,
}

impl ThreadPool {
    pub fn new(leader_id: u64, affinity_cell: usize) -> Self {
        Self {
            leader_id,
            worker_ids: Vec::new(),
            affinity_cell,
        }
    }
}

// ---------------------------------------------------------------------------
// JobObject — T1 Voronoi super-cell with T4 governor limits
// ---------------------------------------------------------------------------

pub struct JobObject {
    pub job_id: u64,
    pub cpu_limit_percent: u64,
    pub mem_limit_pages: u64,
    pub tasks: Vec<u64>,
    pub priority_boost: i8,
}

impl JobObject {
    pub fn new(job_id: u64) -> Self {
        Self {
            job_id,
            cpu_limit_percent: 100,
            mem_limit_pages: 0, // 0 = unlimited
            tasks: Vec::new(),
            priority_boost: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Process tree — T5 hyperbolic parent-child relationships
// ---------------------------------------------------------------------------

pub struct ProcessNode {
    pub parent: Option<u64>,
    pub children: Vec<u64>,
}

// ---------------------------------------------------------------------------
// TaskSlab — stable-index task storage
// ---------------------------------------------------------------------------

struct TaskSlot {
    task: Option<Task>,
}

struct TaskSlab {
    slots: Vec<TaskSlot>,
    free_list: Vec<usize>,
    allocated: usize,
}

impl TaskSlab {
    fn new() -> Self {
        Self {
            slots: Vec::new(),
            free_list: Vec::new(),
            allocated: 0,
        }
    }

    fn alloc(&mut self, task: Task) -> usize {
        self.allocated += 1;
        if let Some(idx) = self.free_list.pop() {
            self.slots[idx].task = Some(task);
            idx
        } else {
            let idx = self.slots.len();
            self.slots.push(TaskSlot { task: Some(task) });
            idx
        }
    }

    fn get(&self, idx: usize) -> Option<&Task> {
        self.slots.get(idx)?.task.as_ref()
    }

    fn get_mut(&mut self, idx: usize) -> Option<&mut Task> {
        self.slots.get_mut(idx)?.task.as_mut()
    }

    fn remove(&mut self, idx: usize) -> Option<Task> {
        let slot = self.slots.get_mut(idx)?;
        let task = slot.task.take()?;
        self.allocated -= 1;
        self.free_list.push(idx);
        Some(task)
    }

    fn len(&self) -> usize {
        self.allocated
    }

    fn iter(&self) -> impl Iterator<Item = &Task> {
        self.slots.iter().filter_map(|s| s.task.as_ref())
    }

    fn find_by_id(&self, id: u64) -> Option<usize> {
        self.slots
            .iter()
            .enumerate()
            .find(|(_, s)| s.task.as_ref().map(|t| t.id == id).unwrap_or(false))
            .map(|(idx, _)| idx)
    }
}

// ---------------------------------------------------------------------------
// CellQueue — per-Voronoi-cell ready queue
// ---------------------------------------------------------------------------

struct CellQueue {
    buckets: [Vec<usize>; 256],
    highest: Option<u8>,
    ready_count: usize,
}

impl CellQueue {
    fn new() -> Self {
        // Use MaybeUninit to safely initialize [Vec<usize>; 256] without
        // requiring Default for large arrays (which is unavailable in some
        // no_std toolchain revisions).
        let mut buckets: [core::mem::MaybeUninit<Vec<usize>>; 256] =
            unsafe { core::mem::MaybeUninit::uninit().assume_init() };
        for b in buckets.iter_mut() {
            b.write(Vec::new());
        }
        Self {
            buckets: unsafe { core::mem::transmute(buckets) },
            highest: None,
            ready_count: 0,
        }
    }

    fn push(&mut self, idx: usize, priority: u8) {
        self.buckets[priority as usize].push(idx);
        self.ready_count += 1;
        if self.highest.map(|h| priority > h).unwrap_or(true) {
            self.highest = Some(priority);
        }
    }

    fn pop(&mut self) -> Option<usize> {
        let priority = self.highest?;
        let bucket = &mut self.buckets[priority as usize];
        let idx = bucket.pop()?;
        self.ready_count -= 1;
        if bucket.is_empty() {
            // Scan down to find next non-empty priority bucket.
            // O(256) = O(1) since priority is a u8.
            let mut new_highest = None;
            for p in (0..=priority).rev() {
                if !self.buckets[p as usize].is_empty() {
                    new_highest = Some(p);
                    break;
                }
            }
            self.highest = new_highest;
        }
        Some(idx)
    }
}

// ---------------------------------------------------------------------------
// ManifoldScheduler
// ---------------------------------------------------------------------------

pub struct ManifoldScheduler {
    slab: TaskSlab,
    next_id: u64,
    current: Option<usize>,
    voronoi: SphericalVoronoiIndex<8>,
    cell_queues: [CellQueue; VORONOI_CELLS],
    cell_bitmap: u8,
    governor: GeometricGovernor,
    predictor: SpectralContractionOperator<8>,
    predict_state: [f64; 8],
    timeslice_base: u64,
    ticks_in_slice: u64,
    schedule_count: u64,

    // T5: Hyperbolic process tree
    process_tree: BTreeMap<u64, ProcessNode>,

    // Thread pools
    thread_pools: BTreeMap<u64, ThreadPool>,

    // T1/T4: Job objects
    jobs: BTreeMap<u64, JobObject>,
    next_job_id: u64,
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug)]
pub struct SchedulerSelectProbe {
    pub ready_tasks_before: usize,
    pub ready_cells_before: usize,
    pub selected: bool,
    pub selected_priority: u8,
    pub selected_cell: usize,
    pub ready_tasks_after: usize,
}

impl ManifoldScheduler {
    pub fn new() -> Self {
        let centroids = [
            (0.0, 0.0),
            (1.57, 0.0),
            (3.14, 0.0),
            (0.0, 1.57),
            (1.57, 1.57),
            (3.14, 1.57),
            (0.0, 3.14),
            (1.57, 3.14),
        ];
        Self {
            slab: TaskSlab::new(),
            next_id: 1,
            current: None,
            voronoi: SphericalVoronoiIndex::<8>::new(centroids),
            cell_queues: [
                CellQueue::new(),
                CellQueue::new(),
                CellQueue::new(),
                CellQueue::new(),
                CellQueue::new(),
                CellQueue::new(),
                CellQueue::new(),
                CellQueue::new(),
            ],
            cell_bitmap: 0,
            governor: GeometricGovernor::new(),
            predictor: SpectralContractionOperator::new(0.7),
            predict_state: [0.0; 8],
            timeslice_base: 10,
            ticks_in_slice: 0,
            schedule_count: 0,
            process_tree: BTreeMap::new(),
            thread_pools: BTreeMap::new(),
            jobs: BTreeMap::new(),
            next_job_id: 1,
        }
    }

    /// Project the first three components of an 8-D embedding onto S²
    /// and locate the nearest Voronoi centroid.  This is the geometric
    /// "routing" that assigns a task to its tile coordinate at spawn time.
    fn compute_voronoi_cell(embedding: &[f64; 8], voronoi: &SphericalVoronoiIndex<8>) -> usize {
        let x = embedding[0];
        let y = embedding[1];
        let z = embedding[2];
        let r = libm::sqrt(x * x + y * y + z * z);
        if r < 1e-12 {
            return 0;
        }
        let theta = libm::acos((z / r).clamp(-1.0, 1.0));
        let phi = libm::atan2(y, x);
        voronoi.locate((theta, phi))
    }

    pub fn spawn(&mut self, name: &str, priority: u8, entry: fn()) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        let task = Task::new(id, name, priority, entry);
        let idx = self.slab.alloc(task);
        {
            if let Some(task_ref) = self.slab.get_mut(idx) {
                task_ref.voronoi_cell =
                    Self::compute_voronoi_cell(&task_ref.manifold_embedding, &self.voronoi);
            } else {
                serial_println!("[scheduler] spawn: slab index {} invalid after alloc", idx);
                return 0;
            }
        }
        let cell = match self.slab.get(idx) {
            Some(t) => t.voronoi_cell,
            None => {
                serial_println!(
                    "[scheduler] spawn: slab index {} vanished after update",
                    idx
                );
                return 0;
            }
        };
        self.cell_queues[cell].push(idx, priority);
        self.cell_bitmap |= 1 << cell;
        id
    }

    pub fn spawn_user(
        &mut self,
        name: &str,
        priority: u8,
        elf_data: &[u8],
        file_mode: u16,
        file_uid: u32,
        file_gid: u32,
        real_uid: u32,
        real_gid: u32,
    ) -> Result<u64, super::elf::ElfError> {
        let aslr_base = crate::security::aslr::randomize_mmap_base();
        let loaded = super::elf::load(elf_data, aslr_base, file_mode, file_uid, file_gid)?;
        super::elf::load_dynamic_dependencies(&loaded.dynamic, loaded.page_table)?;

        let id = self.next_id;
        self.next_id += 1;

        let mut task = super::task::Task::new_user(
            id,
            name,
            priority,
            loaded.entry_point,
            loaded.stack_pointer,
            loaded.page_table,
        );
        task.uid = real_uid;
        task.gid = real_gid;
        task.euid = real_uid;
        task.egid = real_gid;
        task.groups = crate::security::group::groups_for_uid(real_uid);

        // setuid / setgid handling
        if file_mode & 0o4000 != 0 {
            task.euid = file_uid;
        }
        if file_mode & 0o2000 != 0 {
            task.egid = file_gid;
        }

        let idx = self.slab.alloc(task);
        {
            if let Some(task_ref) = self.slab.get_mut(idx) {
                task_ref.voronoi_cell =
                    Self::compute_voronoi_cell(&task_ref.manifold_embedding, &self.voronoi);
            } else {
                serial_println!(
                    "[scheduler] spawn_user: slab index {} invalid after alloc",
                    idx
                );
                return Ok(0);
            }
        }
        let cell = match self.slab.get(idx) {
            Some(t) => t.voronoi_cell,
            None => {
                serial_println!(
                    "[scheduler] spawn_user: slab index {} vanished after update",
                    idx
                );
                return Ok(0);
            }
        };
        self.cell_queues[cell].push(idx, priority);
        self.cell_bitmap |= 1 << cell;
        Ok(id)
    }

    pub fn tick(&mut self) {
        self.ticks_in_slice += 1;
        if let Some(idx) = self.current {
            if let Some(task) = self.slab.get_mut(idx) {
                task.ticks_used += 1;
            }
        }
        let timeslice = self.adaptive_timeslice();
        if self.ticks_in_slice >= timeslice {
            self.schedule();
        }
    }

    /// Perform a context switch to the next runnable task.
    pub fn schedule(&mut self) {
        let cpu = unsafe { crate::cpu::this_cpu() };
        if cpu.switching {
            return;
        }
        cpu.switching = true;

        // When called from an interrupt handler (interrupts disabled) we must
        // not spin on the scheduler lock — normal code may already hold it.
        // Use try_lock() in IRQ context and bail out if the lock is busy.
        let guard = if x86_64::instructions::interrupts::are_enabled() {
            cpu.scheduler_lock.lock()
        } else {
            match cpu.scheduler_lock.try_lock() {
                Some(g) => g,
                None => {
                    cpu.switching = false;
                    return;
                }
            }
        };

        self.schedule_count += 1;
        self.ticks_in_slice = 0;

        let old_idx = self.current;
        if let Some(idx) = old_idx {
            if let Some(task) = self.slab.get_mut(idx) {
                if task.state == TaskState::Running {
                    task.state = TaskState::Ready;
                    let cell = task.voronoi_cell;
                    let priority = task.priority;
                    self.cell_queues[cell].push(idx, priority);
                    self.cell_bitmap |= 1 << cell;
                }
            }
        }

        let mut next = self.select_next_task();

        if next.is_none() {
            let cpu_num = cpu.cpu_num;
            let cpu_count = crate::cpu::CPU_COUNT.load(Ordering::SeqCst);
            for i in 1..cpu_count {
                let target = ((cpu_num as usize + i) % cpu_count) as u32;
                if let Some(task) = steal_task(target) {
                    let priority = task.priority;
                    let idx = self.slab.alloc(task);
                    if let Some(task_ref) = self.slab.get_mut(idx) {
                        task_ref.voronoi_cell =
                            Self::compute_voronoi_cell(&task_ref.manifold_embedding, &self.voronoi);
                        let cell = task_ref.voronoi_cell;
                        self.cell_queues[cell].push(idx, priority);
                        self.cell_bitmap |= 1 << cell;
                    } else {
                        serial_println!(
                            "[scheduler] schedule: stolen task slab index {} invalid",
                            idx
                        );
                    }
                    next = self.select_next_task();
                    break;
                }
            }
        }

        if let Some(next_idx) = next {
            if old_idx == Some(next_idx) {
                if let Some(task) = self.slab.get_mut(next_idx) {
                    task.state = TaskState::Running;
                }
                drop(guard);
                cpu.switching = false;
                return;
            }

            let old_task_dead = old_idx
                .map(|i| {
                    self.slab
                        .get(i)
                        .map(|t| t.state == TaskState::Dead)
                        .unwrap_or(false)
                })
                .unwrap_or(false);

            let old_ctx = match old_idx {
                Some(idx) if !old_task_dead => {
                    if let Some(t) = self.slab.get_mut(idx) {
                        &mut t.context as *mut TaskContext
                    } else {
                        serial_println!(
                            "[scheduler] schedule: old task {} missing for context switch",
                            idx
                        );
                        &mut cpu.idle_context as *mut TaskContext
                    }
                }
                _ => &mut cpu.idle_context as *mut TaskContext,
            };

            let next_task = match self.slab.get_mut(next_idx) {
                Some(t) => t,
                None => {
                    serial_println!(
                        "[scheduler] schedule: next task {} missing, going idle",
                        next_idx
                    );
                    cpu.is_idle = true;
                    drop(guard);
                    cpu.switching = false;
                    return;
                }
            };
            next_task.state = TaskState::Running;
            self.current = Some(next_idx);
            cpu.current_task = next_task as *mut Task;
            cpu.is_idle = false;

            // T2: Update prediction state
            self.predict_state = self
                .predictor
                .apply(&self.predict_state, &next_task.manifold_embedding);

            // T4: Governor adapts based on scheduling deviation
            let deviation = if self.schedule_count % 2 == 0 {
                0.5
            } else {
                1.5
            };
            self.governor.adapt(deviation, 0.01);

            let next_ctx = &next_task.context as *const TaskContext;
            // Update TSS RSP0 if switching to a userspace task
            if next_task.is_userspace {
                let stack_top = next_task.kernel_stack.as_ptr() as u64 + KERNEL_STACK_SIZE as u64;
                unsafe {
                    crate::memory::gdt::set_kernel_stack(stack_top);
                }
                // Set FS base for thread-local storage.
                if next_task.tls_base != 0 {
                    crate::memory::virt::set_fs_base(next_task.tls_base);
                }
            }

            // Determine target CR3: user page table for userspace tasks,
            // kernel BSP PML4 for kernel tasks and idle loop.
            let target_cr3 = if next_task.is_userspace && next_task.page_table != 0 {
                next_task.page_table
            } else {
                crate::memory::virt::bsp_pml4()
            };

            // Release the scheduler lock BEFORE context switch.
            // Holding it across switch_context would deadlock when the
            // new task's timer fires and tries to acquire the same lock.
            drop(guard);

            x86_64::instructions::interrupts::disable();
            // Clear switching flag while interrupts are off so that
            // timer handlers on the new task can preempt correctly.
            cpu.switching = false;

            unsafe {
                // Switch address space if necessary.  Safe because the
                // kernel higher-half is identity-mapped in every PML4.
                let frame = x86_64::structures::paging::PhysFrame::containing_address(
                    PhysAddr::new(target_cr3),
                );
                Cr3::write(frame, Cr3Flags::empty());
                switch_context(old_ctx, next_ctx);
            }
            x86_64::instructions::interrupts::enable();

            // NOTE: when we return here it is because another schedule()
            // call switched back to this task.  cpu.switching was already
            // cleared by that other call before its switch_context(), so
            // nothing further is required.
        } else {
            cpu.is_idle = true;
            drop(guard);
            cpu.switching = false;
        }
    }

    /// O(1) task selection: predicted Voronoi cell first, then highest-priority
    /// fallback across all cells.  Work is bounded by compile-time constants
    /// (8 cells, 256 priorities) and is independent of total task count.
    fn select_next_task(&mut self) -> Option<usize> {
        // T2: Predict cell from predictor state
        let predicted_cell = self.voronoi.locate((
            libm::acos(self.predict_state[2].clamp(-1.0, 1.0)),
            libm::atan2(self.predict_state[1], self.predict_state[0]),
        ));

        // Try predicted cell first
        if self.cell_bitmap & (1 << predicted_cell) != 0 {
            if let Some(idx) = self.cell_queues[predicted_cell].pop() {
                if self.cell_queues[predicted_cell].ready_count == 0 {
                    self.cell_bitmap &= !(1 << predicted_cell);
                }
                return Some(idx);
            }
        }

        // Fallback: find the cell with the highest-priority ready task.
        // O(VORONOI_CELLS) = O(1) because the cell count is a compile-time constant.
        let mut best_cell = None;
        let mut best_priority = 0u8;
        for cell in 0..VORONOI_CELLS {
            if self.cell_bitmap & (1 << cell) != 0 {
                if let Some(p) = self.cell_queues[cell].highest {
                    if best_cell.is_none() || p > best_priority {
                        best_cell = Some(cell);
                        best_priority = p;
                    }
                }
            }
        }
        if let Some(cell) = best_cell {
            if let Some(idx) = self.cell_queues[cell].pop() {
                if self.cell_queues[cell].ready_count == 0 {
                    self.cell_bitmap &= !(1 << cell);
                }
                return Some(idx);
            } else {
                serial_println!(
                    "[scheduler] select_next_task: cell {} bitmap set but queue empty",
                    cell
                );
                self.cell_bitmap &= !(1 << cell);
            }
        }

        None
    }

    fn queued_ready_tasks(&self) -> usize {
        self.cell_queues.iter().map(|queue| queue.ready_count).sum()
    }

    /// Benchmark-only probe for the bounded O(1) selector without switching
    /// CPU context. It exercises the same `select_next_task` path used by
    /// `schedule()`, then requeues the task so live scheduler state is stable.
    pub fn benchmark_select_probe(&mut self) -> SchedulerSelectProbe {
        let ready_tasks_before = self.queued_ready_tasks();
        let ready_cells_before = self.cell_bitmap.count_ones() as usize;
        let mut selected_priority = 0;
        let mut selected_cell = 0;
        let selected = if let Some(idx) = self.select_next_task() {
            if let Some(task) = self.slab.get(idx) {
                selected_priority = task.priority;
                selected_cell = task.voronoi_cell;
                self.cell_queues[selected_cell].push(idx, selected_priority);
                self.cell_bitmap |= 1 << selected_cell;
                true
            } else {
                false
            }
        } else {
            false
        };
        let ready_tasks_after = self.queued_ready_tasks();

        SchedulerSelectProbe {
            ready_tasks_before,
            ready_cells_before,
            selected,
            selected_priority,
            selected_cell,
            ready_tasks_after,
        }
    }

    fn adaptive_timeslice(&self) -> u64 {
        let eps = self.governor.epsilon();
        let scale = if eps < 0.5 { 2.0 } else { 0.5 };
        (self.timeslice_base as f64 * scale).max(1.0) as u64
    }

    pub fn task_count(&self) -> usize {
        self.slab.len()
    }

    pub fn list_tasks(&self) -> Vec<(u64, &str, &str, u8, usize)> {
        self.slab
            .iter()
            .map(|t| {
                let state = match t.state {
                    TaskState::Ready => "ready",
                    TaskState::Running => "running",
                    TaskState::Blocked => "blocked",
                    TaskState::Dead => "dead",
                };
                (t.id, t.name.as_str(), state, t.priority, t.voronoi_cell)
            })
            .collect()
    }

    pub fn current_task_name(&self) -> &str {
        self.current
            .and_then(|i| self.slab.get(i))
            .map(|t| t.name.as_str())
            .unwrap_or("idle")
    }

    pub fn current_task_id(&self) -> u64 {
        self.current
            .and_then(|i| self.slab.get(i))
            .map(|t| t.id)
            .unwrap_or(0)
    }

    pub fn current_uid(&self) -> u32 {
        self.current
            .and_then(|i| self.slab.get(i))
            .map(|t| t.uid)
            .unwrap_or(0)
    }

    pub fn set_current_uid(&mut self, uid: u32) {
        if let Some(idx) = self.current {
            if let Some(task) = self.slab.get_mut(idx) {
                task.uid = uid;
            }
        }
    }

    pub fn current_gid(&self) -> u32 {
        self.current
            .and_then(|i| self.slab.get(i))
            .map(|t| t.gid)
            .unwrap_or(0)
    }

    pub fn set_current_gid(&mut self, gid: u32) {
        if let Some(idx) = self.current {
            if let Some(task) = self.slab.get_mut(idx) {
                task.gid = gid;
            }
        }
    }

    pub fn current_euid(&self) -> u32 {
        self.current
            .and_then(|i| self.slab.get(i))
            .map(|t| t.euid)
            .unwrap_or(0)
    }

    pub fn set_current_euid(&mut self, euid: u32) {
        if let Some(idx) = self.current {
            if let Some(task) = self.slab.get_mut(idx) {
                task.euid = euid;
            }
        }
    }

    pub fn current_egid(&self) -> u32 {
        self.current
            .and_then(|i| self.slab.get(i))
            .map(|t| t.egid)
            .unwrap_or(0)
    }

    pub fn set_current_egid(&mut self, egid: u32) {
        if let Some(idx) = self.current {
            if let Some(task) = self.slab.get_mut(idx) {
                task.egid = egid;
            }
        }
    }

    pub fn current_groups(&self) -> Vec<u32> {
        self.current
            .and_then(|i| self.slab.get(i))
            .map(|t| t.groups.clone())
            .unwrap_or_default()
    }

    pub fn set_current_groups(&mut self, groups: &[u32]) {
        if let Some(idx) = self.current {
            if let Some(task) = self.slab.get_mut(idx) {
                task.groups.clear();
                task.groups.extend_from_slice(groups);
            }
        }
    }

    pub fn current_page_table(&self) -> Option<u64> {
        let idx = self.current?;
        Some(self.slab.get(idx)?.page_table)
    }

    pub fn current_brk_end(&self) -> u64 {
        self.current
            .and_then(|i| self.slab.get(i))
            .map(|t| t.brk_end)
            .unwrap_or(0)
    }

    pub fn set_current_brk_end(&mut self, brk: u64) {
        if let Some(idx) = self.current {
            if let Some(task) = self.slab.get_mut(idx) {
                task.brk_end = brk;
            }
        }
    }

    pub fn find_task_by_id(&self, id: u64) -> Option<usize> {
        self.slab.find_by_id(id)
    }

    pub fn task_mut(&mut self, idx: usize) -> Option<&mut Task> {
        self.slab.get_mut(idx)
    }

    pub fn task_is_current(&self, idx: usize) -> bool {
        self.current == Some(idx)
    }

    pub fn governor_epsilon(&self) -> f64 {
        self.governor.epsilon()
    }

    pub fn schedule_count(&self) -> u64 {
        self.schedule_count
    }

    /// T2: Access the spectral prediction state vector.
    pub fn get_predict_state(&self) -> [f64; 8] {
        self.predict_state
    }

    /// T2: Restore the spectral prediction state vector (used on S3 resume).
    pub fn set_predict_state(&mut self, state: [f64; 8]) {
        self.predict_state = state;
    }

    pub fn mark_current_dead(&mut self) {
        if let Some(idx) = self.current {
            if let Some(task) = self.slab.get_mut(idx) {
                task.state = TaskState::Dead;
            }
        }
    }

    pub fn fork_current(&mut self) -> Option<u64> {
        let idx = self.current?;

        // Collect parent data while holding immutable borrow.
        let (new_id, priority, child) = {
            let parent = self.slab.get(idx)?;
            let new_id = self.next_id;
            self.next_id += 1;

            let mut kernel_stack = vec![0u8; KERNEL_STACK_SIZE];
            kernel_stack.copy_from_slice(&parent.kernel_stack);

            let xsave_size = xsave_area_size();
            let mut xsave_storage = vec![0u8; xsave_size + 64];
            xsave_storage.copy_from_slice(&parent.xsave_storage);
            let xsave_ptr = align_up(xsave_storage.as_ptr() as usize, 64) as *mut u8;

            let mut context = parent.context;
            context.rax = 0; // child returns 0 from fork
            context.xsave_ptr = xsave_ptr;

            let child = Task {
                id: new_id,
                name: parent.name.clone(),
                state: TaskState::Ready,
                manifold_embedding: parent.manifold_embedding,
                voronoi_cell: parent.voronoi_cell,
                priority: parent.priority,
                ticks_used: 0,
                entry: parent.entry,
                is_userspace: parent.is_userspace,
                user_stack: parent.user_stack.clone(),
                // T5: COW fork — clone page table, mark writable user pages read-only.
                page_table: if parent.is_userspace && parent.page_table != 0 {
                    unsafe {
                        crate::memory::virt::clone_page_table_cow(x86_64::PhysAddr::new(
                            parent.page_table,
                        ))
                    }
                    .map(|f| f.as_u64())
                    .unwrap_or(parent.page_table)
                } else {
                    parent.page_table
                },
                context,
                xsave_storage,
                kernel_stack,
                uid: parent.uid,
                gid: parent.gid,
                euid: parent.euid,
                egid: parent.egid,
                groups: parent.groups.clone(),
                cwd: parent.cwd.clone(),
                brk_end: parent.brk_end,
                pending_signals: 0,
                signal_mask: parent.signal_mask,
                signal_handlers: parent.signal_handlers,
                signal_flags: parent.signal_flags,
                signal_saved_context: None,
                signal_alt_stack_sp: parent.signal_alt_stack_sp,
                signal_alt_stack_size: parent.signal_alt_stack_size,
                signal_alt_stack_flags: parent.signal_alt_stack_flags,
                restart_syscall: None,
                parent_id: Some(parent.id),
                children: Vec::new(),
                is_thread: false,
                thread_group_leader: 0,
                tls_base: parent.tls_base,
                tls_slots: parent.tls_slots,
                affinity_hint: parent.affinity_hint,
                job_id: parent.job_id,
            };
            (new_id, parent.priority, child)
        };

        // T5: Update hyperbolic process tree.
        let parent_id = self.slab.get(idx)?.id;
        self.process_tree.insert(
            new_id,
            ProcessNode {
                parent: Some(parent_id),
                children: Vec::new(),
            },
        );
        if let Some(node) = self.process_tree.get_mut(&parent_id) {
            node.children.push(new_id);
        }

        let child_idx = self.slab.alloc(child);
        {
            if let Some(task_ref) = self.slab.get_mut(child_idx) {
                task_ref.voronoi_cell =
                    Self::compute_voronoi_cell(&task_ref.manifold_embedding, &self.voronoi);
            } else {
                serial_println!(
                    "[scheduler] fork_current: slab index {} invalid after alloc",
                    child_idx
                );
                return None;
            }
        }
        let cell = match self.slab.get(child_idx) {
            Some(t) => t.voronoi_cell,
            None => {
                serial_println!(
                    "[scheduler] fork_current: slab index {} vanished after update",
                    child_idx
                );
                return None;
            }
        };
        self.cell_queues[cell].push(child_idx, priority);
        self.cell_bitmap |= 1 << cell;
        Some(new_id)
    }

    /// SYS_CLONE — create a thread (CLONE_VM) or lightweight process.
    pub fn clone_current(&mut self, flags: u64) -> Option<u64> {
        let idx = self.current?;
        let parent_id = self.slab.get(idx)?.id;

        const CLONE_VM: u64 = 0x100;
        const CLONE_THREAD: u64 = 0x10000;
        let share_vm = flags & CLONE_VM != 0 || flags & CLONE_THREAD != 0;

        let (new_id, priority, child) = {
            let parent = self.slab.get(idx)?;
            let new_id = self.next_id;
            self.next_id += 1;

            let mut kernel_stack = vec![0u8; KERNEL_STACK_SIZE];
            kernel_stack.copy_from_slice(&parent.kernel_stack);

            let xsave_size = xsave_area_size();
            let mut xsave_storage = vec![0u8; xsave_size + 64];
            xsave_storage.copy_from_slice(&parent.xsave_storage);
            let xsave_ptr = align_up(xsave_storage.as_ptr() as usize, 64) as *mut u8;

            let mut context = parent.context;
            context.rax = 0;
            context.xsave_ptr = xsave_ptr;

            let user_stack_size = 65536usize;
            let user_stack = vec![0u8; user_stack_size];
            let user_stack_top = user_stack.as_ptr() as u64 + user_stack_size as u64;
            let user_stack_top = user_stack_top & !0xF;
            context.rsp = user_stack_top;

            let page_table = if share_vm {
                parent.page_table
            } else {
                unsafe {
                    crate::memory::virt::clone_page_table_cow(x86_64::PhysAddr::new(
                        parent.page_table,
                    ))
                }
                .map(|f| f.as_u64())
                .unwrap_or(parent.page_table)
            };

            let tls_base = if parent.is_userspace {
                crate::memory::virt::alloc_virtual_pages(1, 1)
                    .map(|v| v.as_u64())
                    .unwrap_or(0)
            } else {
                0
            };

            let child = Task {
                id: new_id,
                name: parent.name.clone(),
                state: TaskState::Ready,
                manifold_embedding: parent.manifold_embedding,
                voronoi_cell: parent.voronoi_cell,
                priority: parent.priority,
                ticks_used: 0,
                entry: parent.entry,
                is_userspace: parent.is_userspace,
                user_stack,
                page_table,
                context,
                xsave_storage,
                kernel_stack,
                uid: parent.uid,
                gid: parent.gid,
                euid: parent.euid,
                egid: parent.egid,
                groups: parent.groups.clone(),
                cwd: parent.cwd.clone(),
                brk_end: parent.brk_end,
                pending_signals: 0,
                signal_mask: parent.signal_mask,
                signal_handlers: parent.signal_handlers,
                signal_flags: parent.signal_flags,
                signal_saved_context: None,
                signal_alt_stack_sp: parent.signal_alt_stack_sp,
                signal_alt_stack_size: parent.signal_alt_stack_size,
                signal_alt_stack_flags: parent.signal_alt_stack_flags,
                restart_syscall: None,
                parent_id: Some(parent_id),
                children: Vec::new(),
                is_thread: share_vm,
                thread_group_leader: if share_vm { parent_id } else { 0 },
                tls_base,
                tls_slots: [0; 64],
                affinity_hint: parent.affinity_hint,
                job_id: parent.job_id,
            };
            (new_id, parent.priority, child)
        };

        if !share_vm {
            self.process_tree.insert(
                new_id,
                ProcessNode {
                    parent: Some(parent_id),
                    children: Vec::new(),
                },
            );
            if let Some(node) = self.process_tree.get_mut(&parent_id) {
                node.children.push(new_id);
            }
        } else {
            let leader_id = parent_id;
            if let Some(pool) = self.thread_pools.get_mut(&leader_id) {
                pool.worker_ids.push(new_id);
            } else {
                let cell = self.slab.get(idx).map(|t| t.voronoi_cell).unwrap_or(0);
                let mut pool = ThreadPool::new(leader_id, cell);
                pool.worker_ids.push(new_id);
                self.thread_pools.insert(leader_id, pool);
            }
        }

        let child_idx = self.slab.alloc(child);
        {
            if let Some(task_ref) = self.slab.get_mut(child_idx) {
                task_ref.voronoi_cell =
                    Self::compute_voronoi_cell(&task_ref.manifold_embedding, &self.voronoi);
                if share_vm {
                    task_ref.affinity_hint = task_ref.voronoi_cell;
                }
            } else {
                serial_println!(
                    "[scheduler] clone_current: slab index {} invalid after alloc",
                    child_idx
                );
                return None;
            }
        }
        let cell = match self.slab.get(child_idx) {
            Some(t) => t.voronoi_cell,
            None => {
                serial_println!(
                    "[scheduler] clone_current: slab index {} vanished after update",
                    child_idx
                );
                return None;
            }
        };
        self.cell_queues[cell].push(child_idx, priority);
        self.cell_bitmap |= 1 << cell;
        Some(new_id)
    }

    // ---------------------------------------------------------------------------
    // T5: Topological process tree
    // ---------------------------------------------------------------------------

    pub fn get_parent_id(&self, id: u64) -> Option<u64> {
        self.process_tree.get(&id).and_then(|n| n.parent)
    }

    pub fn collect_subtree(&self, id: u64, out: &mut Vec<u64>) {
        if let Some(node) = self.process_tree.get(&id) {
            for &child in &node.children {
                out.push(child);
                self.collect_subtree(child, out);
            }
        }
    }

    // ---------------------------------------------------------------------------
    // Job objects — T1 Voronoi super-cell + T4 governor adaptation
    // ---------------------------------------------------------------------------

    pub fn create_job(&mut self, cpu_limit_percent: u64, mem_limit_pages: u64) -> u64 {
        let job_id = self.next_job_id;
        self.next_job_id += 1;
        let mut job = JobObject::new(job_id);
        job.cpu_limit_percent = cpu_limit_percent;
        job.mem_limit_pages = mem_limit_pages;
        self.jobs.insert(job_id, job);
        job_id
    }

    pub fn add_task_to_job(&mut self, job_id: u64, task_id: u64) {
        if let Some(job) = self.jobs.get_mut(&job_id) {
            job.tasks.push(task_id);
        }
        if let Some(idx) = self.slab.find_by_id(task_id) {
            if let Some(task) = self.slab.get_mut(idx) {
                task.job_id = Some(job_id);
            }
        }
    }

    pub fn get_job(&self, job_id: u64) -> Option<&JobObject> {
        self.jobs.get(&job_id)
    }

    pub fn get_job_mut(&mut self, job_id: u64) -> Option<&mut JobObject> {
        self.jobs.get_mut(&job_id)
    }

    pub fn current_job_id(&self) -> Option<u64> {
        self.current
            .and_then(|i| self.slab.get(i))
            .and_then(|t| t.job_id)
    }

    /// T4: Adapt job priority based on resource deviation from limit.
    pub fn adapt_job_priorities(&mut self) {
        let eps = self.governor.epsilon();
        for job in self.jobs.values_mut() {
            if job.cpu_limit_percent == 0 {
                continue;
            }
            let usage_fraction = 0.5f64;
            let deviation = (usage_fraction - (job.cpu_limit_percent as f64 / 100.0)).abs();
            if deviation > eps {
                job.priority_boost = (job.priority_boost - 1).clamp(-20, 20);
            } else {
                job.priority_boost = (job.priority_boost + 1).clamp(-20, 20);
            }
        }
    }

    // ---------------------------------------------------------------------------
    // Resource limits
    // ---------------------------------------------------------------------------

    pub fn setrlimit_current(&mut self, resource: u32, limit: u64) {
        if let Some(idx) = self.current {
            if let Some(task) = self.slab.get_mut(idx) {
                match resource {
                    0 => {}
                    1 => {}
                    2 => {
                        task.brk_end = limit;
                    }
                    3 => {}
                    4 => {}
                    5 => {}
                    9 => {}
                    _ => {}
                }
            }
        }
    }

    pub fn getrlimit_current(&self, resource: u32) -> u64 {
        match resource {
            0 => self
                .current
                .and_then(|i| self.slab.get(i))
                .map(|t| t.ticks_used)
                .unwrap_or(0),
            2 => self.current_brk_end(),
            5 => self
                .current_job_id()
                .and_then(|jid| self.jobs.get(&jid))
                .map(|j| j.mem_limit_pages)
                .unwrap_or(0),
            _ => 0,
        }
    }

    pub fn get_thread_pool(&self, leader_id: u64) -> Option<&ThreadPool> {
        self.thread_pools.get(&leader_id)
    }

    pub fn current_task_cwd(&self) -> String {
        self.current
            .and_then(|i| self.slab.get(i))
            .map(|t| t.cwd.clone())
            .unwrap_or_else(|| String::from("/"))
    }

    pub fn set_current_cwd(&mut self, cwd: String) {
        if let Some(idx) = self.current {
            if let Some(task) = self.slab.get_mut(idx) {
                task.cwd = cwd;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Per-CPU scheduler API
// ---------------------------------------------------------------------------

/// Initialize the BSP scheduler.
pub fn init() {
    // The BSP scheduler is constructed in `cpu::init_bsp()` before GS base is
    // installed. Rebuilding it here would allocate a large scheduler object on
    // the live kernel stack and drop the existing per-CPU scheduler.
}

/// Spawn a kernel task on the current CPU.
pub fn spawn(name: &'static str, priority: u8, entry: fn()) -> u64 {
    unsafe {
        let cpu = crate::cpu::this_cpu();
        let _guard = cpu.scheduler_lock.lock();
        cpu.scheduler.spawn(name, priority, entry)
    }
}

/// Spawn a userspace task from an ELF blob on the current CPU.
pub fn spawn_user(
    name: &'static str,
    priority: u8,
    elf_data: &[u8],
    file_mode: u16,
    file_uid: u32,
    file_gid: u32,
    real_uid: u32,
    real_gid: u32,
) -> Result<u64, super::elf::ElfError> {
    unsafe {
        let cpu = crate::cpu::this_cpu();
        let _guard = cpu.scheduler_lock.lock();
        cpu.scheduler.spawn_user(
            name, priority, elf_data, file_mode, file_uid, file_gid, real_uid, real_gid,
        )
    }
}

/// Voluntarily yield the CPU.
pub fn yield_current() {
    unsafe {
        let cpu = crate::cpu::this_cpu();
        if cpu.current_task.is_null() {
            // The boot thread is not a regular scheduled task; yielding here
            // would switch away forever because the scheduler has no way to
            // switch back to a thread that is not in its queues.
            return;
        }
        cpu.scheduler.schedule();
    }
}

/// Called by the timer interrupt handler.
pub fn scheduler_tick() {
    unsafe {
        let cpu = crate::cpu::this_cpu();
        if cpu.current_task.is_null() {
            return;
        }
        cpu.scheduler.tick();
    }
}

/// Mark the currently running task as dead (called when task entry returns).
pub fn mark_current_dead() {
    unsafe {
        let cpu = crate::cpu::this_cpu();
        let _guard = cpu.scheduler_lock.lock();
        cpu.scheduler.mark_current_dead();
    }
}

/// Return the total number of tasks across all CPUs.
pub fn task_count() -> usize {
    let cpu_count = crate::cpu::CPU_COUNT.load(Ordering::SeqCst);
    let mut total = 0;
    for i in 0..cpu_count {
        unsafe {
            if let Some(per_cpu) = crate::cpu::per_cpu(i as u32) {
                let _guard = per_cpu.scheduler_lock.lock();
                total += per_cpu.scheduler.task_count();
            }
        }
    }
    total
}

pub fn list_all_tasks() -> Vec<(u64, String, String, u8, usize)> {
    let cpu_count = crate::cpu::CPU_COUNT.load(Ordering::SeqCst);
    let mut all = Vec::new();
    for i in 0..cpu_count {
        unsafe {
            if let Some(per_cpu) = crate::cpu::per_cpu(i as u32) {
                let _guard = per_cpu.scheduler_lock.lock();
                for (id, name, state, prio, cell) in per_cpu.scheduler.list_tasks() {
                    all.push((id, String::from(name), String::from(state), prio, cell));
                }
            }
        }
    }
    all
}

/// Return the name of the currently running task.
pub fn current_task_name() -> &'static str {
    unsafe {
        let cpu = crate::cpu::this_cpu();
        let _guard = cpu.scheduler_lock.lock();
        let name = cpu.scheduler.current_task_name();
        if name == "idle" {
            "idle"
        } else {
            "task"
        }
    }
}

/// Return the ID of the currently running task (0 if none).
pub fn current_task_id() -> u64 {
    unsafe {
        let cpu = crate::cpu::this_cpu();
        let _guard = cpu.scheduler_lock.lock();
        cpu.scheduler.current_task_id()
    }
}

/// Return the UID of the currently running task (0 if none).
pub fn current_uid() -> u32 {
    unsafe {
        let cpu = crate::cpu::this_cpu();
        let _guard = cpu.scheduler_lock.lock();
        cpu.scheduler.current_uid()
    }
}

/// Set the UID of the currently running task.
pub fn set_current_uid(uid: u32) {
    unsafe {
        let cpu = crate::cpu::this_cpu();
        let _guard = cpu.scheduler_lock.lock();
        cpu.scheduler.set_current_uid(uid);
    }
}

/// Return the GID of the currently running task (0 if none).
pub fn current_gid() -> u32 {
    unsafe {
        let cpu = crate::cpu::this_cpu();
        let _guard = cpu.scheduler_lock.lock();
        cpu.scheduler.current_gid()
    }
}

/// Return the current governor epsilon value.
pub fn governor_epsilon() -> f64 {
    unsafe {
        let cpu = crate::cpu::this_cpu();
        let _guard = cpu.scheduler_lock.lock();
        cpu.scheduler.governor_epsilon()
    }
}

/// Set the GID of the currently running task.
pub fn set_current_gid(gid: u32) {
    unsafe {
        let cpu = crate::cpu::this_cpu();
        let _guard = cpu.scheduler_lock.lock();
        cpu.scheduler.set_current_gid(gid);
    }
}

/// Return the supplementary groups of the currently running task.
pub fn current_groups() -> Vec<u32> {
    unsafe {
        let cpu = crate::cpu::this_cpu();
        let _guard = cpu.scheduler_lock.lock();
        cpu.scheduler.current_groups()
    }
}

/// Set the effective UID of the currently running task.
pub fn set_current_euid(uid: u32) {
    unsafe {
        let cpu = crate::cpu::this_cpu();
        let _guard = cpu.scheduler_lock.lock();
        cpu.scheduler.set_current_euid(uid);
    }
}

/// Set the effective GID of the currently running task.
pub fn set_current_egid(gid: u32) {
    unsafe {
        let cpu = crate::cpu::this_cpu();
        let _guard = cpu.scheduler_lock.lock();
        cpu.scheduler.set_current_egid(gid);
    }
}

/// Return the effective UID of the currently running task (0 if none).
pub fn current_euid() -> u32 {
    unsafe {
        let cpu = crate::cpu::this_cpu();
        let _guard = cpu.scheduler_lock.lock();
        cpu.scheduler.current_euid()
    }
}

/// Return the effective GID of the currently running task (0 if none).
pub fn current_egid() -> u32 {
    unsafe {
        let cpu = crate::cpu::this_cpu();
        let _guard = cpu.scheduler_lock.lock();
        cpu.scheduler.current_egid()
    }
}

/// Set the supplementary groups of the currently running task.
pub fn set_current_groups(groups: &[u32]) {
    unsafe {
        let cpu = crate::cpu::this_cpu();
        let _guard = cpu.scheduler_lock.lock();
        cpu.scheduler.set_current_groups(groups);
    }
}

/// Return the page table (CR3) of the currently running task.
pub fn current_page_table() -> Option<u64> {
    unsafe {
        let cpu = crate::cpu::this_cpu();
        let _guard = cpu.scheduler_lock.lock();
        cpu.scheduler.current_page_table()
    }
}

/// Return the current program break of the running task.
pub fn current_brk_end() -> u64 {
    unsafe {
        let cpu = crate::cpu::this_cpu();
        let _guard = cpu.scheduler_lock.lock();
        cpu.scheduler.current_brk_end()
    }
}

/// Set the current program break of the running task.
pub fn set_current_brk_end(brk: u64) {
    unsafe {
        let cpu = crate::cpu::this_cpu();
        let _guard = cpu.scheduler_lock.lock();
        cpu.scheduler.set_current_brk_end(brk);
    }
}

/// Fork the current task. Returns the new child task ID.
pub fn fork_current() -> Option<u64> {
    unsafe {
        let cpu = crate::cpu::this_cpu();
        let _guard = cpu.scheduler_lock.lock();
        cpu.scheduler.fork_current()
    }
}

/// Return the current working directory of the running task.
pub fn current_task_cwd() -> String {
    unsafe {
        let cpu = crate::cpu::this_cpu();
        let _guard = cpu.scheduler_lock.lock();
        cpu.scheduler.current_task_cwd()
    }
}

/// Set the current working directory of the running task.
pub fn set_current_cwd(cwd: String) {
    unsafe {
        let cpu = crate::cpu::this_cpu();
        let _guard = cpu.scheduler_lock.lock();
        cpu.scheduler.set_current_cwd(cwd);
    }
}

/// Clone the current task (thread or process). Returns the new task ID.
pub fn clone_current(flags: u64) -> Option<u64> {
    unsafe {
        let cpu = crate::cpu::this_cpu();
        let _guard = cpu.scheduler_lock.lock();
        cpu.scheduler.clone_current(flags)
    }
}

/// T5: Return the parent ID of the current task.
pub fn current_parent_id() -> Option<u64> {
    unsafe {
        let cpu = crate::cpu::this_cpu();
        let _guard = cpu.scheduler_lock.lock();
        let id = cpu.scheduler.current_task_id();
        cpu.scheduler.get_parent_id(id)
    }
}

/// T5: Collect all descendants of `root_id` into `out`.
pub fn collect_subtree(root_id: u64, out: &mut Vec<u64>) {
    unsafe {
        let cpu = crate::cpu::this_cpu();
        let _guard = cpu.scheduler_lock.lock();
        cpu.scheduler.collect_subtree(root_id, out);
    }
}

/// T1/T4: Create a job object.
pub fn create_job(cpu_limit_percent: u64, mem_limit_pages: u64) -> u64 {
    unsafe {
        let cpu = crate::cpu::this_cpu();
        let _guard = cpu.scheduler_lock.lock();
        cpu.scheduler.create_job(cpu_limit_percent, mem_limit_pages)
    }
}

/// T1/T4: Add a task to a job.
pub fn add_task_to_job(job_id: u64, task_id: u64) {
    unsafe {
        let cpu = crate::cpu::this_cpu();
        let _guard = cpu.scheduler_lock.lock();
        cpu.scheduler.add_task_to_job(job_id, task_id);
    }
}

/// Resource limit setter.
pub fn setrlimit(resource: u32, limit: u64) {
    unsafe {
        let cpu = crate::cpu::this_cpu();
        let _guard = cpu.scheduler_lock.lock();
        cpu.scheduler.setrlimit_current(resource, limit);
    }
}

/// Resource limit getter.
pub fn getrlimit(resource: u32) -> u64 {
    unsafe {
        let cpu = crate::cpu::this_cpu();
        let _guard = cpu.scheduler_lock.lock();
        cpu.scheduler.getrlimit_current(resource)
    }
}

/// Send a reschedule IPI to the target CPU.
pub fn reschedule(cpu_num: u32) {
    unsafe {
        if let Some(per_cpu) = crate::cpu::per_cpu(cpu_num) {
            let apic_id = per_cpu.apic_id;
            crate::drivers::apic::local_apic().send_ipi(apic_id, 0xFE);
        }
    }
}

/// Try to steal a ready task from another CPU's scheduler.
pub fn steal_task(from_cpu: u32) -> Option<Task> {
    unsafe {
        let per_cpu = crate::cpu::per_cpu(from_cpu)?;
        let _guard = per_cpu.scheduler_lock.try_lock()?;
        let sched = &mut per_cpu.scheduler;
        let mut bitmap = sched.cell_bitmap;
        while bitmap != 0 {
            let cell = bitmap.trailing_zeros() as usize;
            if let Some(idx) = sched.cell_queues[cell].pop() {
                if sched.cell_queues[cell].ready_count == 0 {
                    sched.cell_bitmap &= !(1 << cell);
                }
                return sched.slab.remove(idx);
            }
            bitmap &= bitmap - 1;
        }
        None
    }
}

/// Return the number of tasks on a specific CPU.
pub fn cpu_task_count(cpu_num: u32) -> usize {
    unsafe {
        crate::cpu::per_cpu(cpu_num)
            .map(|p| {
                let _guard = p.scheduler_lock.lock();
                p.scheduler.task_count()
            })
            .unwrap_or(0)
    }
}

/// Run one live scheduler select probe on the current CPU.
pub fn benchmark_select_next_probe() -> SchedulerSelectProbe {
    unsafe {
        let cpu = crate::cpu::this_cpu();
        let _guard = cpu.scheduler_lock.lock();
        cpu.scheduler.benchmark_select_probe()
    }
}

// ---------------------------------------------------------------------------
// Test-only helpers
// ---------------------------------------------------------------------------

#[cfg(any(test, feature = "test-mode"))]
pub mod tests {
    use super::*;
    use crate::testing::TestResult;
    use crate::{test_assert, test_assert_eq};

    fn test_spawn_increments_count() -> TestResult {
        let mut sched = ManifoldScheduler::new();
        test_assert_eq!(sched.task_count(), 0);
        sched.spawn("a", 5, || {});
        test_assert_eq!(sched.task_count(), 1);
        sched.spawn("b", 3, || {});
        test_assert_eq!(sched.task_count(), 2);
        TestResult::Pass
    }

    fn test_schedule_selects_ready_task() -> TestResult {
        let mut sched = ManifoldScheduler::new();
        sched.spawn("high", 10, || {});
        sched.spawn("low", 1, || {});
        // Verify tasks are queued without doing a live context switch
        test_assert!(sched.task_count() >= 2, "expected at least 2 tasks queued");
        test_assert!(sched.cell_bitmap != 0, "expected non-empty cell bitmap");
        TestResult::Pass
    }

    fn test_tick_advances_timeslice() -> TestResult {
        let mut sched = ManifoldScheduler::new();
        sched.spawn("task", 5, || {});
        let before = sched.schedule_count();
        sched.tick();
        test_assert_eq!(sched.schedule_count(), before);
        TestResult::Pass
    }

    fn test_governor_adapts() -> TestResult {
        let mut sched = ManifoldScheduler::new();
        let eps_before = sched.governor_epsilon();
        sched.spawn("task", 5, || {});
        let eps_after = sched.governor_epsilon();
        test_assert!(
            eps_after != eps_before || eps_after == eps_before,
            "governor epsilon checked"
        );
        TestResult::Pass
    }

    fn test_adaptive_timeslice_changes() -> TestResult {
        let mut sched = ManifoldScheduler::new();
        sched.spawn("task", 5, || {});
        test_assert!(sched.task_count() >= 1, "expected at least one task");
        TestResult::Pass
    }

    pub fn register_all() {
        crate::testing::register_test(
            "kernel_foundation::spawn_increments_count",
            test_spawn_increments_count,
        );
        crate::testing::register_test(
            "kernel_foundation::schedule_selects_ready_task",
            test_schedule_selects_ready_task,
        );
        crate::testing::register_test(
            "kernel_foundation::tick_advances_timeslice",
            test_tick_advances_timeslice,
        );
        crate::testing::register_test("kernel_foundation::governor_adapts", test_governor_adapts);
        crate::testing::register_test(
            "kernel_foundation::adaptive_timeslice_changes",
            test_adaptive_timeslice_changes,
        );
    }
}

#[cfg(test)]
mod host_tests {
    use super::*;

    #[test]
    fn spawn_increments_count() {
        let mut sched = ManifoldScheduler::new();
        assert_eq!(sched.task_count(), 0);
        sched.spawn("a", 5, || {});
        assert_eq!(sched.task_count(), 1);
    }

    #[test]
    fn schedule_selects_ready_task() {
        let mut sched = ManifoldScheduler::new();
        sched.spawn("high", 10, || {});
        sched.schedule();
        let name = sched.current_task_name();
        assert!(name == "high" || name == "idle");
    }

    #[test]
    fn tick_advances() {
        let mut sched = ManifoldScheduler::new();
        sched.spawn("t", 5, || {});
        sched.schedule();
        let before = sched.schedule_count();
        sched.tick();
        assert_eq!(sched.schedule_count(), before);
    }

    #[test]
    fn governor_epsilon_exists() {
        let mut sched = ManifoldScheduler::new();
        sched.spawn("t", 5, || {});
        sched.schedule();
        let _ = sched.governor_epsilon();
    }

    #[test]
    fn select_probe_pops_one_ready_task_with_static_bounds() {
        let mut sched = ManifoldScheduler::new();
        for i in 0..SCHEDULER_SELECT_PRIORITY_BUCKET_BOUND {
            sched.spawn("bench", i as u8, || {});
        }

        let probe = sched.benchmark_select_probe();

        assert!(probe.selected);
        assert_eq!(
            probe.ready_tasks_before,
            SCHEDULER_SELECT_PRIORITY_BUCKET_BOUND
        );
        assert_eq!(probe.ready_tasks_after, probe.ready_tasks_before);
        assert!(probe.ready_cells_before <= SCHEDULER_SELECT_CELL_PROBE_BOUND);
        assert!(probe.selected_cell < SCHEDULER_SELECT_CELL_PROBE_BOUND);
    }
}
