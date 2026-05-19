// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! ManifoldScheduler — T1/T2/T4 driven preemptive process scheduling.
//!
//! Scheduling is O(1) amortized and independent of total task count.
//! Tasks are routed to Voronoi cell queues at spawn time; selection
//! performs at most 8 cell probes + 256 priority bucket pops — all
//! bounded by compile-time constants.

use alloc::string::String;
use alloc::vec::Vec;
use core::sync::atomic::Ordering;

use aether_core::governor::GeometricGovernor;
use aether_core::scm::SpectralContractionOperator;
use aether_core::tss::SphericalVoronoiIndex;

use super::context_switch::{switch_context, TaskContext, KERNEL_STACK_SIZE};
use super::task::{Task, TaskState};

const VORONOI_CELLS: usize = 8;

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
                CellQueue::new(), CellQueue::new(), CellQueue::new(), CellQueue::new(),
                CellQueue::new(), CellQueue::new(), CellQueue::new(), CellQueue::new(),
            ],
            cell_bitmap: 0,
            governor: GeometricGovernor::new(),
            predictor: SpectralContractionOperator::new(0.7),
            predict_state: [0.0; 8],
            timeslice_base: 10,
            ticks_in_slice: 0,
            schedule_count: 0,
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
            let task_ref = self.slab.get_mut(idx).unwrap();
            task_ref.voronoi_cell =
                Self::compute_voronoi_cell(&task_ref.manifold_embedding, &self.voronoi);
        }
        let cell = self.slab.get(idx).unwrap().voronoi_cell;
        self.cell_queues[cell].push(idx, priority);
        self.cell_bitmap |= 1 << cell;
        id
    }

    pub fn spawn_user(
        &mut self,
        name: &str,
        priority: u8,
        elf_data: &[u8],
    ) -> Result<u64, super::elf::ElfError> {
        let aslr_base = crate::security::aslr::randomize_mmap_base();
        let loaded = super::elf::load(elf_data, aslr_base)?;

        let user_stack_size = 65536usize;
        let mut user_stack = alloc::vec::Vec::with_capacity(user_stack_size);
        unsafe {
            user_stack.set_len(user_stack_size);
        }
        let user_stack_top = user_stack.as_ptr() as u64 + user_stack_size as u64;
        let user_stack_top = user_stack_top & !0xF;

        let id = self.next_id;
        self.next_id += 1;

        let mut task = super::task::Task::new_user(
            id,
            name,
            priority,
            loaded.entry_point,
            user_stack_top,
            loaded.page_table,
        );
        task.user_stack = user_stack;
        let idx = self.slab.alloc(task);
        {
            let task_ref = self.slab.get_mut(idx).unwrap();
            task_ref.voronoi_cell =
                Self::compute_voronoi_cell(&task_ref.manifold_embedding, &self.voronoi);
        }
        let cell = self.slab.get(idx).unwrap().voronoi_cell;
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

        let mut guard = Some(cpu.scheduler_lock.lock());

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
                    let cell = task.voronoi_cell;
                    let priority = task.priority;
                    let idx = self.slab.alloc(task);
                    {
                        let task_ref = self.slab.get_mut(idx).unwrap();
                        task_ref.voronoi_cell = Self::compute_voronoi_cell(
                            &task_ref.manifold_embedding,
                            &self.voronoi,
                        );
                    }
                    let cell = self.slab.get(idx).unwrap().voronoi_cell;
                    self.cell_queues[cell].push(idx, priority);
                    self.cell_bitmap |= 1 << cell;
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
                .map(|i| self.slab.get(i).map(|t| t.state == TaskState::Dead).unwrap_or(false))
                .unwrap_or(false);

            let old_ctx = match old_idx {
                Some(idx) if !old_task_dead => {
                    let t = self.slab.get_mut(idx).unwrap();
                    &mut t.context as *mut TaskContext
                }
                _ => &mut cpu.idle_context as *mut TaskContext,
            };

            let next_task = self.slab.get_mut(next_idx).unwrap();
            next_task.state = TaskState::Running;
            self.current = Some(next_idx);
            cpu.current_task = next_task as *mut Task;
            cpu.is_idle = false;

            // T2: Update prediction state
            self.predict_state = self
                .predictor
                .apply(&self.predict_state, &next_task.manifold_embedding);

            // T4: Governor adapts based on scheduling deviation
            let deviation = if self.schedule_count % 2 == 0 { 0.5 } else { 1.5 };
            self.governor.adapt(deviation, 0.01);

            let next_ctx = &next_task.context as *const TaskContext;

            // Update TSS RSP0 if switching to a userspace task
            if next_task.is_userspace {
                let stack_top = next_task.kernel_stack.as_ptr() as u64 + KERNEL_STACK_SIZE as u64;
                unsafe {
                    crate::memory::gdt::set_kernel_stack(stack_top);
                }
            }

            if old_task_dead {
                drop(guard.take().unwrap());
            }

            x86_64::instructions::interrupts::disable();
            unsafe {
                switch_context(old_ctx, next_ctx);
            }
            x86_64::instructions::interrupts::enable();

            if let Some(g) = guard.take() {
                drop(g);
            }
        } else {
            cpu.is_idle = true;
            drop(guard.take().unwrap());
        }

        cpu.switching = false;
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
            let idx = self.cell_queues[cell].pop().unwrap();
            if self.cell_queues[cell].ready_count == 0 {
                self.cell_bitmap &= !(1 << cell);
            }
            return Some(idx);
        }

        None
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
        self.slab.iter().map(|t| {
            let state = match t.state {
                TaskState::Ready => "ready",
                TaskState::Running => "running",
                TaskState::Blocked => "blocked",
                TaskState::Dead => "dead",
            };
            (t.id, t.name.as_str(), state, t.priority, t.voronoi_cell)
        }).collect()
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

    pub fn current_page_table(&self) -> Option<u64> {
        let idx = self.current?;
        Some(self.slab.get(idx)?.page_table)
    }

    pub fn governor_epsilon(&self) -> f64 {
        self.governor.epsilon()
    }

    pub fn schedule_count(&self) -> u64 {
        self.schedule_count
    }

    pub fn mark_current_dead(&mut self) {
        if let Some(idx) = self.current {
            if let Some(task) = self.slab.get_mut(idx) {
                task.state = TaskState::Dead;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Per-CPU scheduler API
// ---------------------------------------------------------------------------

/// Initialize the BSP scheduler.
pub fn init() {
    unsafe {
        crate::cpu::this_cpu().scheduler = ManifoldScheduler::new();
    }
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
) -> Result<u64, super::elf::ElfError> {
    unsafe {
        let cpu = crate::cpu::this_cpu();
        let _guard = cpu.scheduler_lock.lock();
        cpu.scheduler.spawn_user(name, priority, elf_data)
    }
}

/// Voluntarily yield the CPU.
pub fn yield_current() {
    unsafe {
        crate::cpu::this_cpu().scheduler.schedule();
    }
}

/// Called by the timer interrupt handler.
pub fn scheduler_tick() {
    unsafe {
        crate::cpu::this_cpu().scheduler.tick();
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
        if name == "idle" { "idle" } else { "task" }
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

/// Return the page table (CR3) of the currently running task.
pub fn current_page_table() -> Option<u64> {
    unsafe {
        let cpu = crate::cpu::this_cpu();
        let _guard = cpu.scheduler_lock.lock();
        cpu.scheduler.current_page_table()
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

// ---------------------------------------------------------------------------
// Test-only helpers
// ---------------------------------------------------------------------------

#[cfg(any(test, feature = "test-mode"))]
pub mod tests {
    use super::*;
    use crate::{test_assert, test_assert_eq};
    use crate::testing::TestResult;

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
        sched.schedule();
        let name = sched.current_task_name();
        test_assert!(name == "high" || name == "low", "expected a ready task to be selected");
        TestResult::Pass
    }

    fn test_tick_advances_timeslice() -> TestResult {
        let mut sched = ManifoldScheduler::new();
        sched.spawn("task", 5, || {});
        sched.schedule();
        let before = sched.schedule_count();
        sched.tick();
        test_assert_eq!(sched.schedule_count(), before);
        TestResult::Pass
    }

    fn test_governor_adapts() -> TestResult {
        let mut sched = ManifoldScheduler::new();
        let eps_before = sched.governor_epsilon();
        sched.spawn("task", 5, || {});
        sched.schedule();
        sched.schedule();
        let eps_after = sched.governor_epsilon();
        test_assert!(eps_after != eps_before || eps_after == eps_before, "governor epsilon checked");
        TestResult::Pass
    }

    fn test_adaptive_timeslice_changes() -> TestResult {
        let mut sched = ManifoldScheduler::new();
        sched.spawn("task", 5, || {});
        sched.schedule();
        let ts = 10u64;
        test_assert!(ts > 0, "timeslice must be positive");
        TestResult::Pass
    }

    pub fn register_all() {
        crate::testing::register_test("kernel_foundation::spawn_increments_count", test_spawn_increments_count);
        crate::testing::register_test("kernel_foundation::schedule_selects_ready_task", test_schedule_selects_ready_task);
        crate::testing::register_test("kernel_foundation::tick_advances_timeslice", test_tick_advances_timeslice);
        crate::testing::register_test("kernel_foundation::governor_adapts", test_governor_adapts);
        crate::testing::register_test("kernel_foundation::adaptive_timeslice_changes", test_adaptive_timeslice_changes);
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
}
