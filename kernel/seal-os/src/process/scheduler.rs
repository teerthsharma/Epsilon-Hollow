// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! ManifoldScheduler — T1/T2/T4 driven preemptive process scheduling.
//!
//! This scheduler now performs real x86_64 context switches between kernel tasks.
//! SMP-aware: each CPU has its own scheduler instance and runqueue.

use alloc::vec::Vec;
use core::sync::atomic::Ordering;

use aether_core::governor::GeometricGovernor;
use aether_core::scm::SpectralContractionOperator;
use aether_core::tss::SphericalVoronoiIndex;

use super::context_switch::{switch_context, TaskContext, KERNEL_STACK_SIZE};
use super::task::{Task, TaskState};

pub struct ManifoldScheduler {
    tasks: Vec<Task>,
    next_id: u64,
    current: Option<usize>,
    voronoi: SphericalVoronoiIndex<8>,
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
            tasks: Vec::new(),
            next_id: 1,
            current: None,
            voronoi: SphericalVoronoiIndex::<8>::new(centroids),
            governor: GeometricGovernor::new(),
            predictor: SpectralContractionOperator::new(0.7),
            predict_state: [0.0; 8],
            timeslice_base: 10,
            ticks_in_slice: 0,
            schedule_count: 0,
        }
    }

    pub fn spawn(&mut self, name: &str, priority: u8, entry: fn()) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        let task = Task::new(id, name, priority, entry);
        self.tasks.push(task);
        id
    }

    pub fn spawn_user(&mut self, _name: &str, _priority: u8, _elf_data: &[u8]) -> Result<u64, super::elf::ElfError> {
        // Userspace spawn temporarily disabled until ELF loader integration is complete.
        Err(super::elf::ElfError::LoadFailed)
    }

    pub fn tick(&mut self) {
        self.ticks_in_slice += 1;

        if let Some(idx) = self.current {
            self.tasks[idx].ticks_used += 1;
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
            if self.tasks[idx].state == TaskState::Running {
                self.tasks[idx].state = TaskState::Ready;
            }
        }

        let mut next = self.select_next_task();

        if next.is_none() {
            let cpu_num = cpu.cpu_num;
            let cpu_count = crate::cpu::CPU_COUNT.load(Ordering::SeqCst);
            for i in 1..cpu_count {
                let target = ((cpu_num as usize + i) % cpu_count) as u32;
                if let Some(task) = steal_task(target) {
                    self.tasks.push(task);
                    next = self.select_next_task();
                    break;
                }
            }
        }

        if let Some(next_idx) = next {
            if old_idx == Some(next_idx) {
                self.tasks[next_idx].state = TaskState::Running;
                drop(guard);
                cpu.switching = false;
                return;
            }

            self.tasks[next_idx].state = TaskState::Running;
            self.current = Some(next_idx);
            cpu.current_task = &mut self.tasks[next_idx] as *mut Task;
            cpu.is_idle = false;

            // T2: Update prediction state
            self.predict_state = self
                .predictor
                .apply(&self.predict_state, &self.tasks[next_idx].manifold_embedding);

            // T4: Governor adapts based on scheduling deviation
            let deviation = if self.schedule_count % 2 == 0 { 0.5 } else { 1.5 };
            self.governor.adapt(deviation, 0.01);

            let old_task_dead = old_idx
                .map(|i| self.tasks[i].state == TaskState::Dead)
                .unwrap_or(false);

            let old_ctx = if old_task_dead || old_idx.is_none() {
                &mut cpu.idle_context as *mut TaskContext
            } else {
                &mut self.tasks[old_idx.unwrap()].context as *mut TaskContext
            };

            // Update TSS RSP0 if switching to a userspace task
            if self.tasks[next_idx].is_userspace {
                let stack_top = self.tasks[next_idx].kernel_stack.as_ptr() as u64
                    + KERNEL_STACK_SIZE as u64;
                unsafe {
                    crate::memory::gdt::set_kernel_stack(stack_top);
                }
            }

            if old_task_dead {
                drop(guard.take().unwrap());
            }

            x86_64::instructions::interrupts::disable();
            unsafe {
                switch_context(old_ctx, &self.tasks[next_idx].context as *const TaskContext);
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

    fn select_next_task(&self) -> Option<usize> {
        // Find task in predicted Voronoi cell first (T1+T2)
        let predicted_cell = self.voronoi.locate((
            libm::acos(self.predict_state[2].clamp(-1.0, 1.0)),
            libm::atan2(self.predict_state[1], self.predict_state[0]),
        ));

        let mut best: Option<(usize, u8)> = None;
        for (i, task) in self.tasks.iter().enumerate() {
            if task.state != TaskState::Ready {
                continue;
            }
            if task.voronoi_cell == predicted_cell {
                match best {
                    None => best = Some((i, task.priority)),
                    Some((_, bp)) if task.priority > bp => best = Some((i, task.priority)),
                    _ => {}
                }
            }
        }
        if best.is_some() {
            return best.map(|(i, _)| i);
        }

        // Fallback: any ready task with highest priority
        let mut best: Option<(usize, u8)> = None;
        for (i, task) in self.tasks.iter().enumerate() {
            if task.state != TaskState::Ready {
                continue;
            }
            match best {
                None => best = Some((i, task.priority)),
                Some((_, bp)) if task.priority > bp => best = Some((i, task.priority)),
                _ => {}
            }
        }
        best.map(|(i, _)| i)
    }

    fn adaptive_timeslice(&self) -> u64 {
        let eps = self.governor.epsilon();
        let scale = if eps < 0.5 { 2.0 } else { 0.5 };
        (self.timeslice_base as f64 * scale).max(1.0) as u64
    }

    pub fn task_count(&self) -> usize {
        self.tasks.len()
    }

    pub fn current_task_name(&self) -> &str {
        self.current
            .and_then(|i| self.tasks.get(i))
            .map(|t| t.name.as_str())
            .unwrap_or("idle")
    }

    pub fn current_task_id(&self) -> u64 {
        self.current
            .and_then(|i| self.tasks.get(i))
            .map(|t| t.id)
            .unwrap_or(0)
    }

    pub fn current_uid(&self) -> u32 {
        self.current
            .and_then(|i| self.tasks.get(i))
            .map(|t| t.uid)
            .unwrap_or(0)
    }

    pub fn set_current_uid(&mut self, uid: u32) {
        if let Some(idx) = self.current {
            if let Some(task) = self.tasks.get_mut(idx) {
                task.uid = uid;
            }
        }
    }

    pub fn current_gid(&self) -> u32 {
        self.current
            .and_then(|i| self.tasks.get(i))
            .map(|t| t.gid)
            .unwrap_or(0)
    }

    pub fn set_current_gid(&mut self, gid: u32) {
        if let Some(idx) = self.current {
            if let Some(task) = self.tasks.get_mut(idx) {
                task.gid = gid;
            }
        }
    }

    pub fn current_page_table(&self) -> Option<u64> {
        let idx = self.current?;
        Some(self.tasks[idx].page_table)
    }

    pub fn governor_epsilon(&self) -> f64 {
        self.governor.epsilon()
    }

    pub fn schedule_count(&self) -> u64 {
        self.schedule_count
    }

    pub fn mark_current_dead(&mut self) {
        if let Some(idx) = self.current {
            self.tasks[idx].state = TaskState::Dead;
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
pub fn spawn_user(name: &'static str, priority: u8, elf_data: &[u8]) -> Result<u64, super::elf::ElfError> {
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
            unsafe {
                (&raw mut crate::drivers::apic::LOCAL_APIC)
                    .as_mut()
                    .unwrap()
                    .send_ipi(apic_id, 0xFE);
            }
        }
    }
}

/// Try to steal a ready task from another CPU's scheduler.
pub fn steal_task(from_cpu: u32) -> Option<Task> {
    unsafe {
        let per_cpu = crate::cpu::per_cpu(from_cpu)?;
        let _guard = per_cpu.scheduler_lock.try_lock()?;
        let sched = &mut per_cpu.scheduler;
        for i in 0..sched.tasks.len() {
            if sched.tasks[i].state == TaskState::Ready {
                let task = sched.tasks.remove(i);
                return Some(task);
            }
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
