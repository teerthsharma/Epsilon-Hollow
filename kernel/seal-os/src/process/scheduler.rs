// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! ManifoldScheduler — T1/T2/T4 driven process scheduling.

use alloc::vec::Vec;

use aether_core::governor::GeometricGovernor;
use aether_core::scm::SpectralContractionOperator;
use aether_core::tss::SphericalVoronoiIndex;

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

    pub fn schedule(&mut self) {
        self.schedule_count += 1;
        self.ticks_in_slice = 0;

        if let Some(idx) = self.current {
            if self.tasks[idx].state == TaskState::Running {
                self.tasks[idx].state = TaskState::Ready;
            }
        }

        // T1: Find next task via Voronoi cell selection
        let next = self.select_next_task();

        if let Some(idx) = next {
            self.tasks[idx].state = TaskState::Running;
            self.current = Some(idx);

            // T2: Update prediction state
            self.predict_state = self
                .predictor
                .apply(&self.predict_state, &self.tasks[idx].manifold_embedding);

            // T4: Governor adapts based on scheduling deviation
            let deviation = if self.schedule_count % 2 == 0 {
                0.5
            } else {
                1.5
            };
            self.governor.adapt(deviation, 0.01);
        }
    }

    fn select_next_task(&self) -> Option<usize> {
        // Find task in predicted Voronoi cell first (T1+T2)
        let predicted_cell = self.voronoi.locate((
            libm::acos(self.predict_state[2].clamp(-1.0, 1.0)),
            libm::atan2(self.predict_state[1], self.predict_state[0]),
        ));

        // Try predicted cell first
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
        // T4: Governor epsilon scales timeslice
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

    pub fn governor_epsilon(&self) -> f64 {
        self.governor.epsilon()
    }

    pub fn schedule_count(&self) -> u64 {
        self.schedule_count
    }
}
