// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Minimal async runtime — single-threaded no-op-waker executor. Not Tokio-compatible.

pub mod task;
pub mod timer;
pub mod channel;
pub mod io;

use alloc::collections::VecDeque;
use alloc::vec::Vec;
use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

use self::task::{Task, TaskId};

pub struct Executor {
    tasks: Vec<Task>,
    ready_queue: VecDeque<TaskId>,
    next_id: u64,
}

impl Executor {
    pub fn new() -> Self {
        Self {
            tasks: Vec::new(),
            ready_queue: VecDeque::new(),
            next_id: 1,
        }
    }

    pub fn spawn<F>(&mut self, future: F) -> TaskId
    where
        F: Future<Output = ()> + 'static,
    {
        let id = TaskId(self.next_id);
        self.next_id += 1;
        self.tasks.push(Task::new(id, future));
        self.ready_queue.push_back(id);
        id
    }

    pub fn run_once(&mut self) {
        if let Some(task_id) = self.ready_queue.pop_front() {
            let waker = dummy_waker();
            let mut cx = Context::from_waker(&waker);

            if let Some(task) = self.tasks.iter_mut().find(|t| t.id == task_id) {
                match Pin::new(&mut task.future).poll(&mut cx) {
                    Poll::Ready(()) => {
                        self.tasks.retain(|t| t.id != task_id);
                    }
                    Poll::Pending => {
                        self.ready_queue.push_back(task_id);
                    }
                }
            }
        }
    }

    pub fn block_on<F: Future<Output = ()> + 'static>(&mut self, future: F) {
        self.spawn(future);
        while !self.tasks.is_empty() {
            self.run_once();
        }
    }

    pub fn task_count(&self) -> usize {
        self.tasks.len()
    }

    pub fn pending_count(&self) -> usize {
        self.ready_queue.len()
    }
}

fn dummy_raw_waker() -> RawWaker {
    fn no_op(_: *const ()) {}
    fn clone(p: *const ()) -> RawWaker {
        RawWaker::new(p, &VTABLE)
    }
    const VTABLE: RawWakerVTable = RawWakerVTable::new(clone, no_op, no_op, no_op);
    RawWaker::new(core::ptr::null(), &VTABLE)
}

fn dummy_waker() -> Waker {
    unsafe { Waker::from_raw(dummy_raw_waker()) }
}

pub fn init() {
}
