// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Minimal async runtime — waker-based executor.
//! Supports polling tasks only when woken.

pub mod channel;
pub mod io;
pub mod task;
pub mod timer;

use alloc::collections::{BTreeMap, VecDeque};
use alloc::sync::Arc;
use alloc::task::Wake;
use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll, Waker};
use spin::Mutex;

use self::task::{Task, TaskId};

type ReadyQueue = Arc<Mutex<VecDeque<TaskId>>>;

struct TaskWaker {
    task_id: TaskId,
    ready_queue: ReadyQueue,
}

impl TaskWaker {
    fn new(task_id: TaskId, ready_queue: ReadyQueue) -> Self {
        Self {
            task_id,
            ready_queue,
        }
    }
}

impl Wake for TaskWaker {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.ready_queue.lock().push_back(self.task_id);
    }
}

pub struct Executor {
    tasks: BTreeMap<TaskId, Task>,
    ready_queue: ReadyQueue,
    next_id: u64,
}

impl Executor {
    pub fn new() -> Self {
        Self {
            tasks: BTreeMap::new(),
            ready_queue: Arc::new(Mutex::new(VecDeque::new())),
            next_id: 1,
        }
    }

    pub fn spawn<F>(&mut self, future: F) -> TaskId
    where
        F: Future<Output = ()> + 'static,
    {
        let id = TaskId(self.next_id);
        self.next_id += 1;
        self.tasks.insert(id, Task::new(id, future));
        self.ready_queue.lock().push_back(id);
        id
    }

    pub fn run_once(&mut self) {
        let task_id = self.ready_queue.lock().pop_front();
        if let Some(task_id) = task_id {
            if let Some(task) = self.tasks.get_mut(&task_id) {
                let waker =
                    Waker::from(Arc::new(TaskWaker::new(task_id, self.ready_queue.clone())));
                let mut cx = Context::from_waker(&waker);

                match Pin::new(&mut task.future).poll(&mut cx) {
                    Poll::Ready(()) => {
                        self.tasks.remove(&task_id);
                    }
                    Poll::Pending => {}
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
        self.ready_queue.lock().len()
    }
}

pub fn init() {}
