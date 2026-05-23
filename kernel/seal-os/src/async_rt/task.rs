// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Async task type with JoinHandle and waker.

use alloc::boxed::Box;
use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct TaskId(pub u64);

pub struct Task {
    pub id: TaskId,
    pub future: Pin<Box<dyn Future<Output = ()>>>,
}

impl Task {
    pub fn new<F: Future<Output = ()> + 'static>(id: TaskId, future: F) -> Self {
        Self {
            id,
            future: Box::pin(future),
        }
    }
}

pub struct JoinHandle {
    pub id: TaskId,
    completed: bool,
}

impl JoinHandle {
    pub fn new(id: TaskId) -> Self {
        Self {
            id,
            completed: false,
        }
    }

    pub fn is_finished(&self) -> bool {
        self.completed
    }
}

pub struct YieldNow(bool);

impl YieldNow {
    pub fn new() -> Self {
        Self(false)
    }
}

impl Future for YieldNow {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<()> {
        if self.0 {
            Poll::Ready(())
        } else {
            self.0 = true;
            Poll::Pending
        }
    }
}

pub async fn yield_now() {
    YieldNow::new().await
}
