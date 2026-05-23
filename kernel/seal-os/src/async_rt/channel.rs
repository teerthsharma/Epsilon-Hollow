// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Async mpsc channel — bounded message passing.

use alloc::collections::VecDeque;
use alloc::sync::Arc;
use spin::Mutex;

pub struct Sender<T> {
    inner: Arc<Mutex<ChannelInner<T>>>,
}

pub struct Receiver<T> {
    inner: Arc<Mutex<ChannelInner<T>>>,
}

struct ChannelInner<T> {
    queue: VecDeque<T>,
    capacity: usize,
    closed: bool,
}

pub fn channel<T>(capacity: usize) -> (Sender<T>, Receiver<T>) {
    let inner = Arc::new(Mutex::new(ChannelInner {
        queue: VecDeque::with_capacity(capacity),
        capacity,
        closed: false,
    }));
    (
        Sender {
            inner: inner.clone(),
        },
        Receiver { inner },
    )
}

impl<T> Sender<T> {
    pub fn send(&self, value: T) -> Result<(), T> {
        let mut inner = self.inner.lock();
        if inner.closed || inner.queue.len() >= inner.capacity {
            return Err(value);
        }
        inner.queue.push_back(value);
        Ok(())
    }

    pub fn close(&self) {
        self.inner.lock().closed = true;
    }
}

impl<T> Clone for Sender<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<T> Receiver<T> {
    pub fn try_recv(&self) -> Option<T> {
        self.inner.lock().queue.pop_front()
    }
}
