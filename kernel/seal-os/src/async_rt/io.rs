// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Async I/O primitives — serial, keyboard, network futures.

use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll};

pub struct AsyncRead {
    ready: bool,
}

impl AsyncRead {
    pub fn serial() -> Self {
        Self { ready: false }
    }

    pub fn keyboard() -> Self {
        Self { ready: false }
    }
}

impl Future for AsyncRead {
    type Output = u8;

    fn poll(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<u8> {
        if self.ready {
            Poll::Ready(0)
        } else {
            self.ready = true;
            Poll::Pending
        }
    }
}
