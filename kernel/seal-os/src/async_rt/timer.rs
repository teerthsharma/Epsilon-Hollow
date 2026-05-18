// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Timer futures backed by PIT/HPET interrupts.

use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll};

use crate::drivers::interrupts;

pub struct Sleep {
    target_ticks: u64,
}

impl Sleep {
    pub fn new(ticks: u64) -> Self {
        Self {
            target_ticks: interrupts::ticks() + ticks,
        }
    }
}

impl Future for Sleep {
    type Output = ();

    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<()> {
        if interrupts::ticks() >= self.target_ticks {
            Poll::Ready(())
        } else {
            Poll::Pending
        }
    }
}

pub fn sleep(ticks: u64) -> Sleep {
    Sleep::new(ticks)
}

pub struct Interval {
    period_ticks: u64,
    next_tick: u64,
}

impl Interval {
    pub fn new(period_ticks: u64) -> Self {
        Self {
            period_ticks,
            next_tick: interrupts::ticks() + period_ticks,
        }
    }

    pub async fn tick(&mut self) {
        Sleep::new(self.next_tick.saturating_sub(interrupts::ticks())).await;
        self.next_tick += self.period_ticks;
    }
}
