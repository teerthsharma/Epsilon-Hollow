// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! System call interface — kernel services accessible to userspace.
//! Includes Epsilon extensions for manifold operations.

pub mod ioctl;
pub mod pipe;
pub mod signal;
pub mod table;
pub mod time;
