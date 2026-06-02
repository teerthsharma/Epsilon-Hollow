// Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: Epsilon-Hollow

//! ═══════════════════════════════════════════════════════════════════════════════
//! Aether Standard Library — Self-Hosting Primitives
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Expanded stdlib providing List, Map, StringBuilder, and host fs I/O
//! needed for the Aether-Lang self-hosting compiler.
//!
//! ═══════════════════════════════════════════════════════════════════════════════

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::collections::BTreeMap;
#[cfg(not(feature = "std"))]
use alloc::string::String;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

#[cfg(feature = "std")]
use std::collections::BTreeMap;

use core::alloc::Layout;
use core::ptr::null_mut;
use core::sync::atomic::{AtomicUsize, Ordering};

// ═══════════════════════════════════════════════════════════════════════════════
// Slab Allocator
// ═══════════════════════════════════════════════════════════════════════════════

const SLAB_SIZE: usize = 4096;
const MAX_SLABS: usize = 256;

static mut SLAB_DATA: [[u8; SLAB_SIZE]; MAX_SLABS] = [[0; SLAB_SIZE]; MAX_SLABS];
static SLAB_USED: [AtomicUsize; MAX_SLABS] = [const { AtomicUsize::new(0) }; MAX_SLABS];
static SLAB_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Simple bump-style slab allocator.
///
/// Each slab is 4 KiB. Allocation bumps forward within a slab;
/// when a slab fills, a new one is claimed from the static pool.
pub struct SlabAllocator;

impl SlabAllocator {
    /// Allocate `layout` bytes from the slab pool.
    ///
    /// Returns a valid pointer on success, or null on exhaustion.
    pub fn alloc(layout: Layout) -> *mut u8 {
        let size = layout.size();
        let align = layout.align();

        if size == 0 {
            return align as *mut u8;
        }

        let count = SLAB_COUNT.load(Ordering::Relaxed);

        // Try existing slabs first
        for i in 0..count {
            unsafe {
                let used = SLAB_USED[i].load(Ordering::Relaxed);
                let aligned = (used + align - 1) & !(align - 1);
                let new_used = aligned + size;

                if new_used > SLAB_SIZE {
                    continue;
                }

                if SLAB_USED[i]
                    .compare_exchange(used, new_used, Ordering::SeqCst, Ordering::Relaxed)
                    .is_ok()
                {
                    return SLAB_DATA[i].as_ptr().add(aligned) as *mut u8;
                }
            }
        }

        // Claim a new slab
        if count < MAX_SLABS {
            let idx = SLAB_COUNT.fetch_add(1, Ordering::SeqCst);
            if idx < MAX_SLABS {
                unsafe {
                    let aligned = (0usize + align - 1) & !(align - 1);
                    SLAB_USED[idx].store(aligned + size, Ordering::SeqCst);
                    return SLAB_DATA[idx].as_ptr().add(aligned) as *mut u8;
                }
            }
        }

        null_mut()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// List<T> — Generic growable array backed by SlabAllocator
// ═══════════════════════════════════════════════════════════════════════════════

/// A growable array backed by the slab allocator.
pub struct List<T> {
    ptr: *mut T,
    len: usize,
    cap: usize,
}

impl<T> List<T> {
    /// Create an empty `List`.
    pub const fn new() -> Self {
        Self {
            ptr: null_mut(),
            len: 0,
            cap: 0,
        }
    }

    /// Push a value onto the end of the list.
    pub fn push(&mut self, value: T) {
        if self.len >= self.cap {
            let new_cap = if self.cap == 0 { 8 } else { self.cap * 2 };
            let layout = Layout::array::<T>(new_cap).expect("valid layout");
            let new_ptr = SlabAllocator::alloc(layout) as *mut T;

            assert!(!new_ptr.is_null(), "slab allocator exhausted");

            if !self.ptr.is_null() {
                unsafe {
                    core::ptr::copy_nonoverlapping(self.ptr, new_ptr, self.len);
                }
            }

            self.ptr = new_ptr;
            self.cap = new_cap;
        }

        unsafe {
            self.ptr.add(self.len).write(value);
        }
        self.len += 1;
    }

    /// Remove and return the last element, if any.
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }
        self.len -= 1;
        Some(unsafe { self.ptr.add(self.len).read() })
    }

    /// Get an immutable reference to the element at `index`.
    pub fn get(&self, index: usize) -> Option<&T> {
        if index < self.len {
            Some(unsafe { &*self.ptr.add(index) })
        } else {
            None
        }
    }

    /// Get a mutable reference to the element at `index`.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index < self.len {
            Some(unsafe { &mut *self.ptr.add(index) })
        } else {
            None
        }
    }

    /// Return the number of elements in the list.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Return `true` if the list contains no elements.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<T> Default for List<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Drop for List<T> {
    fn drop(&mut self) {
        // Slab allocator doesn't support individual deallocation,
        // but we must drop the elements to avoid leaks of owned data.
        while self.pop().is_some() {}
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Map<K,V> — Generic map backed by BTreeMap bridge
// ═══════════════════════════════════════════════════════════════════════════════

/// An ordered map backed by `BTreeMap`.
pub struct Map<K, V> {
    inner: BTreeMap<K, V>,
}

impl<K: Ord, V> Map<K, V> {
    /// Create an empty `Map`.
    pub fn new() -> Self {
        Self {
            inner: BTreeMap::new(),
        }
    }

    /// Insert a key-value pair into the map.
    ///
    /// Returns the old value if the key was already present.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        self.inner.insert(key, value)
    }

    /// Get an immutable reference to the value associated with `key`.
    pub fn get(&self, key: &K) -> Option<&V> {
        self.inner.get(key)
    }

    /// Get a mutable reference to the value associated with `key`.
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        self.inner.get_mut(key)
    }

    /// Remove a key from the map, returning the value if present.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        self.inner.remove(key)
    }

    /// Return a `Vec` containing a clone of every key in sorted order.
    pub fn keys(&self) -> Vec<K>
    where
        K: Clone,
    {
        self.inner.keys().cloned().collect()
    }

    /// Return the number of key-value pairs in the map.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Return `true` if the map contains no entries.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

impl<K: Ord, V> Default for Map<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// StringBuilder — Efficient string concatenation
// ═══════════════════════════════════════════════════════════════════════════════

/// A builder for efficiently concatenating strings.
pub struct StringBuilder {
    buffer: String,
}

impl StringBuilder {
    /// Create an empty `StringBuilder`.
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
        }
    }

    /// Create a `StringBuilder` with the specified capacity hint.
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            buffer: String::with_capacity(cap),
        }
    }

    /// Append a string slice.
    pub fn append(&mut self, s: &str) {
        self.buffer.push_str(s);
    }

    /// Append a single character.
    pub fn append_char(&mut self, c: char) {
        self.buffer.push(c);
    }

    /// Consume the builder and return the concatenated `String`.
    pub fn to_string(self) -> String {
        self.buffer
    }

    /// Return the current length in bytes.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Return `true` if the builder is empty.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Clear the builder, retaining the allocated buffer.
    pub fn clear(&mut self) {
        self.buffer.clear();
    }

    /// Return an immutable view of the accumulated string.
    pub fn as_str(&self) -> &str {
        &self.buffer
    }
}

impl Default for StringBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// fs — Host-side file I/O
// ═══════════════════════════════════════════════════════════════════════════════

/// Host-side filesystem operations.
pub mod fs {
    #[cfg(feature = "std")]
    use std::fs::File;
    #[cfg(feature = "std")]
    use std::io::{Read, Write};

    #[cfg(not(feature = "std"))]
    use alloc::string::String;

    /// Read the entire contents of a text file into a `String`.
    ///
    /// # Errors
    ///
    /// Returns `Err` with a message if the file cannot be opened or read.
    #[cfg(feature = "std")]
    pub fn read_text(path: &str) -> Result<String, String> {
        let mut file = File::open(path).map_err(|e| e.to_string())?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)
            .map_err(|e| e.to_string())?;
        Ok(contents)
    }

    /// Write `contents` to a text file at `path`, creating or truncating it.
    ///
    /// # Errors
    ///
    /// Returns `Err` with a message if the file cannot be created or written.
    #[cfg(feature = "std")]
    pub fn write_text(path: &str, contents: &str) -> Result<(), String> {
        let mut file = File::create(path).map_err(|e| e.to_string())?;
        file.write_all(contents.as_bytes())
            .map_err(|e| e.to_string())?;
        Ok(())
    }

    /// Stub for `no_std` environments.
    #[cfg(not(feature = "std"))]
    pub fn read_text(_path: &str) -> Result<String, String> {
        Err(String::from("fs::read_text not available in no_std"))
    }

    /// Stub for `no_std` environments.
    #[cfg(not(feature = "std"))]
    pub fn write_text(_path: &str, _contents: &str) -> Result<(), String> {
        Err(String::from("fs::write_text not available in no_std"))
    }
}
