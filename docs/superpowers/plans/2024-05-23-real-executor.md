# Real Executor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the placeholder no-op waker executor with a functional waker-based system.

**Architecture:** Synchronized `ReadyQueue` (Arc + Mutex + VecDeque) shared between `Executor` and `TaskWaker`. Tasks only polled when woken.

**Tech Stack:** Rust, `no_std`, `alloc`, `spin`.

---

### Task 1: Update `TaskId` Metadata

**Files:**
- Modify: `kernel/seal-os/src/async_rt/task.rs`

- [ ] **Step 1: Add `Ord` and `PartialOrd` to `TaskId`**
Update `TaskId` to derive `PartialOrd` and `Ord` for `BTreeMap` support.

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct TaskId(pub u64);
```

- [ ] **Step 2: Run compilation check**
Run: `cargo check -p seal-os`
Expected: Success

- [ ] **Step 3: Commit**
```bash
git add kernel/seal-os/src/async_rt/task.rs
git commit -m "refactor(async_rt): make TaskId Ord for BTreeMap support"
```

---

### Task 2: Implement `TaskWaker` and `ReadyQueue`

**Files:**
- Modify: `kernel/seal-os/src/async_rt/mod.rs`

- [ ] **Step 1: Add imports and type definitions**
Add `alloc::sync::Arc`, `alloc::collections::BTreeMap`, `core::task::Wake`, and `spin::Mutex`.
Define `ReadyQueue` as `Arc<Mutex<VecDeque<TaskId>>>`.

- [ ] **Step 2: Implement `TaskWaker`**
```rust
struct TaskWaker {
    task_id: TaskId,
    ready_queue: ReadyQueue,
}

impl TaskWaker {
    fn new(task_id: TaskId, ready_queue: ReadyQueue) -> Self {
        Self { task_id, ready_queue }
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
```

- [ ] **Step 3: Commit**
```bash
git add kernel/seal-os/src/async_rt/mod.rs
git commit -m "feat(async_rt): implement TaskWaker with shared ReadyQueue"
```

---

### Task 3: Refactor `Executor`

**Files:**
- Modify: `kernel/seal-os/src/async_rt/mod.rs`

- [ ] **Step 1: Update `Executor` struct definition**
Change `tasks: Vec<Task>` to `tasks: BTreeMap<TaskId, Task>` and `ready_queue` to `ReadyQueue`.

- [ ] **Step 2: Update `Executor::new()`**
Initialize `tasks` as `BTreeMap::new()` and `ready_queue` as `Arc::new(Mutex::new(VecDeque::new()))`.

- [ ] **Step 3: Update `Executor::spawn()`**
Insert task into `BTreeMap` and push ID to `ready_queue`.

- [ ] **Step 4: Update `Executor::run_once()`**
Pop `task_id` from `ready_queue`, get task from map, create `Waker` using `Arc<TaskWaker>`, and poll.
Remove task from map if `Poll::Ready`.

- [ ] **Step 5: Remove dummy waker functions**
Delete `dummy_waker` and `dummy_raw_waker`.

- [ ] **Step 6: Commit**
```bash
git add kernel/seal-os/src/async_rt/mod.rs
git commit -m "feat(async_rt): refactor Executor to use BTreeMap and real wakers"
```

---

### Task 4: Verification

**Files:**
- Modify: `kernel/seal-os/src/async_rt/mod.rs` (add tests)

- [ ] **Step 1: Add unit test for task yielding**
Verify that a task yielding multiple times is re-polled correctly.

- [ ] **Step 2: Run tests**
Run: `cargo test -p seal-os --lib async_rt`
Expected: PASS

- [ ] **Step 3: Final compilation check**
Run: `cargo build -p seal-os`
Expected: Success

- [ ] **Step 4: Commit**
```bash
git add kernel/seal-os/src/async_rt/mod.rs
git commit -m "test(async_rt): add yield test for real executor"
```
