# Design Spec: Real Waker-based Executor for Seal OS

## Overview
Replace the current no-op waker executor in `seal-os` with a real waker-based system that correctly handles task wakeups via a synchronized ready queue.

## Proposed Changes

### 1. Ready Queue Synchronization
- Implement a `ReadyQueue` using `alloc::sync::Arc<spin::Mutex<VecDeque<TaskId>>>`.
- This queue is shared between the `Executor` and all `TaskWaker` instances.

### 2. Task Waker Implementation
- Implement `TaskWaker` which holds `TaskId` and a reference to the `ReadyQueue`.
- Implement `core::task::Wake` for `Arc<TaskWaker>`.
- The `wake()` and `wake_by_ref()` methods will push the `TaskId` into the `ReadyQueue`.

### 3. Executor Refactoring
- Change `Executor.tasks` from `Vec<Task>` to `BTreeMap<TaskId, Task>`.
- Update `spawn` to create an initial wakeup.
- Update `run_once` to:
    1. Dequeue `TaskId` from `ReadyQueue`.
    2. Retrieve the `Task`.
    3. Create a `Waker` for that `TaskId`.
    4. Poll the `Task`.
    5. Clean up on `Poll::Ready`.

### 4. Task Metadata
- `TaskId` will derive `Ord` and `PartialOrd` for `BTreeMap` support.

## Architecture Diagram (Textual)
```
[Executor] <------- [Ready Queue (Shared)] <------- [Waker (TaskId)]
    |                       ^                           |
    | (pop)                 | (push)                    | (wake)
    v                       |                           |
[Task (TaskId)] ------------+---------------------------+
```

## Testing Strategy
- Verify that tasks that return `Poll::Pending` are NOT re-polled until `wake()` is called.
- Verify `block_on` still works for simple futures.
- Verify concurrent wakeups (if multiple components have access to the waker).

## Acceptance Criteria
- `dummy_waker` is removed.
- `run_once` only polls tasks present in the `ReadyQueue`.
- `Task` structure is updated to be compatible.
- Code is `no_std` compliant and uses existing kernel primitives.
