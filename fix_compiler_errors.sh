#!/bin/bash
sed -i 's/let _total = self.total_counts.get(key)?;/let total = self.total_counts.get(key)?;/g' kernel/seal-os/src/ml_engine.rs

sed -i 's/let ptr_opt = PML4_VIRT.lock();/let mut ptr_opt = PML4_VIRT.lock();/g' kernel/seal-os/src/memory/virt.rs
sed -i 's/let fs = crate::fs::manifold_fs::ManifoldFS::new();/let mut fs = crate::fs::manifold_fs::ManifoldFS::new();/g' kernel/seal-os/src/ml_engine.rs

sed -i 's/let user_stack = vec!\[0u8; user_stack_size\];/let mut user_stack = vec![0u8; user_stack_size];/g' kernel/seal-os/src/process/scheduler.rs

sed -i 's/let xsave_storage = vec!\[0u8; xsave_size + 64\];/let mut xsave_storage = vec![0u8; xsave_size + 64];/g' kernel/seal-os/src/process/task.rs
sed -i 's/let kernel_stack = vec!\[0u8; KERNEL_STACK_SIZE\];/let mut kernel_stack = vec![0u8; KERNEL_STACK_SIZE];/g' kernel/seal-os/src/process/task.rs

sed -i 's/let table = FILE_TABLE.lock();/let mut table = FILE_TABLE.lock();/g' kernel/seal-os/src/syscall/table.rs
sed -i 's/for (i, _name) in names.iter().enumerate() {/for (i, name) in names.iter().enumerate() {/g' kernel/seal-os/src/wm/taskbar.rs

sed -i 's/ctx.rip = kernel_task_wrapper as \*const () as u64;/ctx.rip = kernel_task_wrapper as u64;/g' kernel/seal-os/src/process/context_switch.rs
