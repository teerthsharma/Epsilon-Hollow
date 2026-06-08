## 2024-05-17 - Dynamic Disabled Tooltips in Chat Interfaces
**Learning:** In technical chat interfaces, a statically disabled button often leaves users confused about whether the backend is disconnected or if they simply need to provide input.
**Action:** Always provide a dynamic `title` or tooltip on send buttons that explicitly explains *why* the button is disabled (e.g., "Type a message to send"), and pair it with `disabled:cursor-not-allowed` and strong `focus-visible` ring indicators for strict accessibility.
