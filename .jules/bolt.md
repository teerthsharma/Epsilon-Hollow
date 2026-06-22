## 2026-06-01 - Prevented O(N^2) Rendering in Streaming Components
**Learning:** React state arrays that update frequently (like appending chunks to streaming text) cause full list re-renders. This is an O(N^2) performance bottleneck, especially in long chat threads.
**Action:** Always wrap list items in `React.memo()` when rendering frequently updated arrays, especially in streaming applications like DSP Bus chat messages.
## 2026-06-22 - Prevent Time Re-Calculation in Volatile Components
**Learning:** Re-calculating timestamps (`new Date().toLocaleTimeString()`) inline inside array mapping within components that have highly volatile states (like chat input fields updating on every keystroke) not only causes unnecessary CPU overhead during every re-render, but also breaks functional correctness (all old messages get updated to the *current* time).
**Action:** Extract list items into memoized components (`React.memo`) and capture dynamic/temporal values in lazy local state (e.g., `useState(() => new Date().toLocaleTimeString())`) so they are evaluated exactly once upon creation and frozen.
