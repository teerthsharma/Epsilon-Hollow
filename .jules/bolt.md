
## 2024-10-27 - O(N) List Rendering Bottleneck in Console
**Learning:** In React, rendering an append-only array (like `logs.map(...)`) directly within a parent component triggers O(N) re-renders because every item's virtual DOM node is recreated on each state update.
**Action:** Extract list items into a separate `React.memo()` wrapped component (e.g., `LogEntry`) so new additions render in O(1) time while existing elements skip re-rendering.
