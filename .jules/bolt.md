
## 2024-10-10 - React Append-Only List Re-render Bottleneck
**Learning:** When rendering append-only lists (like logs) in React, mapping items directly in the parent component causes O(N) re-renders across the entire list for every new addition.
**Action:** Extract list items into a separate `React.memo()` wrapped component (e.g., `LogEntry`) to achieve O(1) rendering for new additions while skipping existing ones.
