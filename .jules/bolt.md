
## 2024-10-31 - Memoizing append-only list items
**Learning:** In React, mapping items directly in a parent component causes O(N) re-renders across the entire list whenever a new item is added, which can be an issue for constantly growing lists like logs.
**Action:** Extract list items (e.g. `LogEntry`) and wrap them in `React.memo()` to achieve O(1) rendering for new additions while skipping re-renders for existing elements.
