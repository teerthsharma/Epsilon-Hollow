
## 2024-10-14 - React Append-Only Lists Re-render Issue
**Learning:** Mapping log entries directly within a parent component for an append-only list in React causes an O(N) re-render of every item on every new addition.
**Action:** Extract list items into a `React.memo()` wrapped component to ensure O(1) re-rendering for new list additions.
