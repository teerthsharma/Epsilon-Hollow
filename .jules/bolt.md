

## 2024-05-24 - React List Memoization
**Learning:** Rendering append-only lists (like logs) in React by mapping items directly in the parent component causes O(N) re-renders across the entire list for every new addition.
**Action:** Extract the list item into a separate `React.memo()` wrapped component (e.g., `LogEntry`) to achieve O(1) rendering for new additions while skipping existing ones.
