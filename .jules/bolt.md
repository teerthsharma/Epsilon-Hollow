
## 2024-10-25 - React list performance optimization
**Learning:** Mapping over append-only lists (like logs) directly in a parent component causes O(N) re-renders across the entire list for every new addition.
**Action:** Always extract the list item into a separate `React.memo()` wrapped component to achieve O(1) rendering for new additions while skipping existing ones.
