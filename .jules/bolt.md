
## 2025-01-20 - Memoizing Append-Only Lists
**Learning:** In React applications rendering console logs or events, mapping items directly in the parent component causes O(N) re-renders across the entire list for every new addition.
**Action:** Extract the list item into a separate `React.memo()` wrapped component to achieve O(1) rendering for new append additions while skipping existing ones.
