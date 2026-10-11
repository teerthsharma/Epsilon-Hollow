

## 2024-05-24 - Memoizing Append-Only Lists
**Learning:** In React applications with rapidly updating append-only lists (like the Console logs), mapping items directly in the parent component causes O(N) re-renders across the entire list for every new addition.
**Action:** Extract the list item into a separate `React.memo()` wrapped component to achieve O(1) rendering for new additions, skipping unchanged existing ones.
