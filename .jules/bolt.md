
## 2025-02-23 - React append-only list optimization
**Learning:** When rendering append-only lists (like logs) in React, mapping items directly in the parent component causes O(N) re-renders across the entire list for every new addition.
**Action:** To optimize performance, extract the list item into a separate `React.memo()` wrapped component to achieve O(1) rendering for new additions while skipping existing ones.
