

## 2024-10-06 - Append-Only List Performance
**Learning:** In React, mapping over an append-only array (like logs) inline in the render method causes the entire list to re-render for every single new item appended, leading to O(N) rendering performance.
**Action:** Always extract items in append-only lists to `React.memo()` wrapped components to achieve O(1) rendering for new additions, skipping reconciliation for existing items.
