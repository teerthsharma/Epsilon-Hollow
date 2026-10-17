## 2024-10-18 - First Learning
**Learning:** Initializing journal.
**Action:** Ready to record performance learnings.

## 2024-10-18 - React List Rendering Performance
**Learning:** Mapping over append-only lists (like logs) directly in a parent component causes O(N) re-renders across the entire list for every new item.
**Action:** Always extract append-only list items into a separate `React.memo()` component to achieve O(1) rendering for new additions.
