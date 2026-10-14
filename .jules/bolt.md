
## 2024-05-15 - React.memo for append-only lists
**Learning:** When rendering append-only lists like console logs, mapping items directly in the parent causes O(N) re-renders for every new addition.
**Action:** Extract list items into a `React.memo` wrapped component for O(1) rendering on append.
