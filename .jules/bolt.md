
## 2024-03-20 - Memoizing list items in React
**Learning:** In append-only lists like console logs, mapping items directly in the parent component causes O(N) re-renders for every new addition.
**Action:** Extracted list item into a React.memo() component to achieve O(1) rendering for new additions while skipping existing ones.
