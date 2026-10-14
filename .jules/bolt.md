
## 2025-02-14 - O(N) Re-renders in React Append-Only Lists
**Learning:** When mapping over an append-only list (like `logs.map`) directly within a parent component that re-renders, adding a new item causes React to re-render the *entire* list from scratch (O(N)), including evaluating any inline conditional logic (like `.includes()` checks for CSS classes).
**Action:** Always extract the individual list item into a separate, `React.memo()` wrapped component. This allows React to skip re-rendering existing items, turning the render cost for a new addition from O(N) to O(1).
