
## 2024-10-18 - Append-only list re-rendering
**Learning:** In React, appending to an array state (like logs) causes the parent component to re-render. If the list items are inline JSX, React will re-render *all* existing N items every time 1 new item is added, leading to O(N²) total render cost for N items.
**Action:** Always extract items in append-only lists (like consoles, feeds) into `React.memo()` components so React skips re-rendering the existing items, making appends O(1).
