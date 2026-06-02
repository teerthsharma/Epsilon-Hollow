## 2024-11-20 - React Array Rendering Optimization
**Learning:** During text streaming, state array updates in `messages.map` trigger full-list re-renders, causing O(N^2) rendering bottlenecks.
**Action:** Extract list items into a separate named component wrapped in `React.memo()` to prevent re-rendering prior items when a new chunk arrives.
