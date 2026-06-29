## 2026-06-01 - Prevented O(N^2) Rendering in Streaming Components
**Learning:** React state arrays that update frequently (like appending chunks to streaming text) cause full list re-renders. This is an O(N^2) performance bottleneck, especially in long chat threads.
**Action:** Always wrap list items in `React.memo()` when rendering frequently updated arrays, especially in streaming applications like DSP Bus chat messages.


## 2024-05-18 - Extracted component to memoize LiquidStream messages
**Learning:** Inline message rendering where components use dynamic values like new Date() can cause unnecessary recalculations or re-renders. Storing in the data model and memoizing items prevents this.
**Action:** Extract inline components into React.memo and move dynamic timestamps to the data model.
