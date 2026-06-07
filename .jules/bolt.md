## 2026-06-07 - Streaming List Rendering Optimization
**Learning:** In React, streaming chunks into an array of objects (like a chat interface receiving DSP bus responses) causes O(N^2) rendering bottlenecks because state updates trigger a re-render of the entire mapped list.
**Action:** Always wrap individual list items in React.memo() when dealing with dynamically streamed text chunks to ensure only the actively updating component re-renders.
