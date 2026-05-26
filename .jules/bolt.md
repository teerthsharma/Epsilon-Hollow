## 2024-05-24 - Streaming Render O(N^2) Anti-Pattern
**Learning:** During chat message streaming via DSP Bus, Next.js/React re-renders the entire message array for every single incoming chunk, creating an O(N^2) bottleneck. This is because streaming updates the `messages` state which causes a re-render of the list.
**Action:** Always wrap list items in `React.memo()` when dealing with streamed text arrays to ensure only the currently streaming active message re-renders.
