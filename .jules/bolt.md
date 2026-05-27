
## 2024-05-27 - Streamed State Arrays Cause O(N^2) Rendering Bottlenecks
**Learning:** When rendering streamed text arrays (like DSP Bus chat messages), state array updates trigger full-list re-renders on every streamed chunk. If there are N messages and a message receives M chunks, it triggers N * M renders, leading to O(N^2) main-thread blocking and noticeable lag during generation.
**Action:** Always wrap individual list items in `React.memo()` when dealing with continuously streaming state updates in arrays. This limits the re-render strictly to the single item that is currently receiving new data chunks.
