## 2024-05-14 - React List Rendering Bottleneck with Streamed Chunks
**Learning:** When rendering streamed text arrays (like DSP Bus chat messages), state array updates trigger full-list re-renders on every streamed chunk, causing O(N^2) rendering bottlenecks.
**Action:** Wrap individual list items in `React.memo()` to prevent unnecessary re-renders of the entire list when only the newest streamed chunk is updating.
