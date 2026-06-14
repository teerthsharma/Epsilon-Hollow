## 2024-05-01 - DSP Bus Streaming Rendering Bottleneck
**Learning:** The DSP Bus in this codebase streams messages in small chunks, rapidly updating the `messages` state array in `useApeiron.ts`. If message components are mapped inline, every chunk forces a full list re-render, leading to an O(N^2) performance degradation as the chat length increases.
**Action:** Always extract streamed list items into standalone components wrapped with `React.memo(function Name() {...})` to isolate renders to only the item currently receiving stream updates.
