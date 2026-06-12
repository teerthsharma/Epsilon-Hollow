## 2024-06-12 - Prevent O(N^2) render bottlenecks in streaming chat
**Learning:** Streaming text arrays (like DSP Bus chat messages) that update frequently will cause O(N^2) rendering bottlenecks if individual list items aren't memoized. A state array update causes the full list to re-render.
**Action:** Extract list items into standalone functional components and wrap them in `React.memo(function ComponentName() {...})` to ensure only the newly added/changed items render, preserving performance during high-frequency streaming events.
