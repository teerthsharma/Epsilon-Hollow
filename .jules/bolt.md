## 2026-06-01 - Prevented O(N^2) Rendering in Streaming Components
**Learning:** React state arrays that update frequently (like appending chunks to streaming text) cause full list re-renders. This is an O(N^2) performance bottleneck, especially in long chat threads.
**Action:** Always wrap list items in `React.memo()` when rendering frequently updated arrays, especially in streaming applications like DSP Bus chat messages.

## 2026-06-26 - Prevent O(N^2) Rendering Caused by Inline Date Computation
**Learning:** Generating dynamic values like `new Date().toLocaleTimeString()` inline within a mapped React list prevents `React.memo()` from working properly, and forces recalculation on every render (causing O(N^2) bottlenecks as arrays grow).
**Action:** When extracting components to be memoized, ensure dynamic generation (like timestamps) is captured in local state or the data model upstream, passing static values down as props.
