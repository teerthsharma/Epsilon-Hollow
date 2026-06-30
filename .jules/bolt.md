## 2026-06-01 - Prevented O(N^2) Rendering in Streaming Components
**Learning:** React state arrays that update frequently (like appending chunks to streaming text) cause full list re-renders. This is an O(N^2) performance bottleneck, especially in long chat threads.
**Action:** Always wrap list items in `React.memo()` when rendering frequently updated arrays, especially in streaming applications like DSP Bus chat messages.

## 2024-06-30 - Freezing Dynamic Timestamps in Memoized Components
**Learning:** When extracting inline rendering logic (like chat messages) into a `React.memo()` component to prevent O(N^2) render bottlenecks, dynamic inline functions (like `new Date().toLocaleTimeString()`) become problematic. If left inline, they cause bugs by recalculating the timestamp for *all* messages whenever the list is forced to re-render, destroying historical accuracy.
**Action:** When memoizing list items containing temporal or dynamic data, always update the underlying data model (e.g., adding a `timestamp` property to the `Message` type) to freeze the value at creation time and pass it as a prop, ensuring both render purity and historical correctness.
