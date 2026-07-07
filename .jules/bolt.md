## 2026-06-01 - Prevented O(N^2) Rendering in Streaming Components
**Learning:** React state arrays that update frequently (like appending chunks to streaming text) cause full list re-renders. This is an O(N^2) performance bottleneck, especially in long chat threads.
**Action:** Always wrap list items in `React.memo()` when rendering frequently updated arrays, especially in streaming applications like DSP Bus chat messages.

## 2026-07-04 - Freezing Timestamps in Dynamic Arrays
**Learning:** Generating dynamic timestamps inside render functions for array items (e.g., `new Date().toLocaleTimeString()`) causes all items in the array to update to the current time whenever the parent component re-renders (like during text input).
**Action:** When adding timestamps to dynamically growing lists like chat messages, always generate and freeze the timestamp in the item's data model at creation time, rather than calling the date function within the render loop.
