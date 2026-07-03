## 2026-06-01 - Prevented O(N^2) Rendering in Streaming Components
**Learning:** React state arrays that update frequently (like appending chunks to streaming text) cause full list re-renders. This is an O(N^2) performance bottleneck, especially in long chat threads.
**Action:** Always wrap list items in `React.memo()` when rendering frequently updated arrays, especially in streaming applications like DSP Bus chat messages.

## 2024-07-04 - Memoizing List Items and Capturing Dynamic Values at Creation
**Learning:** Using inline dynamic values like `new Date().toLocaleTimeString()` inside non-memoized mapped lists is a major performance footgun. When typing in a sibling input field, the parent component re-renders. This causes every message in the list to re-render, forcing `new Date()` to recalculate. This not only causes O(N^2) render bottlenecks as the list grows but also fundamentally changes the timestamp of past messages.
**Action:** Always extract complex list items into memoized components (`React.memo`). Additionally, capture dynamic temporal values (like timestamps) at the time of creation (e.g., in `handleSend`) and store them in the item's data model, rather than calculating them dynamically during render.
