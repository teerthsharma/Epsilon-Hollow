## 2026-06-01 - Prevented O(N^2) Rendering in Streaming Components
**Learning:** React state arrays that update frequently (like appending chunks to streaming text) cause full list re-renders. This is an O(N^2) performance bottleneck, especially in long chat threads.
**Action:** Always wrap list items in `React.memo()` when rendering frequently updated arrays, especially in streaming applications like DSP Bus chat messages.

## 2024-06-28 - Moving Dynamic Values to State for Memoization
**Learning:** Inline dynamic values like `new Date().toLocaleTimeString()` will cause `React.memo()` components to show incorrect/updated values on re-renders if evaluated during render. Extracting them to the data model (state) ensures they are calculated once and correctly frozen upon creation.
**Action:** When extracting inline message rendering into memoized list item components, ensure dynamic values are captured in local state or the item's data model so they are correctly frozen upon creation and not recalculated across renders.
