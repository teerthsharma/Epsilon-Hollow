## 2026-06-01 - Prevented O(N^2) Rendering in Streaming Components
**Learning:** React state arrays that update frequently (like appending chunks to streaming text) cause full list re-renders. This is an O(N^2) performance bottleneck, especially in long chat threads.
**Action:** Always wrap list items in `React.memo()` when rendering frequently updated arrays, especially in streaming applications like DSP Bus chat messages.


## 2026-06-02 - Capture dynamic values in state for memoized items
**Learning:** When extracting inline message rendering into memoized list item components, ensure dynamic values like `new Date().toLocaleTimeString()` are captured in local state or the item's data model so they are correctly frozen upon creation and not recalculated across renders.
**Action:** Always capture timestamp and other dynamic values in the data model for `React.memo()` list items.
