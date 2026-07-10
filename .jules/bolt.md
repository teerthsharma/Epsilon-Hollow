## 2026-06-01 - Prevented O(N^2) Rendering in Streaming Components
**Learning:** React state arrays that update frequently (like appending chunks to streaming text) cause full list re-renders. This is an O(N^2) performance bottleneck, especially in long chat threads.
**Action:** Always wrap list items in `React.memo()` when rendering frequently updated arrays, especially in streaming applications like DSP Bus chat messages.

## 2026-07-08 - Extracted Input State from Chat Lists
**Learning:** In React chat applications, keeping input state (`input`, `setInput`) in the same component that renders a large array of messages causes the entire component (and potentially the list, if not perfectly memoized) to re-render on every single keystroke. This causes noticeable O(N) typing latency.
**Action:** Always extract the input field and its local state into a separate child component (e.g., `ChatInput`) to prevent the large message list from re-rendering until a message is actually sent.

## $(date +%Y-%m-%d) - Prevented O(N^2) Rendering in LiquidStream
**Learning:** In React, mapping over arrays to render inline JSX blocks (like message histories) causes the entire list to re-render whenever the array updates (e.g., when a new message is added). Furthermore, if functions like `new Date().toLocaleTimeString()` are called directly within the render loop, every item in the list will re-calculate that function on every render, leading to dynamic values (like timestamps) unintentionally overwriting older elements' data.
**Action:** Always extract inline list items into dedicated components wrapped in `React.memo()`. For dynamic creation-time values (like timestamps), store them in the extracted component's local state (`useState(() => new Date().toLocaleTimeString())`) so they are correctly frozen upon creation and not recalculated across renders.
