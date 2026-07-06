## 2026-06-01 - Prevented O(N^2) Rendering in Streaming Components
**Learning:** React state arrays that update frequently (like appending chunks to streaming text) cause full list re-renders. This is an O(N^2) performance bottleneck, especially in long chat threads.
**Action:** Always wrap list items in `React.memo()` when rendering frequently updated arrays, especially in streaming applications like DSP Bus chat messages.

## 2024-07-06 - Prevent O(N) Re-rendering during typing
**Learning:** In React chat applications, keeping input state (`input`, `setInput`) in the same top-level component that renders a large array of messages causes the entire component to re-render on every single keystroke, causing O(N) typing latency.
**Action:** Extract the input field and its local state into a separate child component (e.g., `ChatInput`) to prevent the large message list from re-rendering until a message is actually sent.
