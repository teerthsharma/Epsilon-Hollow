## 2026-06-01 - Prevented O(N^2) Rendering in Streaming Components
**Learning:** React state arrays that update frequently (like appending chunks to streaming text) cause full list re-renders. This is an O(N^2) performance bottleneck, especially in long chat threads.
**Action:** Always wrap list items in `React.memo()` when rendering frequently updated arrays, especially in streaming applications like DSP Bus chat messages.

## 2026-07-08 - Extracted Input State from Chat Lists
**Learning:** In React chat applications, keeping input state (`input`, `setInput`) in the same component that renders a large array of messages causes the entire component (and potentially the list, if not perfectly memoized) to re-render on every single keystroke. This causes noticeable O(N) typing latency.
**Action:** Always extract the input field and its local state into a separate child component (e.g., `ChatInput`) to prevent the large message list from re-rendering until a message is actually sent.


## 2026-07-09 - Stabilize Dynamic Render Values with Memoization
**Learning:** When using React.memo() to optimize lists (like chat streams), dynamic values rendered inside the items (e.g., `new Date().toLocaleTimeString()`) will be frozen at their initial render time. If not captured in the item's data model, this causes incorrect UI behavior (all messages showing the same timestamp if the component re-renders) or forces the memo to invalidate if passed as a prop from a higher-level recalculation.
**Action:** Always capture dynamic values like timestamps directly into the item's data model state upon creation, rather than calculating them during the render phase of the item component.
