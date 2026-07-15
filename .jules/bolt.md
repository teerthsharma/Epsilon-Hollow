## 2026-06-01 - Prevented O(N^2) Rendering in Streaming Components
**Learning:** React state arrays that update frequently (like appending chunks to streaming text) cause full list re-renders. This is an O(N^2) performance bottleneck, especially in long chat threads.
**Action:** Always wrap list items in `React.memo()` when rendering frequently updated arrays, especially in streaming applications like DSP Bus chat messages.

## 2026-07-08 - Extracted Input State from Chat Lists
**Learning:** In React chat applications, keeping input state (`input`, `setInput`) in the same component that renders a large array of messages causes the entire component (and potentially the list, if not perfectly memoized) to re-render on every single keystroke. This causes noticeable O(N) typing latency.
**Action:** Always extract the input field and its local state into a separate child component (e.g., `ChatInput`) to prevent the large message list from re-rendering until a message is actually sent.


## 2026-07-11 - Prevented O(N^2) Rendering in LiquidStream
**Learning:** In streaming chat applications like LiquidStream, rendering an entire array of messages directly within the parent component via `messages.map` causes full-list re-renders every time a new message (or message chunk) is added. This creates an O(N^2) performance bottleneck. Furthermore, recalculating dynamic values like timestamps (`new Date().toLocaleTimeString()`) inline causes hydration mismatches or recalculates values for older messages unexpectedly.
**Action:** Always extract message rendering into a dedicated child component (e.g., `MessageItem`) wrapped in `React.memo()`. Also, freeze dynamic values like timestamps by adding them to the message model state upon creation rather than calculating them during render. Fix hydration mismatches by wrapping initial state-setting logic within a `useEffect` and a `setTimeout(() => { ... }, 0)`.

## 2026-07-15 - Prevented O(N) Array Mapping Overhead on Frequent Parent Re-renders
**Learning:** Even if child list components (like `MessageItem`) are wrapped in `React.memo()`, if the parent component frequently re-renders due to unrelated state (like streaming telemetry logs or pulsing UI states), any inline `array.map()` calls in the parent's render function will execute on every single re-render, creating an O(N) performance bottleneck.
**Action:** Always wrap the entire `.map()` operation in a `useMemo` hook (e.g., `useMemo(() => array.map(...), [array])`) when rendering large arrays inside a parent that updates frequently, so the mapping operation is skipped entirely unless the array itself changes.
