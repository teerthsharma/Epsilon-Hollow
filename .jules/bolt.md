## 2026-06-01 - Prevented O(N^2) Rendering in Streaming Components
**Learning:** React state arrays that update frequently (like appending chunks to streaming text) cause full list re-renders. This is an O(N^2) performance bottleneck, especially in long chat threads.
**Action:** Always wrap list items in `React.memo()` when rendering frequently updated arrays, especially in streaming applications like DSP Bus chat messages.
## 2026-06-01 - Prevented O(N) Date Recalculations in Render Array
**Learning:** Calling `new Date().toLocaleTimeString()` directly within an array mapping function inside a React render causes every list item to recalculate its timestamp on every render, leading to O(N) recalculations and making historical messages incorrectly show the current time.
**Action:** When rendering message streams, wrap the message item in `React.memo()` and capture the timestamp on mount using lazy state initialization: `const [timestamp] = useState(() => new Date().toLocaleTimeString());` to freeze it and prevent recalculations.
