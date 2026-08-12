## 2026-06-01 - Prevented O(N^2) Rendering in Streaming Components
**Learning:** React state arrays that update frequently (like appending chunks to streaming text) cause full list re-renders. This is an O(N^2) performance bottleneck, especially in long chat threads.
**Action:** Always wrap list items in `React.memo()` when rendering frequently updated arrays, especially in streaming applications like DSP Bus chat messages.

## 2026-07-08 - Extracted Input State from Chat Lists
**Learning:** In React chat applications, keeping input state (`input`, `setInput`) in the same component that renders a large array of messages causes the entire component (and potentially the list, if not perfectly memoized) to re-render on every single keystroke. This causes noticeable O(N) typing latency.
**Action:** Always extract the input field and its local state into a separate child component (e.g., `ChatInput`) to prevent the large message list from re-rendering until a message is actually sent.


## 2026-07-11 - Prevented O(N^2) Rendering in LiquidStream
**Learning:** In streaming chat applications like LiquidStream, rendering an entire array of messages directly within the parent component via `messages.map` causes full-list re-renders every time a new message (or message chunk) is added. This creates an O(N^2) performance bottleneck. Furthermore, recalculating dynamic values like timestamps (`new Date().toLocaleTimeString()`) inline causes hydration mismatches or recalculates values for older messages unexpectedly.
**Action:** Always extract message rendering into a dedicated child component (e.g., `MessageItem`) wrapped in `React.memo()`. Also, freeze dynamic values like timestamps by adding them to the message model state upon creation rather than calculating them during render. Fix hydration mismatches by wrapping initial state-setting logic within a `useEffect` and a `setTimeout(() => { ... }, 0)`.

## 2026-07-24 - Extracted Map Operations from Render Loop
**Learning:** Even if child components are memoized with `React.memo()`, a parent component mapping over a large array directly in its render function executes that O(N) mapping on every parent re-render. If the parent re-renders frequently due to unrelated state (e.g., streaming telemetry), the array mapping logic should be extracted and wrapped in a `useMemo` hook at the top level of the component to prevent unnecessary overhead.
**Action:** Always wrap array mapping logic in `useMemo` if the parent component frequently re-renders due to unrelated state.

## 2024-07-25 - React.memo Component Referential Equality
**Learning:** Even if a child component is wrapped in `React.memo` (like `LiquidInput`), if the parent passes an inline or standard function (like `handleSend`) as a prop, that function gets a new reference on every parent render. This defeats the `React.memo` because the props will shallow-compare as unequal, causing unnecessary re-renders of the memoized child.
**Action:** Always wrap event handlers passed as props to `React.memo` components in `useCallback` to preserve referential equality and actually realize the performance benefits of memoization.

## 2024-07-30 - Expensive Initialization in React

**Learning:** `useMemo` should not be used for expensive, one-time initialization if the logic contains impure functions (like `Math.random()`), as React can throw away memoized values or linting rules (like `react-hooks/purity`) will flag the impure function. The computation will rerun unexpectedly on re-renders, causing performance drops or UI flashes (e.g. regenerating 3D coordinates).
**Action:** Use a lazy `useState` initializer (`useState(() => expensiveInitialization())`) for data that must be calculated once and kept stable across all re-renders.

## 2026-08-04 - Fixed Atlas Proof Validation

**Learning:** When using `seal()` on an object to change its properties from read-write-execute to read-execute, if the seal step fails or is skipped, the memory protections checks inside CI/Parsers will fail (`wx=fail` instead of `wx=text_rx_data_rw_nx`).
**Action:** Always ensure that `image.seal()` is successfully called and validated during the initialization or grafting phase of chart objects.

## 2026-08-10 - Prevented O(N) Rendering with Zustand useShallow
**Learning:** In React applications using Zustand, destructuring from the full store (`const { a, b } = useStore()`) implicitly subscribes the component to the *entire* state object. This means any state update, even completely unrelated to `a` or `b`, will trigger a re-render of the component. In applications with many panels or frequent state updates, this creates a massive O(N) re-rendering bottleneck.
**Action:** Always use individual selectors (`useStore(s => s.a)`) or `useShallow` (`useStore(useShallow(s => ({ a: s.a })))`) to explicitly restrict the component's subscription to only the state slices it actually depends on, preventing unnecessary global re-renders.
