
## 2024-10-19 - O(N) Array Rendering in Zustand Subscriptions
**Learning:** Even with Zustand's `usePick` or `useShallow` to limit re-renders from state changes, mapping over an ever-growing array of primitives (like strings) inside a component will cause all existing DOM nodes in that list to re-render when a new item is appended, leading to an O(N) rendering bottleneck.
**Action:** Always extract items in frequently appended lists into their own `React.memo` wrapped components so that only the newly appended items trigger a render, achieving O(1) performance for additions.
