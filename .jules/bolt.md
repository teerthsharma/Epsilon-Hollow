

## 2024-05-24 - Zustand Full Store Subscription Anti-Pattern
**Learning:** Destructuring directly from `useStore()` (e.g., `const { a, b } = useStore()`) in Zustand causes the component to subscribe to the entire store state. In high-level root components like `App.tsx`, this is a massive performance bottleneck as any unrelated update (like appending a log) forces the entire application tree to re-render.
**Action:** Always use individual selectors (e.g., `const a = useStore(s => s.a)`) to extract state slices, especially in root components, to strictly limit re-renders.
