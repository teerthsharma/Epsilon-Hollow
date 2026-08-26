
## 2024-10-24 - Zustand Full-Store Destructuring Anti-pattern
**Learning:** Destructuring the full store directly from `useStore()` (e.g., `const { a, b } = useStore()`) in components subscribes them to every single state update across the entire app, triggering severe and unnecessary full-tree re-renders (especially when telemetry/logs update at high frequency).
**Action:** Always extract individual selectors explicitly (e.g., `useStore(s => s.a)` or use `useShallow`) to ensure components only re-render when their specific data dependencies change.
