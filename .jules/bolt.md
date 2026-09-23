
## 2024-05-24 - Zustand Root Destructuring Anti-pattern
**Learning:** Destructuring the full Zustand store (`const { a, b } = useStore()`) in a high-level component like App.tsx causes the entire application tree to re-render on any unrelated store update (e.g., when new logs are added).
**Action:** Always use individual selectors (`useStore(s => s.a)`) or `useShallow` in React components to subscribe only to necessary state slices.
