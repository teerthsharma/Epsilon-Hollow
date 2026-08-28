
## 2024-05-24 - Zustand Destructuring Root App Re-renders
**Learning:** Destructuring the full Zustand store (e.g., `const { a, b } = useStore()`) inside the root `<App />` component is a critical performance bottleneck because it subscribes the root to ALL state changes, causing the entire component tree to unnecessarily re-render on every state update (like logging).
**Action:** Always extract individual Zustand selectors (`const a = useStore(s => s.a)`) to maintain tight component subscriptions and prevent expensive cascading renders.
