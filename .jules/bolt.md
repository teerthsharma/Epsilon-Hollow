
## 2024-10-24 - Zustand Root Re-render Bottleneck
**Learning:** Destructuring `useStore()` in the root `App.tsx` component forces the entire React tree to re-render on every state change (including frequent log updates).
**Action:** Always use individual Zustand selectors (`useStore(s => s.property)`) instead of object destructuring, especially in high-level components.
