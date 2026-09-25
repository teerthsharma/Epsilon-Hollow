
## 2024-05-24 - Zustand Full-Store Subscriptions
**Learning:** Found a severe anti-pattern in the codebase: destructuring from the full Zustand store in the root `App.tsx` component. This causes the entire application component tree to re-render on *every* single state change (such as high-frequency log updates).
**Action:** Always extract individual selectors (e.g., `useStore(s => s.a)`) or use `useShallow`, particularly in root or high-level components, to strictly scope re-renders to only when the selected state changes.
