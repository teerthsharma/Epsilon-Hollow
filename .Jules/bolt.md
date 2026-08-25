
## 2024-10-25 - Prevent full app re-renders from Zustand destructuring
**Learning:** Destructuring variables from a Zustand store (e.g., `const { a, b } = useStore()`) in a root component like `App.tsx` forces the component to subscribe to the entire store, causing full application re-renders whenever ANY state changes.
**Action:** Always use individual selectors (e.g., `const a = useStore(s => s.a)`) or `useShallow` to limit re-renders to when the selected state changes.
