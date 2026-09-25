
## 2024-05-18 - Zustand Full Store Anti-pattern
**Learning:** Found a widespread anti-pattern in the React app where components were destructuring the full Zustand store (e.g., `const { a, b } = useStore();`). This breaks the automatic optimization and causes the component to re-render whenever *any* state in the store changes, even unrelated state.
**Action:** Always extract Zustand selectors individually (e.g., `const a = useStore(s => s.a);`) to guarantee components only re-render when their specific dependencies change.
