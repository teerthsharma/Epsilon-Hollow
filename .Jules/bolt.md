
## 2024-10-26 - Zustand useShallow
**Learning:** Destructuring from the full Zustand store (e.g., `const { a, b } = useStore()`) implicitly subscribes the component to the entire store, causing unnecessary re-renders on ANY state change. This is especially detrimental in components with heavy WebGL/Canvas rendering like `TopologyScope.tsx`.
**Action:** Always wrap the selected state in `useShallow` from `zustand/react/shallow` when extracting multiple properties from a Zustand store (e.g., `const { a, b } = useStore(useShallow(state => ({ a: state.a, b: state.b })))`).
