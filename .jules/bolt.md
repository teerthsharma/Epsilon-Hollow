## 2024-09-17 - Unnecessary re-renders from Zustand store destructuring
**Learning:** Destructuring the full Zustand store (e.g., `const { a, b } = useStore()`) subscribes the component to ALL store changes, causing massive re-render trees, especially in root components like App.tsx.
**Action:** Always use individual selectors or `useShallow` when using Zustand to prevent unnecessary re-renders. Individual selectors (`const a = useStore(s => s.a)`) are preferred for better TypeScript inference.
