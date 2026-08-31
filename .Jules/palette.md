
## 2023-11-09 - Accessible Icon Buttons
**Learning:** Icon-only utility buttons using `lucide-react` (like the console clear button) lack accessible names for screen readers and tooltips for mouse users, violating WCAG.
**Action:** When working on UI panels with icon-only utility buttons, ensure each button includes an explicit `title` for mouse users, `aria-label` for screen readers, and clear `focus-visible` styling (e.g., `focus-visible:outline-none focus-visible:ring-*`) for keyboard navigation.
