
## 2023-10-25 - Icon-only buttons lacking a11y context
**Learning:** Found an icon-only "Clear Console" button in the `ConsolePanel` that completely lacked `aria-label`, `title`, and visible keyboard focus styles, making it invisible to screen readers and difficult to use for keyboard-only users.
**Action:** Always add `aria-label`, a descriptive `title` (for mouse hover), and `focus-visible:ring-*` styles for keyboard navigation to any icon-only button to ensure full accessibility and context.
