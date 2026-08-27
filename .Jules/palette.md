
## 2024-08-27 - Console Panel Clear Button A11y
**Learning:** Icon-only utility buttons in the UI (like the clear button in the console panel) lacked `aria-label` attributes for screen readers and visible focus indicators for keyboard users. Adding a `title` provides a tooltip, `aria-label` supports screen readers, and `focus-visible` ring styling enables clear keyboard navigation.
**Action:** When implementing icon-only buttons, consistently apply `aria-label`, `title`, and `focus-visible` Tailwind classes (like `focus-visible:ring-2 focus-visible:ring-gov-accent focus-visible:outline-none`) to ensure full accessibility.
