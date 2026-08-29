## 2024-05-18 - Icon Button Accessibility
**Learning:** Icon-only buttons lacking ARIA labels, titles, and keyboard focus indicators (`focus-visible`) are a common accessibility issue. Adding `aria-label`, `title`, and `focus-visible:ring-*` styles ensures screen readers can announce the action and keyboard users can see where focus is.
**Action:** Always add explicit `aria-label`, `title`, and `focus-visible` styling (e.g., `focus-visible:outline-none focus-visible:ring-1`) to all icon-only buttons.
