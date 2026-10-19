

## 2023-10-25 - Icon Button Focus Accessibility
**Learning:** Icon-only utility buttons in the governor UI lacked both semantic labels and keyboard focus indicators, making them completely inaccessible to keyboard and screen reader users.
**Action:** Ensure all icon buttons include explicit `title` (for mouse), `aria-label` (for screen readers), and `focus-visible` styling (matching hover context, e.g., `gov-error` for destructive actions) for keyboard navigation.

## 2024-10-19 - Keyboard Accessible Drop Zones
**Learning:** In React applications, converting non-interactive elements like `div` drop zones into fully keyboard-accessible interactive elements requires not only `tabIndex` and `role`, but an explicit `onKeyDown` handler that intercepts 'Enter' and 'Space' and calls `e.preventDefault()` to prevent unwanted page scrolling before triggering the action. Standard focus indicators must also be applied.
**Action:** When creating custom interactive areas like drop zones from generic containers, always implement the full suite of keyboard accessibility attributes (`role="button"`, `tabIndex={0}`, `onKeyDown`, `aria-label`, `focus-visible`).
