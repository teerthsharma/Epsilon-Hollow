

## 2023-10-25 - Icon Button Focus Accessibility
**Learning:** Icon-only utility buttons in the governor UI lacked both semantic labels and keyboard focus indicators, making them completely inaccessible to keyboard and screen reader users.
**Action:** Ensure all icon buttons include explicit `title` (for mouse), `aria-label` (for screen readers), and `focus-visible` styling (matching hover context, e.g., `gov-error` for destructive actions) for keyboard navigation.

## 2024-05-24 - Interactive Drop Zones Need Keyboard Support
**Learning:** In the `laamba-governor` UI, interactive elements like file drop zones are sometimes built using generic `div` elements with `onClick` handlers, which completely breaks keyboard accessibility for users relying on Tab navigation.
**Action:** When converting a non-interactive element (like a `div` drop zone) into a fully keyboard-accessible interactive element, ensure it includes `role="button"`, `tabIndex={0}`, an `onKeyDown` handler to trigger the action on `Enter` or ` ` (Space) (calling `e.preventDefault()` to stop page scrolling), a descriptive `aria-label`, and proper `focus-visible` styling using existing design system classes.
