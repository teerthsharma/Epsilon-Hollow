

## 2023-10-25 - Icon Button Focus Accessibility
**Learning:** Icon-only utility buttons in the governor UI lacked both semantic labels and keyboard focus indicators, making them completely inaccessible to keyboard and screen reader users.
**Action:** Ensure all icon buttons include explicit `title` (for mouse), `aria-label` (for screen readers), and `focus-visible` styling (matching hover context, e.g., `gov-error` for destructive actions) for keyboard navigation.

## 2023-11-10 - Div as Button Accessibility
**Learning:** In the SampleBay panel, a non-interactive `div` used as a drop zone was missing keyboard support and ARIA semantics, making it inaccessible for keyboard and screen reader users to trigger the fallback file import dialog.
**Action:** Always add `role="button"`, `tabIndex={0}`, an `onKeyDown` handler for `Enter` and ` ` (Space) (with `e.preventDefault()`), `aria-label`, and `focus-visible` styling when making non-interactive elements interactive.
