

## 2023-10-25 - Icon Button Focus Accessibility
**Learning:** Icon-only utility buttons in the governor UI lacked both semantic labels and keyboard focus indicators, making them completely inaccessible to keyboard and screen reader users.
**Action:** Ensure all icon buttons include explicit `title` (for mouse), `aria-label` (for screen readers), and `focus-visible` styling (matching hover context, e.g., `gov-error` for destructive actions) for keyboard navigation.

## 2023-11-02 - Drop Zone Keyboard Accessibility
**Learning:** Div-based drag-and-drop zones often function as clickable upload triggers but lack native keyboard interactivity, excluding keyboard and screen reader users from accessing core import functionality.
**Action:** When using a non-interactive element as a dual-purpose drop zone and click target, explicitly provide button semantics (`role="button"`, `tabIndex={0}`), an `onKeyDown` handler for Space/Enter (with `e.preventDefault()`), an `aria-label`, and `focus-visible` styling.
