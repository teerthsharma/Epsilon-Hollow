

## 2023-10-25 - Icon Button Focus Accessibility
**Learning:** Icon-only utility buttons in the governor UI lacked both semantic labels and keyboard focus indicators, making them completely inaccessible to keyboard and screen reader users.
**Action:** Ensure all icon buttons include explicit `title` (for mouse), `aria-label` (for screen readers), and `focus-visible` styling (matching hover context, e.g., `gov-error` for destructive actions) for keyboard navigation.

## 2024-10-07 - Make Drop Zones Keyboard Accessible
**Learning:** In laamba-governor, click-to-upload drop zones implemented as `div` elements exclude keyboard users, while standard buttons natively handle focus and activation.
**Action:** When creating click-to-upload areas that also accept drops, explicitly add `role="button"`, `tabIndex={0}`, `onKeyDown` handling for Space/Enter (with `e.preventDefault()`), and `focus-visible` classes to ensure parity with standard buttons.
