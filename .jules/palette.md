

## 2023-10-25 - Icon Button Focus Accessibility
**Learning:** Icon-only utility buttons in the governor UI lacked both semantic labels and keyboard focus indicators, making them completely inaccessible to keyboard and screen reader users.
**Action:** Ensure all icon buttons include explicit `title` (for mouse), `aria-label` (for screen readers), and `focus-visible` styling (matching hover context, e.g., `gov-error` for destructive actions) for keyboard navigation.


## 2024-05-20 - Keyboard Accessible Dropzones
**Learning:** Interactive elements like file dropzones built with standard `div` tags lack built-in accessibility semantics, completely excluding keyboard users.
**Action:** Convert interactive `div`s into accessible controls by adding `role="button"`, `tabIndex={0}`, an `onKeyDown` handler (for `Enter` and `Space`), and `focus-visible` styling for visual feedback.
