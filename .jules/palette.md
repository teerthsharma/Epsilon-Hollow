

## 2023-10-25 - Icon Button Focus Accessibility
**Learning:** Icon-only utility buttons in the governor UI lacked both semantic labels and keyboard focus indicators, making them completely inaccessible to keyboard and screen reader users.
**Action:** Ensure all icon buttons include explicit `title` (for mouse), `aria-label` (for screen readers), and `focus-visible` styling (matching hover context, e.g., `gov-error` for destructive actions) for keyboard navigation.


## 2024-05-15 - Interactive Div Keyboard Accessibility
**Learning:** Interactive `div` elements used as drop zones or custom buttons lack native keyboard support and semantic meaning, preventing keyboard-only and screen reader users from accessing them.
**Action:** When converting non-interactive elements (like `div`) to actionable ones, always add `role="button"`, `tabIndex={0}`, an `onKeyDown` handler (for `Enter` and Space), a descriptive `aria-label`, and `focus-visible` styling.
