

## 2023-10-25 - Icon Button Focus Accessibility
**Learning:** Icon-only utility buttons in the governor UI lacked both semantic labels and keyboard focus indicators, making them completely inaccessible to keyboard and screen reader users.
**Action:** Ensure all icon buttons include explicit `title` (for mouse), `aria-label` (for screen readers), and `focus-visible` styling (matching hover context, e.g., `gov-error` for destructive actions) for keyboard navigation.

## 2024-05-15 - Interactive Drop Zones Accessibility
**Learning:** In React applications, using `div` elements for drag-and-drop zones often neglects keyboard and screen reader users who cannot utilize mouse interactions to trigger fallback actions.
**Action:** Consistently convert non-interactive `div` drop zones into accessible buttons by adding `role="button"`, `tabIndex={0}`, an `onKeyDown` handler (for `Enter` and Space), a descriptive `aria-label`, and `focus-visible` styling for keyboard navigation.
