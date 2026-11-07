

## 2023-10-25 - Icon Button Focus Accessibility
**Learning:** Icon-only utility buttons in the governor UI lacked both semantic labels and keyboard focus indicators, making them completely inaccessible to keyboard and screen reader users.
**Action:** Ensure all icon buttons include explicit `title` (for mouse), `aria-label` (for screen readers), and `focus-visible` styling (matching hover context, e.g., `gov-error` for destructive actions) for keyboard navigation.

## 2024-11-07 - Accessible Drop Zones
**Learning:** The `div` drop zone pattern in this app's components (like SampleBay) lacks built-in keyboard accessibility, meaning keyboard-only and screen reader users cannot trigger the import file dialog.
**Action:** When converting non-interactive `div` elements into interactive drop zones, always add `role="button"`, `tabIndex={0}`, an `onKeyDown` handler (for Enter/Space), a descriptive `aria-label`, and `focus-visible` styling to ensure parity with mouse interactions.
