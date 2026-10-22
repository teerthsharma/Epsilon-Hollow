

## 2023-10-25 - Icon Button Focus Accessibility
**Learning:** Icon-only utility buttons in the governor UI lacked both semantic labels and keyboard focus indicators, making them completely inaccessible to keyboard and screen reader users.
**Action:** Ensure all icon buttons include explicit `title` (for mouse), `aria-label` (for screen readers), and `focus-visible` styling (matching hover context, e.g., `gov-error` for destructive actions) for keyboard navigation.

## 2024-05-18 - Drop Zone Keyboard Accessibility
**Learning:** The file drop zone `div` had a click handler (`onClick={handleImportClick}`) but lacked keyboard interaction and semantic roles, making the import action inaccessible via keyboard navigation.
**Action:** Always add `role="button"`, `tabIndex={0}`, an `onKeyDown` handler (for Space/Enter), and `focus-visible` styling to `div` elements that act as interactive buttons.
