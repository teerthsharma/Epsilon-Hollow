

## 2023-10-25 - Icon Button Focus Accessibility
**Learning:** Icon-only utility buttons in the governor UI lacked both semantic labels and keyboard focus indicators, making them completely inaccessible to keyboard and screen reader users.
**Action:** Ensure all icon buttons include explicit `title` (for mouse), `aria-label` (for screen readers), and `focus-visible` styling (matching hover context, e.g., `gov-error` for destructive actions) for keyboard navigation.


## 2024-05-24 - Drop Zone Accessibility
**Learning:** Found a common pattern where `div` elements used as drop zones or file upload triggers via `onClick` lack keyboard accessibility. Non-interactive elements require semantic roles (`role="button"`), focus management (`tabIndex={0}`), and keyboard event handlers (`onKeyDown` for Enter/Space) to be fully usable by keyboard-only users.
**Action:** When creating or reviewing custom interactive components built with `div` or `span` (like drop zones), always convert them to behave like native buttons (role, tabIndex, onKeyDown, focus-visible).
