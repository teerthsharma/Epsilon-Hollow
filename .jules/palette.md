

## 2023-10-25 - Icon Button Focus Accessibility
**Learning:** Icon-only utility buttons in the governor UI lacked both semantic labels and keyboard focus indicators, making them completely inaccessible to keyboard and screen reader users.
**Action:** Ensure all icon buttons include explicit `title` (for mouse), `aria-label` (for screen readers), and `focus-visible` styling (matching hover context, e.g., `gov-error` for destructive actions) for keyboard navigation.

## 2024-11-09 - Accessible Drop Zones
**Learning:** Pure `div` drag-and-drop zones are inaccessible to keyboard-only users who cannot use a mouse to drag files or click the container.
**Action:** When a drop zone also supports clicking to import (via file dialog), convert it to a button-like element with `role="button"`, `tabIndex={0}`, an `onKeyDown` handler (Enter/Space), an `aria-label`, and `focus-visible` styling to allow keyboard activation.
