

## 2023-10-25 - Icon Button Focus Accessibility
**Learning:** Icon-only utility buttons in the governor UI lacked both semantic labels and keyboard focus indicators, making them completely inaccessible to keyboard and screen reader users.
**Action:** Ensure all icon buttons include explicit `title` (for mouse), `aria-label` (for screen readers), and `focus-visible` styling (matching hover context, e.g., `gov-error` for destructive actions) for keyboard navigation.


## 2024-10-12 - Keyboard Accessible Drop Zones
**Learning:** In laamba-governor, file drop zones implemented as `div` elements lack keyboard navigability and interactivity for non-mouse users, which is a key accessibility requirement.
**Action:** Always add `role="button"`, `tabIndex={0}`, an `onKeyDown` handler (Enter/Space), a descriptive `aria-label`, and standard `focus-visible` utility classes when converting static `div` drop zones into interactive elements.
