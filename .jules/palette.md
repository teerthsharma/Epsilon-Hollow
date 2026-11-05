

## 2023-10-25 - Icon Button Focus Accessibility
**Learning:** Icon-only utility buttons in the governor UI lacked both semantic labels and keyboard focus indicators, making them completely inaccessible to keyboard and screen reader users.
**Action:** Ensure all icon buttons include explicit `title` (for mouse), `aria-label` (for screen readers), and `focus-visible` styling (matching hover context, e.g., `gov-error` for destructive actions) for keyboard navigation.

## 2024-11-05 - Accessible Drop Zones
**Learning:** In Laamba Governor, transforming interactive `div` elements (like the SampleBay file drop zone) into properly accessible buttons requires a full suite of attributes: `role="button"`, `tabIndex={0}`, an `onKeyDown` handler (intercepting Enter/Space to prevent scrolling and trigger the action), and `focus-visible` outline styles, to ensure screen reader compatibility and keyboard navigation parity.
**Action:** Always map generic clickable wrappers to semantic interactive states, ensuring they can receive focus and be activated completely independently of mouse input.
