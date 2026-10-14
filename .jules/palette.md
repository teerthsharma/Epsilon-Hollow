

## 2023-10-25 - Icon Button Focus Accessibility
**Learning:** Icon-only utility buttons in the governor UI lacked both semantic labels and keyboard focus indicators, making them completely inaccessible to keyboard and screen reader users.
**Action:** Ensure all icon buttons include explicit `title` (for mouse), `aria-label` (for screen readers), and `focus-visible` styling (matching hover context, e.g., `gov-error` for destructive actions) for keyboard navigation.

## 2024-05-18 - Div-based Interactive Elements
**Learning:** The application uses div elements for complex interactions like drag-and-drop file import zones, which initially lack keyboard accessibility, preventing non-mouse users from accessing core functionality.
**Action:** When utilizing divs as interactive elements, always implement `role="button"`, `tabIndex={0}`, keyboard event handlers (`onKeyDown` for Space/Enter), explicit `aria-label`s, and `focus-visible` styling to match standard button accessibility.
