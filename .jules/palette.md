

## 2023-10-25 - Icon Button Focus Accessibility
**Learning:** Icon-only utility buttons in the governor UI lacked both semantic labels and keyboard focus indicators, making them completely inaccessible to keyboard and screen reader users.
**Action:** Ensure all icon buttons include explicit `title` (for mouse), `aria-label` (for screen readers), and `focus-visible` styling (matching hover context, e.g., `gov-error` for destructive actions) for keyboard navigation.

## 2024-05-14 - Interactive Element Focus Accessibility
**Learning:** Many icon-only utility buttons across panels (like SampleBay, EngineRack, and FormulaEditor) lacked standard keyboard focus indicators, making the app hard to navigate with a keyboard.
**Action:** Applied Tailwind `focus-visible` styles (`focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-*`) consistently to all interactive elements to ensure clear visual feedback for keyboard users.
