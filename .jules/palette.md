## 2024-05-15 - Initial
**Learning:** Initializing palette journal
**Action:** None

## 2024-05-15 - Missing ARIA Labels on Icon-only Buttons
**Learning:** Found a widespread pattern of icon-only utility buttons (e.g. clear logs, close window, import) missing both `aria-label` and `focus-visible` styling, harming screen reader and keyboard accessibility.
**Action:** Added explicit `aria-label` attributes explaining the button's action and standard Tailwind `focus-visible` classes to ensure visible focus states without custom CSS.
