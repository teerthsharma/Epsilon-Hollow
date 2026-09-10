

## 2024-10-24 - Console Clear Button Accessibility
**Learning:** Icon-only utility buttons in panel headers (like the Console clear button) represent destructive actions but lack proper focus styling and screen reader labels. Using red (`gov-error`) for the destructive focus and hover states provides better context than generic accent colors.
**Action:** Added `aria-label`, `title`, and specific `focus-visible` ring styling using `gov-error` to the Trash icon button in the Console panel to improve both accessibility and visual feedback for keyboard users.
