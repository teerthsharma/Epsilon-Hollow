
## 2024-08-28 - Icon-Only Button Accessibility
**Learning:** Found an icon-only button without an ARIA label or title in `ConsolePanel.tsx` (the "Clear logs" Trash2 button). Icon-only buttons must have `title` and `aria-label` for screen reader and hover accessibility. They should also have focus styles for keyboard users.
**Action:** Add `aria-label`, `title`, and `focus-visible` styles to the clear logs button.


## 2024-08-28 - Icon-Only Button Accessibility Pattern
**Learning:** Found multiple icon-only buttons across components (`SampleBay.tsx`, `ConsolePanel.tsx`) missing proper accessibility attributes (`aria-label`, focus styles). This seems to be a recurring pattern in the UI.
**Action:** Will add `aria-label`, `title`, and `focus-visible` to one of these as a starting point, specifically the one in `ConsolePanel.tsx` since it's a global action. I'll also add it to `SampleBay.tsx` to clear it out.
