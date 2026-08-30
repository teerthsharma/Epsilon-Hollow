
## 2024-10-27 - Missing Accessible Labels and Focus Rings on Icon-Only Buttons
**Learning:** The laamba-governor application frequently uses icon-only utility buttons (e.g., in ConsolePanel, FormulaEditor, SampleBay) that lack explicit `aria-label` or `title` attributes, making them inaccessible to screen readers and difficult for mouse users. They also lack `focus-visible` rings, hindering keyboard navigation.
**Action:** When implementing or fixing icon-only buttons, always ensure they have an explicit `title` (for mouse hover), `aria-label` (for screen readers), and clear `focus-visible` styling (like `focus-visible:ring-2 focus-visible:ring-gov-accent focus-visible:outline-none rounded`). Use `gov-error` instead for destructive actions.
