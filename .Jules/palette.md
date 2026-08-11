
## 2024-08-09 - Accessible Icon-Only Buttons
**Learning:** Icon-only buttons (like `Trash2` for clearing consoles) often lack accessible names (ARIA labels) and hover tooltips (`title`), making them invisible to screen readers and ambiguous to users who don't recognize the icon. Additionally, they often lack explicit keyboard focus states (`focus-visible:outline-none focus-visible:ring-*`).
**Action:** When working on UI panels with icon-only utility buttons, ensure each button includes an explicit `title` for mouse users, `aria-label` for screen readers, and clear `focus-visible` styling for keyboard navigation.
