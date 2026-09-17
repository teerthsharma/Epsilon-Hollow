

## 2024-05-24 - Accessible Icon Buttons
**Learning:** When implementing icon-only buttons like the trash icon in ConsolePanel, adding an explicit `aria-label` ensures screen reader compatibility, and `title` assists mouse users. Adding `focus-visible` styling is essential for keyboard navigation.
**Action:** Always include `aria-label`, `title`, and `focus-visible:ring-2 focus-visible:outline-none rounded` on icon-only buttons.
