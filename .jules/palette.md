
## 2025-01-20 - Ensure icon-only buttons have accessible names and focus states
**Learning:** Icon-only buttons (like the Trash icon in the console panel) are often overlooked for accessibility. They need `aria-label` for screen readers and `title` for mouse users to understand their purpose. Focus states (`focus-visible`) are critical for keyboard navigation users.
**Action:** Always verify that icon-only buttons have explicit ARIA labels and focus-visible styling (like `focus-visible:ring-2 focus-visible:outline-none`) implemented via Tailwind classes. Use `title` for native tooltips as an added enhancement.
