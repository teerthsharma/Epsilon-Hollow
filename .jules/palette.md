## 2024-10-27 - Dynamic Titles and Focus-Visible for Disabled Elements
**Learning:** Empty interactive elements disabled with CSS classes like `opacity-50` alone lack context for screen readers and keyboard users.
**Action:** Always add dynamic `title` attributes explaining why a button is disabled, and pair visual states with `focus-visible:ring-2 focus-visible:outline-none` for consistent keyboard navigation accessibility without breaking mouse click visuals.
