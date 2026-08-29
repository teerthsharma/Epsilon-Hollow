
## 2024-10-25 - Accessible utility buttons in Tailwind
**Learning:** Icon-only utility buttons in Tailwind require explicit `aria-label` or `title` attributes for screen readers, and matching `focus-visible` ring colors based on the context (e.g. `gov-error` for close/destructive, `gov-accent` for general utility) to ensure proper keyboard navigation visibility.
**Action:** Always pair `focus-visible:ring-2 focus-visible:outline-none rounded` with `aria-label` when using `lucide-react` icons in standalone buttons.
