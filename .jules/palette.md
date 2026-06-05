## 2024-06-05 - Dynamic Accessibility Attributes for Disabled Forms
**Learning:** In interactive React applications, combining dynamic `aria-label` and `title` attributes based on input validation state provides screen reader users and visual users with clear context on *why* a button is disabled, rather than just presenting a non-interactive element.
**Action:** Always pair `disabled={condition}` with `aria-label={condition ? 'Reason' : 'Action'}` and visual indicators like `disabled:cursor-not-allowed focus-visible:ring-2` on interactive form submit buttons.
