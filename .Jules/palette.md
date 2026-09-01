
## 2024-05-18 - Icon Button Focus Styling Pattern
**Learning:** When styling icon-only utility buttons for keyboard accessibility in this dark-themed app, standard browser focus rings don't have enough contrast. Destructive buttons require `focus-visible:ring-gov-error`, while standard buttons require `focus-visible:ring-gov-accent`.
**Action:** Always explicitly define `focus-visible:ring-2 focus-visible:ring-* focus-visible:outline-none rounded` for all new utility icon buttons to ensure keyboard users have clear focus indicators that match the app's visual language.
