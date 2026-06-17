## 2024-05-18 - Thematic Empty States
**Learning:** Adding a thematic empty state (like "KERNEL AWAITING INPUT" with a pulsing CPU icon) is significantly more engaging than a blank screen and helps set the tone for the application's sci-fi aesthetic immediately upon load. It also guides the user on what action to take ("Inject knowledge").
**Action:** When working on empty states, try to match the application's unique tone rather than just providing generic "No messages" text.

## 2024-05-18 - Keyboard Focus in Dark Themes
**Learning:** In dark mode interfaces (like Apeiron's black/gray theme), default browser focus rings are often invisible or clash terribly. Using explicitly styled `focus-visible:ring-2 focus-visible:ring-[brand-color]` (like `ring-green-500` here) is crucial for keyboard navigation visibility while maintaining the aesthetic.
**Action:** Always test keyboard focus in dark mode and apply explicit `focus-visible` utility classes that match the theme's accent color.