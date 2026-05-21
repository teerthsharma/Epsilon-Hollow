## 2024-05-21 - Keyboard Navigation & Contextual Tooltips
**Learning:** Found an accessibility issue pattern where custom interactive elements (buttons, inputs) in the Apeiron Cockpit lack explicit `focus-visible` styles for keyboard navigation, and disabled buttons don't provide contextual screen-reader feedback (e.g., explaining why they are disabled).
**Action:** Always add `focus-visible:ring-2`, `focus:outline-none`, `disabled:cursor-not-allowed`, and dynamic `aria-label`/`title` based on state for all custom forms/buttons in this repository.
