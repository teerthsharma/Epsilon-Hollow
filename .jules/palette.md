## 2024-06-16 - Accessible disabled states on chat inputs
**Learning:** Adding dynamic title and aria-label attributes explaining *why* an element is disabled greatly improves screen reader utility. Combining visual disabled cues (`cursor-not-allowed`) with `focus-visible` ensures keyboard accessibility even when elements are partially obscured or integrated within other forms.
**Action:** When implementing disable states on inputs or buttons, use dynamic title/aria-labels based on state, and ensure focus-visible styles are cleanly applied using Tailwind.
