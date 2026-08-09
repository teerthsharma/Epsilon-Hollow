
## 2024-08-09 - Zustand Empty Array Edge Cases
**Learning:** Sometimes "clear" functions in stores don't result in `[ ]`, but rather a placeholder initial state (e.g., `["> Console cleared"]`). Checking `array.length === 0` is insufficient for empty state UX if the system replaces content with a dummy element. Also, ensure empty states are mutually exclusive with data mapping to prevent visual text overlap.
**Action:** When creating empty states, explicitly check for known "empty" placeholder values in the array or state before rendering the empty state UI, and conditionally render either the empty state OR the data map using a ternary to prevent layout collision.
