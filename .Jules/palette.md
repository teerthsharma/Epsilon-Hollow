## 2024-08-07 - Accessible Dynamic Streams & Empty States
**Learning:** Dynamic text areas (like a console or telemetry stream) must use `role="log"` and `aria-live="polite"` for screen readers to correctly announce new content. Additionally, completely empty data lists leave users wondering if the system is broken; an explicit empty state string clarifies readiness.
**Action:** Always add `role="log"` and `aria-live="polite"` to append-only text stream containers, and ensure a non-empty fallback message (e.g., 'Waiting for data...') is rendered when the data array is empty.
