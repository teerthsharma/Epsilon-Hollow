
## 2026-10-24 - React.memo for Append-Only Logs
**Learning:** In React, mapping append-only arrays (like logs) directly in a parent component causes O(N) re-renders across the entire list whenever a new item is added.
**Action:** Always extract the individual list item into a `React.memo()` wrapped component for append-only lists to achieve O(1) rendering for new additions and prevent unnecessary renders of existing items.

## 2026-10-24 - Cargo dependencies resolving with vulnerabilities
**Learning:** Older unpinned cargo dependencies like `candle-core` can transitively pull in crates with critical vulnerabilities like `zip` (RUSTSEC-2026-0367, RUSTSEC-2026-0368, RUSTSEC-2026-0369), failing `cargo-audit` checks.
**Action:** Update the parent cargo package (e.g. `candle-core@0.3.3` -> `0.8.4`) in the `Cargo.toml` across all workspace crates using it. Also use `--locked` with a specific version of cargo-audit to avoid MSRV breakage.
