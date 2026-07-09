# Contribution Standard

Small research repositories still need explicit standards.

## Code

- Rust APIs should return typed errors across public boundaries.
- Production-facing Rust should avoid `unwrap()` unless the invariant is local
  and obvious.
- Unsafe code needs a local safety explanation and a test or proof gate for the
  boundary it relies on.
- Optional runtimes must stay optional and report missing gates clearly.
- Host examples must not be documented as OS runtime behavior unless an OS proof
  artifact exercises them.

## Docs

Every concept page should include:

- the mechanical model;
- a plain-language explanation;
- an example or diagram;
- a failure mode;
- a claim boundary.

Every benchmark page should include:

- command;
- raw artifact path;
- baseline;
- environment;
- correctness metric;
- seed or determinism note.

## Commits

Group related changes into reviewable commits. Do not mix benchmark claims,
runtime changes, and documentation tone edits unless the commit explains the
evidence that ties them together.
