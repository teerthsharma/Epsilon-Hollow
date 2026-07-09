# AETHER Benchmark Policy

Benchmark documentation records evidence. It does not create evidence.

The active MkDocs benchmark page is `benchmarks/index.md`. This legacy page is
kept as a pointer for readers who open the flat Markdown tree directly.

## Current Position

No broad Aether speedup claim is active from this page. Timing claims require a
raw artifact with:

- repository commit;
- command;
- hardware and software environment;
- baseline implementation;
- input size and seed;
- correctness metric;
- timing distribution.

## Active Claim Boundary

The repository contains parser, interpreter, VM, kernel-adjacent, and example
code that can be tested at crate scope. It does not currently publish a
reproducible artifact here that proves LLM, geometric-regression, or topology
speed leadership.
