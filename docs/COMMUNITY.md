# Community Guide

Welcome to the Seal OS community. This page is your starting point for asking questions, reporting problems, and finding ways to contribute.

## How to ask questions

- **GitHub Discussions** — For usage questions, troubleshooting, open-ended ideas, or showing off something you built.  
  https://github.com/teerthsharma/Epsilon-Hollow/discussions

- **GitHub Issues** — For concrete bugs, feature requests, or documentation problems with a clear scope.  
  If you are unsure whether something is a bug, start a Discussion first; we can convert it to an Issue once confirmed.

## How to report security issues

Please **do not** open a public issue for security-sensitive reports.

- Preferred: Use [GitHub Security Advisories](https://github.com/teerthsharma/Epsilon-Hollow/security/advisories/new) ("Report a vulnerability" on the Security tab).
- Fallback: Email `teerths57@gmail.com` with reproduction steps, affected version/commit, and impact.

Details: `SECURITY.md`

## How to find something to work on

1. Browse [Issues labelled `good first issue`](https://github.com/teerthsharma/Epsilon-Hollow/labels/good%20first%20issue). These are small, well-scoped, and ideal for new contributors.
2. Browse [Issues labelled `help wanted`](https://github.com/teerthsharma/Epsilon-Hollow/labels/help%20wanted). These are larger tasks where maintainer bandwidth is limited.
3. Check the **subsystem map** in `CONTRIBUTING.md` to find the code area that matches your skills or interests.
4. If you have your own idea, open a Discussion or Feature Request issue first so we can align on design before you invest time in implementation.

## Communication norms

- **Be constructive.** Critique ideas and code, not people.
- **Be specific.** When reporting a problem, include version numbers, reproduction steps, and serial logs when applicable.
- **Be patient.** Seal OS is a research OS; reviews may take time because theorem gates and benchmark regressions are checked carefully.
- **Stay on topic.** Keep issue threads focused on the reported bug or feature. Use Discussions for tangential questions.
- **Respect the Code of Conduct.** See `CODE_OF_CONDUCT.md`. Harassment or disrespectful language will not be tolerated.

## Getting started checklist

- [ ] Read `README.md` for the high-level architecture overview.
- [ ] Read `CONTRIBUTING.md` for build instructions, style guide, and theorem-gate requirements.
- [ ] Read `docs/BOOT.md` to get Seal OS running in QEMU.
- [ ] Pick a `good first issue` or open a Discussion with your idea.
- [ ] Run `cargo fmt --all && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace` before opening any PR.

We are glad you are here. Happy hacking — and keep it topological.
