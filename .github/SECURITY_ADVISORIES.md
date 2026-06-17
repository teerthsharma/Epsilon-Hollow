# Security Advisory Template & Policy

## 1. Advisory Format

When a security issue is confirmed, draft an advisory using the following
structure. If you are an external reporter, you do not need to fill every
field — the maintainers will complete it during triage.

```markdown
## Title
Short, specific description of the vulnerability.

## Severity
Low / Medium / High / Critical (CVSS optional)

## Affected Versions
List tags, branches, or commit ranges. Use `git describe` when possible.
Example: `v0.4.0` – `v0.4.7.5`

## Description
What is the flaw? Root cause, affected component, and preconditions.

## Impact
What can an attacker achieve? Data disclosure, privilege escalation, DoS, etc.

## Proof of Concept (optional)
Minimal steps, code, or configuration to trigger the issue.

## Mitigation / Workaround
Steps users can take *now* to reduce risk before a patch is available.

## Fix
Link to the commit or PR that resolves the issue.

## Credits
Reporter name / handle (and affiliation, if desired).
```

---

## 2. CVE Request Process

1. **Confirm the vulnerability** — at least one maintainer must reproduce or
   review the report and agree it is a security issue.
2. **Draft a GitHub Security Advisory** in the repository's *Security →
   Advisories* tab.
3. **Request a CVE** — GitHub can assign a CVE automatically for repositories
   with advisories enabled. If GitHub assignment is unavailable, the maintainer
   will request a CVE from MITRE via the [CVE Request web form](https://cveform.mitre.org/)
   or contact a CNA.
4. **Publish the advisory** only after a patched release is available or at the
   end of the disclosure timeline (see §3).

---

## 3. Disclosure Timeline

Seal OS follows a **90-day default** coordinated disclosure timeline:

| Day | Action |
|---|---|
| 0 | Reporter submits private report. |
| ≤7 | Maintainer acknowledges receipt and begins triage. |
| ≤14 | Maintainer confirms or rejects the issue; shares preliminary severity. |
| ≤30 | Maintainer shares target fix date or requests timeline extension. |
| ≤90 | Patch released and advisory published. |

**Exceptions:**
- **Active exploitation observed:** Disclosure may be accelerated (≤7 days) if
  a fix is ready, or a temporary workaround is published immediately.
- **Complex fix required:** A single extension of up to 30 days may be granted
  with reporter agreement.
- **Reporter requests earlier disclosure:** We will honour that request if a fix
  or workaround is available.

---

## 4. Credit Policy

- Researchers who report valid security issues **will be credited by default**
  in the advisory and release notes, unless they explicitly request anonymity.
- We do not offer a bug-bounty program at this time.
- Disputes about severity or scope will be resolved through good-faith
  discussion; we will not retaliate against researchers who follow this policy.

---

## 5. Contact

- GitHub Security Advisories: Use the "Report a vulnerability" button on the
  repo's Security tab.
- Email fallback: `teerths57@gmail.com` (PGP key available on request).

---

*Last updated: 2026-06-01*
