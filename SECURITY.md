# Security Policy

## Reporting a vulnerability

Please report suspected vulnerabilities privately. Do **not** open a public issue
for security-sensitive reports.

- Preferred: GitHub Security Advisories ("Report a vulnerability" on the repo's
  Security tab).
- Email fallback: `security@epsilon-hollow.invalid` (replace with the
  maintainer's address before publishing).

Include reproduction steps, affected version/commit, and impact. Expect an
acknowledgement within a reasonable window; coordinated disclosure is
appreciated.

## Threat model

Epsilon-Hollow's FastAPI backend (`kernel/epsilon/epsilon-ide/pentesting/backend/main.py`)
is designed for **local-only** operation by default:

- Default bind host is `127.0.0.1`. Non-loopback binds require
  `EPSILON_API_TOKEN`.
- Mutating endpoints require a bearer token (constant-time compared) unless the
  process is in dev mode **and** bound to loopback.
- Shell-level endpoints (`/api/v1/claw/execute`, `/ws/terminal`) are
  triple-gated: `EPSILON_DEV_MODE=1` **and** `EPSILON_DEV_TERMINAL=1` **and**
  the same bearer-token check.
- File APIs are jailed to `workspace_root` (path traversal is rejected).
- CORS defaults to the documented frontend origin only.

Out of scope: multi-tenant deployment, internet exposure, untrusted operators
on the host machine.

## Environment variables

| Variable | Default | Effect | Security implication |
| --- | --- | --- | --- |
| `EPSILON_API_TOKEN` | unset | Bearer token required on mutating endpoints. | **Required** unless `EPSILON_DEV_MODE=1` AND bind is loopback. |
| `EPSILON_DEV_MODE` | `0` | Enables dev convenience features. | Permits empty token only in combination with loopback bind. |
| `EPSILON_DEV_TERMINAL` | `0` | Mounts `/api/v1/claw/execute` and `/ws/terminal`. | Required on top of `EPSILON_DEV_MODE=1` to expose shell access. |
| `EPSILON_BIND_HOST` | `127.0.0.1` | uvicorn bind host. | Any non-loopback value forces `EPSILON_API_TOKEN` to be set. |
| `EPSILON_BIND_PORT` | `8742` | uvicorn bind port. | Informational. |
| `EPSILON_CORS_ORIGINS` | `http://127.0.0.1:8742,http://localhost:8742` | Comma-separated allowed origins. | Avoid `*`; that mode is no longer the default. |
| `EPSILON_WORKSPACE_ROOT` | `~` | Initial workspace jail root. | Determines what the file APIs can see. |

## Known limitations

- `TeleportTarget::RemoteVoid` (see
  `kernel/epsilon/epsilon/crates/epsilon/src/teleport.rs`) is an unauthenticated
  stub returning `RemoteUnimplemented`. It is not safe for use over untrusted
  networks and must not be exposed beyond a trusted host.
- The dev terminal, even when triple-gated, runs commands with the host user's
  privileges. Treat it like SSH — only enable it on a machine and network you
  control.
- Native C++ extensions are optional; Python fallbacks have not been audited
  for parity.
