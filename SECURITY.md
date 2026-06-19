# Security Policy

OpenBird is a local-first personal-memory tool: it captures window text and
system audio and stores them on-device. Security and privacy issues are taken
seriously.

## Reporting a vulnerability

**Please do not open a public issue for security vulnerabilities.**

Report privately via GitHub's
[private vulnerability reporting](https://github.com/bishnubista/openbird/security/advisories/new)
("Report a vulnerability" under the repository's *Security* tab).

Please include:

- A description of the issue and its impact.
- Steps to reproduce (a minimal proof-of-concept if possible).
- The affected version/commit and your environment (macOS version, extras).

You can expect an initial acknowledgement within a few days. Please allow time
for a fix before any public disclosure.

## Scope of interest

Because OpenBird handles sensitive on-device data, we are especially interested
in:

- **Content leakage** — captured text/audio, prompts, or DB contents escaping the
  device (e.g. via logs, error messages, telemetry, or the chat/cloud path).
- **Encryption gaps** — the SQLCipher/Keychain storage gate failing open, or keys
  being exposed.
- **Capture-boundary issues** — the allowlist/blocklist, redaction policy, or
  capture indicator being bypassed.
- **Integration write actions** (MCP) executing without explicit confirmation.

## Supported versions

OpenBird is pre-1.0 (early but working). Security fixes are applied to the
latest `main`; there is no long-term support branch yet.
