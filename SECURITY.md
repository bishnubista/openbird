# Security Policy

OpenBird is a local-first personal-memory tool: it reads the text of your active
window, transcribes meetings, and stores everything in an on-device database. A
security flaw here can expose deeply personal data, so we take reports seriously.

## Reporting a vulnerability

**Please do not open a public issue for security vulnerabilities.**

Report privately through GitHub's coordinated-disclosure channel:

1. Go to the repository's **Security** tab → **Report a vulnerability**
   (GitHub Private Vulnerability Reporting).
2. Describe the issue, the affected component, and reproduction steps.

If you cannot use GitHub's reporting flow, open a regular issue that contains
**only** the sentence "I would like to report a security issue privately" — with
no technical detail — and a maintainer will arrange a private channel.

### What to include

- The affected component (capture-helper, audio-helper, the Python core, the
  macOS app, the encryption gate, an MCP integration, the release/signing flow).
- Version or commit (`openbird doctor` reports a content-safe version, or the
  beta `.dmg` tag).
- Reproduction steps and impact (what data or capability is exposed).
- A proof-of-concept if you have one.

> **Do not include captured memory or real personal data in a report.** Reproduce
> with synthetic input. If a real value is unavoidable to demonstrate the issue,
> redact it and say so — the same privacy rules that govern the product govern the
> report.

## Scope

In scope — issues that undermine OpenBird's privacy or integrity guarantees:

- Captured text, window titles, or URLs leaking into logs, exceptions, argv, or
  any off-device destination.
- Redaction / allowlist / blocklist bypasses that cause unintended capture.
- Encryption-at-rest weaknesses: the SQLCipher gate passing when it should fail
  closed, key material exposure, or weak file permissions on the data store.
- A cloud route activating without the explicit `OPENBIRD_ALLOW_CLOUD=1` opt-in
  or without surfacing the CLOUD ACTIVE banner.
- Code-signing / notarization / TCC-attribution flaws in the release artifacts
  that let an untrusted binary inherit capture permissions.
- MCP integration paths that perform unconfirmed write actions.

Out of scope:

- Vulnerabilities in third-party dependencies (Ollama, model weights, PyPI
  packages) — report those upstream, though we welcome a heads-up.
- Issues that require an already-compromised local account with full disk access
  (the threat model assumes the local user is trusted).
- Missing hardening that is documented as a known limitation in the README
  (e.g. the unsigned Homebrew bundle cannot obtain capture permissions by design).

## Supported versions

OpenBird is pre-1.0 and ships as rolling beta builds. Only the **latest** release
(the most recent `beta-dmg-*` tag and current `main`) receives security fixes.
There are no long-term-support branches yet.

| Version            | Supported |
| ------------------ | --------- |
| Latest `main` + newest `beta-dmg-*` tag | ✅ |
| Older tagged betas | ❌ |

## Disclosure process

- We aim to acknowledge a report within **5 business days**.
- We'll work with you on a fix and a coordinated disclosure timeline; the default
  target is **90 days** or the release of a fix, whichever comes first.
- With your consent, we'll credit you in the release notes for the fix.
