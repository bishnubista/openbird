# Design: read-only MCP server for OpenBird memory

Status: **proposed** (design only — no implementation until this is approved + Codex consensus)
Owner: TBD · Target: post-0.2.0

## 1. Goal

Let an MCP client (Claude Code, Claude Desktop, and — later — claude.ai) query a
user's captured OpenBird memory through the Model Context Protocol, **read-only**,
so a user can ask "what did I work on yesterday?" from their assistant of choice
instead of only the `openbird` CLI.

## 2. The one principle this design exists to enforce

**Enabling the MCP server moves data across OpenBird's core privacy boundary.**
OpenBird's entire thesis is *local-first: your data never leaves your device by
default*. An MCP server is, by definition, a way for an external process to read
that data. The server does not "send data to Anthropic" — it answers whatever the
**connected client** asks, and OpenBird cannot control what that client then does
(a cloud LLM relays the content off-device; even local Claude Desktop sends to
Anthropic's API).

Therefore the honest framing — which the UX must state literally — is:

> **While MCP is enabled, any connected app can read your captured memory, and
> your data is no longer device-only.** The connected client (for example a cloud
> AI assistant) receives whatever it queries.

This is the same boundary OpenBird already gates for cloud LLM/embed routes
(`OPENBIRD_ALLOW_CLOUD` + the "CLOUD ACTIVE" banner). **We reuse that exact mental
model and machinery rather than inventing a new one.**

## 3. Non-goals

- **No write path.** No tool mutates memory, deletes, or triggers capture. Mirrors
  the existing `openbird/integrations/mcp.py` stance (writes raise
  `MCPWriteDisabledError`). The MCP *server* is read-only the way the MCP *client*
  connectors are.
- **No new retrieval logic.** Tools wrap the existing engine
  (`MemoryStore.search`, `RAG.answer`, `MemoryStore.stats`, `day_sessions`,
  `time_range`) — we do not reimplement search or citations.
- **No remote/cloud transport in phase 1** (see §6).
- **No always-on background listener** without an explicit, visible opt-in.

## 4. Architecture

```text
MCP client (Claude Code / Desktop / claude.ai)
        │  MCP (stdio | streamable-http)
        ▼
openbird mcp serve  ── FastMCP/MCPServer (mcp 1.27.x)
        │  read-only
        ▼
MemoryStore (opened read-only) ── search / stats / time_range
        │
        ▼
SQLite (~/.openbird/openbird.db, SQLCipher if a key is available)
```

- **New module:** `openbird/integrations/mcp_server.py` (server) + a thin
  `openbird mcp serve` CLI command. Keep it import-light: the `mcp` SDK is the
  optional `integrations` extra, so the module must import lazily and degrade with
  a clear error if the extra is absent (same pattern as the MCP client today).
- **SDK API (verified against installed `mcp` 1.27.2).** Use the high-level
  `from mcp.server.fastmcp import FastMCP` decorator API (`@mcp.tool()`,
  `@mcp.resource()`). **`FastMCP.run()` takes only `(transport, mount_path)`** —
  `host`/`port` are **constructor** settings. So: `mcp = FastMCP(name, host=…,
  port=…)` then `mcp.run()` (stdio) or `mcp.run(transport="streamable-http")`.
  Pin the `integrations` extra to the verified API surface (`mcp>=1.27,<1.28`
  rather than the current loose `mcp>=1.0`) before implementation, and re-check the
  signature against whatever version is locked (project rule: code from the
  installed API, not memory).
- **Store access is read-only — but `MemoryStore.__init__` is NOT read-only
  today.** On open it runs `_apply_schema()`, schema migrations + `user_version`
  stamping, and `_record_cohort()` (`INSERT`/`UPDATE`); the crypto open path also
  sets `journal_mode=WAL`, chmods, and can create files. A new **read-only open
  path** is a prerequisite (see §10, Phase 0): no directory creation, no chmod, no
  WAL switch, no schema apply, no migration, no cohort write; set
  `PRAGMA query_only=ON`; verify schema-version + cohort compatibility read-only and
  **fail closed** if missing/incompatible. The MCP server must use only this path.
- **Bind-host safety needs its own classifier — do NOT reuse `is_loopback_host()`.**
  That helper (`config.py`) intentionally treats `0.0.0.0` as loopback for *outbound*
  model-route classification; for a *server bind* `0.0.0.0` means "all interfaces"
  and would silently expose private memory. Add a dedicated MCP bind classifier:
  **only `127.0.0.1` / `localhost` / `::1` are local**; `0.0.0.0`, `::`, any LAN IP,
  hostname, or malformed value is remote and requires `mcp_allow_remote` + auth + TLS.

## 5. Tool surface (read-only, curated, scoped)

Expose a *small, curated* surface — not a generic `run_sql`/`dump_all`. The surface
is itself a privacy control: scoping is consent made concrete.

**Reality check — the current engine has no scoped retrieval.** `MemoryStore.search`
and `RAG.answer` accept only `(query, k, semantic)`; `time_range(start_ts, end_ts)`
has no app/source filter. Only `day_sessions(start_ts, end_ts, *, source=…)` and
`time_range_text(…, source=…)` carry any scope (time + source, not arbitrary apps).
**Since scope is a privacy control, it cannot be faked by filtering after an
unscoped retrieval** (the unscoped query has already searched everything). So
either we add scoped retrieval APIs first, or a tool only advertises the scope it
can actually enforce in SQL/retrieval.

| Tool | Wraps (today) | Enforceable scope now | Phase |
|---|---|---|---|
| `memory_stats()` | `MemoryStore.stats()` | n/a (counts only, content-free) | 1 |
| `timeline(since, until)` | `day_sessions(start, end, source=…)` | time + source (in SQL) | 1 |
| `search_memory(query, since?, until?, apps?, limit=10)` | scoped search (Phase 0) | time + apps (in SQL) | **blocked until Phase 0 scoped retrieval lands** |
| `ask_memory(question, since?, until?, apps?)` | `RAG.answer` over scoped search | time + apps (in SQL) | after scoped search (Phase 2) |

Optional MCP **resources** (phase 2): expose a daily briefing as a read-only
resource (`openbird://briefing/today`).

Design rules for tools:
- **Mandatory result cap** (`limit`, hard max e.g. 50) so one call can't siphon the
  whole store. This is enforceable today (it's just `k`).
- **Scope is enforced in retrieval — no unscoped fallback.** `search_memory` (and
  `ask_memory`) must **not** ship until Phase 0 adds a scoped search path that filters
  `since`/`until`/`apps` **in the SQL/candidate generation**, plus an optional
  server-side scope **floor** (`OPENBIRD_MCP_MAX_AGE_DAYS`, `OPENBIRD_MCP_SCOPE_APPS`)
  the client cannot widen. We do **not** ship an unscoped `search_memory` guarded only
  by consent copy — that would make scope a copy-only safeguard rather than a
  mechanism, defeating its purpose. Until scoped retrieval exists, Phase 1 exposes only
  `memory_stats` (content-free) and `timeline` (time + source enforced in SQL).
- **Egress redaction:** run `redact.scrub()` (and `scrub_metadata`) over every
  field returned. Captured text is *usually* already scrubbed at capture, but
  `source="mcp"`/`ingest` content may not be — scrub on the way out as
  defense-in-depth. Snippets stay short (existing 240-char citation cap).

## 6. Consent & gating (the heart of the design)

### 6.1 Off by default, explicit enable
A new gate, off by default, mirroring `allow_cloud`:

| Setting | Env | Default | Meaning |
|---|---|---|---|
| `mcp_enable` | `OPENBIRD_MCP_ENABLE` | `false` | `openbird mcp serve` refuses unless set/confirmed |
| `mcp_allow_remote` | `OPENBIRD_MCP_ALLOW_REMOTE` | `false` | **second, stronger** gate required for the HTTP transport |
| `mcp_host` / `mcp_port` | `OPENBIRD_MCP_HOST` / `_PORT` | `127.0.0.1` / `un-set` | loopback default; non-loopback requires `mcp_allow_remote` |
| `mcp_auth_token` | `OPENBIRD_MCP_AUTH_TOKEN` | — | required for HTTP; bearer auth |

First enable (CLI or GUI) shows the §2 consent sentence and requires an explicit
confirm. Non-interactive (no TTY) refuses unless the env gate is already set —
same shape as the cloud opt-in (`CloudOptInRequired`).

### 6.2 Transport is the privacy tier — phase it

| Transport | Clients | Exposure | Phase |
|---|---|---|---|
| **stdio** | Claude Code, Claude Desktop | client spawns server locally; data leaves device only via the client's own LLM | **Phase 1** |
| **streamable-http (loopback + token)** | local HTTP clients | local network listener; needs auth | Phase 2 |
| **streamable-http (non-loopback)** | **claude.ai web**, remote | highest exposure; needs auth + TLS + `mcp_allow_remote` | Phase 2, last |

claude.ai (web) requires the remote HTTP path, so it is the **highest-exposure and
last** thing we ship — deliberately, behind the second gate. The `mcp_host` is
classified by the **dedicated MCP bind classifier** from §4 (NOT `is_loopback_host`,
which allows `0.0.0.0`): anything other than `127.0.0.1`/`localhost`/`::1` is remote
and refused without `mcp_allow_remote` + auth + TLS. Binding `0.0.0.0`/`::` is never
silently treated as local.

### 6.3 Visible "MCP ACTIVE" indicator
While the server runs, surface a persistent indicator (CLI banner on start; GUI
menu-bar state akin to the capture indicator) reading e.g.
`MCP ACTIVE — memory is queryable by connected clients`. Prefer per-session
(server runs while you want it) over silent-forever.

## 7. Observability (privacy-safe, mandatory)

Per OpenBird's rule (reason codes / metadata / counts — **never** captured text),
log each tool call content-free:

```text
mcp_query tool=search_memory results=7 window_days=1 scope_apps=2 status=ok
```

Never log the query string, the question, snippets, or returned content. This lets
a user audit *what was asked of their memory* without the log itself becoming a
leak. Reuse the existing structured-logging style (cf. `_log_rerank_skip`).

## 8. Encryption interaction (must be designed, not discovered)

If the store is SQLCipher-encrypted, the server needs the DB key. We saw the
failure mode: a non-signed interpreter times out on the Keychain
(`keyring get_password timed out … DB opened WITHOUT app-level encryption`).

**Key access binds to the process that calls `keyring` — it is NOT inherited by a
child process.** `crypto._get_or_create_key()` fetches the key in whatever process
runs it; the Keychain ACL is satisfied by *that* process's code signature, so the
signed app **spawning** a separate `openbird mcp serve` interpreter does **not**
grant the child access (the child has a different/absent signature). Valid options,
in preference order:

- **Host the server in-process inside the signed app** (the process with the stable
  Keychain identity opens the store and runs `FastMCP`). Cleanest; also gives a
  natural place for the `MCP ACTIVE` indicator.
- **A signed helper** with its own stable Keychain access group / grant (like the
  capture/audio helpers), which fetches the key under its own identity.
- **Explicitly hand the key to the server process** over a designed secure channel
  (e.g. the app passes `OPENBIRD_DB_KEY` to a child it launches), never logged.

For a **plaintext** store (today's default when keyring is unavailable) the server
reads directly. Document plainly: *agent access to an encrypted store is not
guaranteed from an arbitrary interpreter, and a bare `uv run` / brew CLI generally
cannot decrypt it.* This makes "host inside the signed app" the recommended answer
to Open Question 1.

## 9. Threat model & abuse cases

- **Untrusted captured content → prompt injection.** Captured text/observations
  are *untrusted data* (the MCP client comment already says so). Returned snippets
  may contain adversarial instructions aimed at the consuming LLM. Mitigation: we
  only ever *return* content as data (never execute it), keep tools read-only, and
  document that the client is responsible for treating tool output as untrusted.
- **Malicious/over-eager client siphoning everything.** Mitigation: result caps,
  scope floors, no bulk-dump tool, and the audit log makes mass querying visible.
- **Local listener exposure (HTTP).** Mitigation: loopback default, bearer auth
  required, non-loopback behind `mcp_allow_remote`; never bind `0.0.0.0` silently.
- **Stale consent.** Mitigation: per-session server + visible indicator; consider
  re-confirm on transport change (stdio→http).

## 10. Phasing

1. **Design doc** (this) → Codex adversarial review → consensus.
2. **Phase 0 — prerequisites (engine work, before any server ships):**
   - **Read-only store open path** (§4): `query_only=ON`, no create/chmod/WAL/schema/
     migration/cohort write, fail-closed on incompatible schema/cohort.
   - **Dedicated MCP bind-host classifier** (§4): only `127.0.0.1`/`localhost`/`::1`
     local; everything else (incl. `0.0.0.0`/`::`) remote.
   - **Scoped retrieval API** (§5): `since`/`until`/`apps` enforced *in retrieval*,
     plus optional server-side scope floor. (If deferred, Phase 1 ships unscoped
     `search_memory` with consent copy that says so.)
3. **Phase 1** — server hosted **in the signed app** (§8) over **stdio** +
   `mcp_enable` gate + consent + `MCP ACTIVE` indicator + `memory_stats` + `timeline`
   + egress redaction + content-free audit logging, using the read-only open path.
   **No `search_memory` until Phase 0's scoped retrieval lands** (no unscoped
   fallback). Works with Claude Code / Claude Desktop.
4. **Phase 2** — streamable-http via `FastMCP(host=…, port=…)` + bearer auth +
   `mcp_allow_remote` gate + the bind classifier → unlocks claude.ai. `ask_memory`
   and the briefing resource land here (or with scoped search). Resources optional.

Each phase: research-pinned SDK API → Codex review of design delta → implement →
local CI green (`pytest`, `shellcheck`) → Codex review of diff → PR.

## 11. Testing strategy

- Unit: each tool wraps the real engine against a fake/in-memory store; assert
  read-only (no writes), result caps, scope filtering, and that **every returned
  field passes redaction** (feed a planted secret through an observation, assert
  it's masked in tool output).
- Gating: `mcp serve` refuses without `mcp_enable`; HTTP refuses without
  `mcp_allow_remote`; non-loopback host refused without remote opt-in.
- Bind classifier: assert `0.0.0.0`, `::`, LAN IPs, hostnames, and malformed values
  are classified **remote** (regression guard for the `is_loopback_host` 0.0.0.0
  quirk); only `127.0.0.1`/`localhost`/`::1` are local.
- Read-only open: assert the MCP store-open performs no writes — no schema apply,
  migration, cohort `INSERT`/`UPDATE`, WAL switch, chmod, or file creation — and
  that `query_only=ON` rejects any write; fail-closed on incompatible schema/cohort.
- Logging: assert no captured text/query strings appear in emitted logs (extend
  the existing content-safety test pattern used for `doctor`).
- Encryption: documented manual check that the signed-app-hosted server can read an
  encrypted store while a bare interpreter cannot.

## 12. Open questions (for review)

1. ~~Hosted by the GUI app vs standalone CLI?~~ **Leaning resolved (§8):** host
   in-process inside the signed app, because Keychain key access does not transfer
   to a spawned child. Standalone CLI only viable for plaintext stores or with an
   explicitly passed `OPENBIRD_DB_KEY`.
2. Is a server-side **scope floor** (`OPENBIRD_MCP_MAX_AGE_DAYS` / `_SCOPE_APPS`)
   worth it in phase 1, or deferred?
3. Do we expose `ask_memory` (runs the local LLM) in phase 1, or start with
   `search_memory`/`memory_stats` only to keep the surface minimal?
4. Per-session consent vs. persisted opt-in — how sticky should `mcp_enable` be?
