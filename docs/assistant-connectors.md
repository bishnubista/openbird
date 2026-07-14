# Desktop assistant access

OpenBird exposes a local, read-only Model Context Protocol (MCP) server so a desktop assistant can
request captured context without receiving a background feed or direct database access.

## Claude Desktop

[Claude Desktop supports local MCP servers](https://support.anthropic.com/en/articles/10949351-getting-started-with-local-mcp-servers-on-claude-desktop).
Connect the installed OpenBird CLI:

```bash
openbird assistant install-claude
openbird assistant status
```

The installer:

- shows an egress warning and requires confirmation;
- preserves every unrelated Claude setting and MCP server;
- writes a private `0600` backup before changing the config;
- adds only the `openbird` stdio server; and
- never opens a network listener.

Restart Claude Desktop after connecting. Example prompts:

- "Use OpenBird to tell me what I worked on in the last hour."
- "Search my OpenBird capture for the release blocker and cite the observations."
- "Check whether OpenBird capture is healthy without reading any captured text."

## Tool contract

The server exposes exactly four read-only tools:

| Tool | Purpose | Bounds |
|---|---|---|
| `openbird_recent_capture` | Recent captured excerpts, deduplicated and pageable | 1-1440 minutes, 1-20 groups per page, 200-row page scan |
| `openbird_search_capture` | Local BM25 capture search | 500-character query, 1-20 requested results |
| `openbird_activity_summary` | Per-app durations, meetings, focus rollup | 1-1440 minutes; metadata only, no captured content |
| `openbird_capture_status` | Memory and exclusion counts | Metadata only; no captured content |

Content tools cap every excerpt at 2,000 characters and total excerpt text at 12,000 characters per
call. They use direct local SQLite/FTS reads and never call an embedding, reranking, or completion
model. Existing Deep Brain app, source, and observation exclusions are applied before serialization
(a malformed `re:` exclusion pattern fails every tool closed rather than silently not matching).
Legacy observations without app provenance fail closed. URLs and window titles stay local.

### Paging and deduplication (`openbird_recent_capture`)

Each result is one distinct piece of content: identical captured text seen many times inside a page
collapses to a single group carrying `seen_count`, `first_ts`, and `last_ts`, anchored on the newest
occurrence for citation. The response's `window_start_ts`/`window_end_ts` report the queried window,
and `next_cursor` continues it: pass the token back to read strictly older results from the same
frozen window; `null` means the window is exhausted. Cursors are opaque random handles — they carry
no data, are **single-use** (consuming one returns a fresh `next_cursor`; replaying a used, expired,
or restart-invalidated token fails and the walk restarts with a fresh first call), and expire after
15 minutes.

### Activity summary

`openbird_activity_summary` answers "what did I focus on" from **trusted metadata only** — activity
spans, never captured text. It returns per-app foreground and meeting durations with span counts
(top 30 apps; the rest folded into a nameless `other_apps` bucket reporting only its total seconds
and app count), AFK time, context-switch count, and the longest focus block. Redacted
(coarse-tier) spans keep their app identity as metadata, so redacted time is attributed:
`redacted_by_app` lists `{bundle_id, reason, seconds}` per app (closed reason vocabulary —
`not_allowlisted`, `blocklisted`, `dangerous`, `private`, `paused`, `self_capture`, plus an
`unknown` defense-in-depth sentinel emitted only for a corrupted row), with
`redacted_unattributed_seconds` covering spans that never had an app (paused gaps, unknown
app) and `redacted_other_seconds` folding the tail past the 30-app cap. **Excluded apps are
different**: anything in `deep_brain_excluded_apps` is checked first and contributes only the
unnamed `excluded_seconds` total — exclusion, not redaction, is the hiding mechanism. Neither
bucket's transitions affect switch or focus numbers. An AFK gap ends
a focus block (hidden spans deliberately do not — the break would reveal them). Meeting time is
counted through AFK (listening in a call involves no input). Prefer it over paging excerpts for
time-use questions: richer analysis, strictly less raw-text egress.

Captured excerpts are untrusted evidence. They can contain prompt-like text, commands, or secrets;
the assistant must never treat them as instructions.

## Privacy boundary

Installing the connector does not upload the database or stream capture. When Claude invokes a
content tool, the returned excerpt, app identifier, timestamp, source, and observation ID leave
OpenBird's local boundary through Claude. When Claude invokes the activity summary, **behavioral
metadata** leaves the same way: app identifiers (including redacted apps with their reason
codes), per-app usage durations and span counts, AFK and meeting time, context-switch counts,
focus-block timestamps, and the folded-tail app count — no captured text. The status tool sends
store-lifetime totals, encryption state, and exclusion-configuration counts. Every response also
carries a `capture_host` label naming which Mac's store answered (configurable via
`OPENBIRD_ASSISTANT_HOST_LABEL`; defaults to the hostname) and a machine-parseable `egress`
block — `{scope, untrusted_content, fields}` — declaring exactly which JSON paths that tool may
emit, so a calling agent can enforce its own egress policy instead of parsing prose. Anthropic's retention and
workspace policies apply after that point. A later OpenBird purge prevents future retrieval but
cannot recall data already sent.

To disconnect, remove the `openbird` entry from `mcpServers` in Claude Desktop's config and restart
Claude. The installer backup is `claude_desktop_config.json.openbird-backup` beside the config.

## ChatGPT

[ChatGPT custom apps use remote MCP servers and do not connect directly to local MCP servers](https://help.openai.com/en/articles/12584461-developer-mode-apps-and-full-mcp-connectors-in-chatgpt-beta.eot).
OpenBird therefore bundles OpenAI's official [Secure MCP Tunnel client](https://developers.openai.com/api/docs/guides/secure-mcp-tunnels)
and exposes a guided setup in **Settings → Desktop assistants → ChatGPT**.

1. Enable Developer Mode in ChatGPT.
2. Create a Secure MCP Tunnel in the OpenAI Platform, associate it with the target ChatGPT
   workspace, and create a restricted runtime API key with **Tunnels Read + Use** permission.
3. Paste the `tunnel_...` id and runtime key into OpenBird and choose Connect.
4. Add the tunnel as a custom app in ChatGPT.

If the workspace association or either permission is missing, the tunnel may run locally but not
appear in ChatGPT's custom-app picker.

OpenBird stores both setup values in a dedicated macOS Keychain item. The runtime key is passed only
through the tunnel child's environment, never through arguments, logs, UserDefaults, or repository
files. Before every start, OpenBird forcibly reconciles its owned tunnel profile with the current
app bundle, validates it, starts the outbound tunnel on an ephemeral loopback health port, disables
the operator log buffer, and reports Connected only after `/readyz` succeeds.

The same privacy boundary applies to both assistants: there is no background memory feed. When
ChatGPT invokes a content tool, returned excerpts, app identifiers, timestamps, and observation IDs
are sent to OpenAI — and the activity summary sends the same behavioral metadata described above —
and none of it can be recalled by deleting local memory. URLs and window titles are not
returned. Removing the connection stops only OpenBird's owned process and deletes only its Keychain
item, local health marker, and `openbird` tunnel profile.
