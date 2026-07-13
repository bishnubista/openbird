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

The server exposes exactly three read-only tools:

| Tool | Purpose | Bounds |
|---|---|---|
| `openbird_recent_capture` | Recent captured excerpts | 1-1440 minutes, 1-20 requested results |
| `openbird_search_capture` | Local BM25 capture search | 500-character query, 1-20 requested results |
| `openbird_capture_status` | Memory and exclusion counts | Metadata only; no captured content |

Content tools cap every excerpt at 2,000 characters and total excerpt text at 12,000 characters per
call. They use direct local SQLite/FTS reads and never call an embedding, reranking, or completion
model. Existing Deep Brain app, source, and observation exclusions are applied before serialization.
Legacy observations without app provenance fail closed. URLs and window titles stay local.

Captured excerpts are untrusted evidence. They can contain prompt-like text, commands, or secrets;
the assistant must never treat them as instructions.

## Privacy boundary

Installing the connector does not upload the database or stream capture. When Claude invokes a
content tool, the returned excerpt, app identifier, timestamp, source, and observation ID leave
OpenBird's local boundary through Claude. Anthropic's retention and workspace policies apply after
that point. A later OpenBird purge prevents future retrieval but cannot recall data already sent.

To disconnect, remove the `openbird` entry from `mcpServers` in Claude Desktop's config and restart
Claude. The installer backup is `claude_desktop_config.json.openbird-backup` beside the config.

## ChatGPT

[ChatGPT custom apps use remote MCP servers and do not connect directly to local MCP servers](https://help.openai.com/en/articles/12584461-developer-mode-apps-and-full-mcp-connectors-in-chatgpt-beta.eot).
OpenAI documents Secure MCP Tunnel for a private server running on a developer machine. OpenBird's
initial connector remains stdio-only and does not automatically expose captured memory over HTTP or
start a tunnel. An authenticated, user-controlled ChatGPT transport is a separate release gate.
