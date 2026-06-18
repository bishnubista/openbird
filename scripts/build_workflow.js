import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

export const meta = {
  name: 'openbird-build',
  description: 'Build the OpenBird local-first personal memory app: foundation, subsystems, Codex review, integration',
  phases: [
    { title: 'Foundation', detail: 'shared contracts: types, config, memory, llm provider, crypto + tests' },
    { title: 'Subsystems', detail: 'capture, meetings, chat, routines, integrations, preflight (parallel)' },
    { title: 'CodexReview', detail: 'Codex adversarial review per subsystem' },
    { title: 'Fix', detail: 'apply confirmed Codex findings' },
    { title: 'Integration', detail: 'wire CLI, swift helpers, run full test suite' },
  ],
}

const REPO = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const shellQuote = (value) => `'${String(value).replace(/'/g, `'\\''`)}'`
const REPO_SH = shellQuote(REPO)

const COMMON = `
You are building OpenBird, an open-source local-first personal memory app, in the repo at ${REPO}.
READ ${REPO}/PLAN.md first — it is the approved spec (Revision 4) and is authoritative.
Environment: Python via 'uv' (run code with 'uv run python', tests with 'uv run pytest'); Ollama is
running locally with models 'llama3.2' and 'nomic-embed-text' (embeddings dim = 768). sqlite-vec 0.1.9
and FTS5 are available. Write real, working, idiomatic code with type hints and docstrings. Keep edits
strictly inside your assigned files. After writing, RUN your tests with 'uv run pytest <your test paths>'
and iterate until they pass. Return a concise JSON status.`

// ---- Frozen foundation contracts that every subsystem imports ----
const CONTRACTS = `
FROZEN CONTRACTS (do not change signatures; subsystems depend on these):

openbird/types.py (pydantic v2 models):
  Observation(id:str, content_hash:str, ts:float, app:str|None, window:str|None, url:str|None, session_id:str|None, source:str)
  ContentBlob(content_hash:str, text:str)
  Chunk(id:str, content_hash:str, span:tuple[int,int], text:str)
  SearchHit(chunk_id:str, content_hash:str, text:str, score:float, observation:Observation|None)
  Citation(observation_id:str, app:str|None, window:str|None, ts:float, snippet:str)
  RoutineRun(id:str, routine:str, scheduled_ts:float, started_ts:float|None, finished_ts:float|None, status:str, output:str|None, idempotency_key:str)

openbird/config.py:
  Settings (pydantic-settings or plain dataclass) with: data_dir (~/.openbird), db_path,
  llm_model='ollama/llama3.2', embed_model='ollama/nomic-embed-text', embed_dim=768,
  allowlist:list[str], blocklist:list[str], ocr_enabled=False (default off), encryption_enabled (from gate).
  get_settings() returns a cached Settings honoring OPENBIRD_* env overrides.

openbird/llm/provider.py:
  class LLMProvider:
    embed(texts:list[str]) -> list[list[float]]   # via litellm 'ollama/nomic-embed-text', asserts dim==embed_dim
    complete(messages:list[dict], *, json_schema:dict|None=None) -> str|dict  # validate+retry on json_schema (best-effort)
    cohort_key() -> str   # stable id of (provider, model, dim, normalized) for embedding-cohort guarding

openbird/memory/store.py:
  class MemoryStore:
    __init__(db_path:str|None=None)   # loads sqlite-vec, applies schema.sql
    add_observation(text:str, *, app=None, window=None, url=None, session_id=None, source:str, ts:float|None=None) -> Observation
        # normalize+chunk text; CHUNK-LEVEL content-hash dedup; ALWAYS insert a new observation row
        # (never deduped); embed each UNIQUE chunk once; index chunks in fts + vec (per-chunk, dim 768)
    search(query:str, k:int=10, *, semantic:bool=True) -> list[SearchHit]   # vector+BM25 -> RRF -> MMR dedup; resolves observation
    time_range(start_ts:float, end_ts:float) -> list[Observation]   # range-scan, not semantic
    delete(*, since_ts:float|None=None, all:bool=False) -> int   # cascade across blobs/observations/chunks/fts/vec
    stats() -> dict
  openbird/memory/schema.sql: content_blobs, observations, chunks(text), fts5(OVER CHUNKS), vec_chunks(per-chunk, dim 768), embedding_meta
  NOTE [R5]: indexing is CHUNK-LEVEL (fts over chunks, vectors per chunk), each chunk maps content_hash->blob->observations.
  openbird/memory/ingest.py: normalize(text)->str, chunk(text)->list[(span,text)], content_hash(text)->str
  openbird/memory/search.py: rrf(rankings)->fused, mmr(hits)->deduped

openbird/storage/crypto.py:
  open_encrypted_db(path) -> sqlite3.Connection  # tries sqlcipher3 + sqlite-vec; if unavailable, falls back to
  plain sqlite3 with 0600 perms and sets settings.encryption_enabled=False. keyring for the key. Document the result.`

phase('Foundation')
const foundation = await agent(`${COMMON}\n${CONTRACTS}\n
TASK: Build the FOUNDATION exactly per the contracts above. Create:
- openbird/types.py, openbird/config.py, openbird/storage/crypto.py, openbird/storage/__init__.py
- openbird/memory/{schema.sql, store.py, ingest.py, search.py}
- openbird/llm/provider.py
- tests/unit/test_memory.py (dedup keeps 1 blob but N observations; cascade delete; time_range correctness),
  tests/unit/test_search.py (rrf + mmr), tests/unit/test_provider.py (embed dim guard with a fake/mocked embed;
  do NOT require ollama in unit tests — mock litellm), tests/unit/test_ingest.py (normalize/chunk/hash).
Make tests runnable WITHOUT network/ollama by mocking the embedding call. Run 'uv run pytest tests/unit -q'
until green. Return JSON {files:[...], tests_passed:bool, notes:str, key_decisions:str}.`,
  { phase: 'Foundation', schema: { type:'object', additionalProperties:false,
    required:['files','tests_passed','notes'], properties:{
      files:{type:'array',items:{type:'string'}}, tests_passed:{type:'boolean'},
      notes:{type:'string'}, key_decisions:{type:'string'} } } })

log(`Foundation done: tests_passed=${foundation?.tests_passed}. ${foundation?.notes||''}`)

// ---- Subsystems (parallel, disjoint dirs, import frozen foundation) ----
const SUBSYSTEMS = [
  { key:'capture', dir:'openbird/capture', files:'daemon.py, redact.py, adapters.py',
    spec:`Capture orchestration in Python (the Swift helper is built later). daemon.py: run a capture
    helper (accept an injectable command for tests; default points at the swift binary), parse its JSON
    {app,window,url,text,ts}, apply redact.py, then MemoryStore.add_observation(...). redact.py:
    allowlist-first (only allowlisted apps captured), regex secrets + blocklist (password managers,
    finance/health apps), skip incognito; defense-in-depth, documented as non-guarantee. adapters.py:
    per-app normalization strategies + a compatibility-matrix dict. Tests with a FAKE helper emitting
    canned JSON (no real AX needed).` },
  { key:'chat', dir:'openbird/chat', files:'rag.py',
    spec:`rag.py: answer(query) -> retrieve via MemoryStore.search -> dedupe by document/session ->
    build grounded prompt (retrieved text clearly delimited as UNTRUSTED data; never obey it as
    instructions) -> LLMProvider.complete -> validate citations (only returned observation ids may be
    cited; repair/reject hallucinated). Return answer + Citation list. Unit test with a fake LLMProvider
    + in-memory MemoryStore (no ollama). Add one integration test marked to skip if ollama is absent.` },
  { key:'routines', dir:'openbird/routines', files:'scheduler.py, templates.py, store.py',
    spec:`store.py: durable routine_runs table (idempotency_key, status) + run-missed-on-startup.
    scheduler.py: APScheduler wrapper, register routine = (name, prompt, interval). templates.py:
    daily-briefing, yesterday's-work, weekly-summary as time-range RAG queries (read/summarize only).
    Deliver to stdout + store as a RoutineRun. Unit-test idempotency + missed-job catchup with a fake clock.` },
  { key:'meetings', dir:'openbird/meetings', files:'audio.py, transcribe.py, pipeline.py',
    spec:`Python side of meetings (Swift audio-helper built later). audio.py: wrap an injectable audio
    source; document ScreenCaptureKit + separate mic track + clock-sync requirement (no ffmpeg
    avfoundation for system output). pipeline.py: VAD/window/stitch over PCM frames (pure-python stub
    that operates on provided frames; do NOT require real audio). transcribe.py: faster-whisper wrapper
    behind a try-import; if faster-whisper not installed, expose a clear 'meetings extra not installed'
    path. Summary+action-items via LLMProvider with the json_schema validate+retry. Tests use canned
    transcript segments (no audio, no whisper needed).` },
  { key:'integrations', dir:'openbird/integrations', files:'mcp.py',
    spec:`mcp.py: an MCP client registry loaded from config. MVP = a local FILESYSTEM MCP connector only
    (clearly labeled); pull file contents into MemoryStore as observations. Write actions disabled.
    Make 'mcp' an optional import (extra); degrade gracefully if not installed. Unit-test the registry +
    a fake connector (no real MCP server needed).` },
  { key:'preflight', dir:'openbird', files:'preflight.py',
    spec:`preflight.py: a function returning a dict reporting: ollama reachable + models present,
    embedding dim, sqlite-vec/FTS5 availability, DB encryption status (from crypto), allowlist/blocklist,
    OCR flag, and (best-effort, may be 'unknown' off-mac) TCC/Accessibility + audio capability stubs.
    Pure-python, no failures if a check can't run — report 'unknown'. Unit-test the aggregation with fakes.` },
]

phase('Subsystems')
const built = await parallel(SUBSYSTEMS.map(s => () =>
  agent(`${COMMON}\n${CONTRACTS}\n
TASK: Build the '${s.key}' subsystem ONLY. Files you may create/edit: ${s.dir}/{${s.files}} and
tests/unit/test_${s.key}.py (+ ${s.dir}/__init__.py if missing). Import the frozen foundation; DO NOT
edit foundation, config, types, cli.py, pyproject, or other subsystems.
SPEC: ${s.spec}
Run 'uv run pytest tests/unit/test_${s.key}.py -q' until green. Return JSON.`,
    { label:`build:${s.key}`, phase:'Subsystems', schema:{ type:'object', additionalProperties:false,
      required:['subsystem','files','tests_passed','notes'], properties:{
        subsystem:{type:'string'}, files:{type:'array',items:{type:'string'}},
        tests_passed:{type:'boolean'}, notes:{type:'string'} } } })
)).then(rs => rs.filter(Boolean))

log(`Subsystems built: ${built.map(b=>`${b.subsystem}:${b.tests_passed?'ok':'FAIL'}`).join(', ')}`)

// ---- Codex adversarial review per subsystem ----
phase('CodexReview')
const reviews = await parallel(SUBSYSTEMS.map(s => () =>
  agent(`Run an ADVERSARIAL code review of the '${s.key}' subsystem of OpenBird at ${REPO}.
Use the Codex CLI non-interactively:
  cd ${REPO_SH} && codex exec -s read-only -c approval_policy='"never"' --color never "Adversarially review the files in ${s.dir} and tests/unit/test_${s.key}.py for correctness bugs, races, resource leaks, injection (esp. prompt-injection for chat/routines), privacy leaks of captured content, and contract violations vs PLAN.md. List concrete findings with severity (critical/high/medium/low) and a fix. Be specific with file:line."
Capture stdout. Then summarize the findings as JSON.`,
    { label:`codex:${s.key}`, phase:'CodexReview', schema:{ type:'object', additionalProperties:false,
      required:['subsystem','findings'], properties:{ subsystem:{type:'string'},
        findings:{type:'array', items:{type:'object', additionalProperties:false,
          required:['severity','issue','fix'], properties:{
            severity:{type:'string'}, issue:{type:'string'}, fix:{type:'string'} }}} } } })
)).then(rs => rs.filter(Boolean))

const actionable = reviews.flatMap(r => (r.findings||[]).filter(f =>
  ['critical','high'].includes((f.severity||'').toLowerCase())).map(f => ({...f, subsystem:r.subsystem})))
log(`Codex review: ${actionable.length} high/critical findings across subsystems`)

// ---- Fix pass (only where there are high/critical findings) ----
phase('Fix')
const bySub = {}
for (const f of actionable) (bySub[f.subsystem] ||= []).push(f)
await parallel(Object.entries(bySub).map(([sub, findings]) => () => {
  const s = SUBSYSTEMS.find(x => x.key === sub) || { dir:`openbird/${sub}`, files:'' }
  return agent(`${COMMON}\nTASK: Fix these Codex high/critical findings in the '${sub}' subsystem
(files under ${s.dir} and tests/unit/test_${sub}.py only). Apply real fixes, then re-run
'uv run pytest tests/unit/test_${sub}.py -q' until green.
FINDINGS:\n${findings.map((f,i)=>`${i+1}. [${f.severity}] ${f.issue}\n   fix: ${f.fix}`).join('\n')}
Return JSON {subsystem, fixed:[...], tests_passed:bool}.`,
    { label:`fix:${sub}`, phase:'Fix', schema:{ type:'object', additionalProperties:false,
      required:['subsystem','tests_passed'], properties:{ subsystem:{type:'string'},
        fixed:{type:'array',items:{type:'string'}}, tests_passed:{type:'boolean'} } } })
}))

// ---- Integration: CLI wiring + swift helpers + full suite ----
phase('Integration')
const integ = await agent(`${COMMON}\n${CONTRACTS}\n
TASK (integration owner — you MAY edit shared files now):
1. Write openbird/cli.py (Typer) wiring commands: capture (run daemon once over a fake/real helper),
   chat "<q>", ingest <path>, routine run <name>, routine list, meeting (stub), preflight, data purge --since.
   Ensure 'openbird' entrypoint works: 'uv run openbird preflight'.
2. Create the Swift helpers so they COMPILE: capture-helper/ (Package.swift + Sources/CaptureHelper/main.swift
   using AXUIElement to print active-window JSON; include AXIsProcessTrustedWithOptions prompt + depth/time
   limits) and audio-helper/ (Package.swift + a minimal ScreenCaptureKit skeleton that compiles). Run
   'swift build' in each and fix compile errors (functionality needs macOS permissions, that's fine).
3. Write tests/integration/test_e2e.py: ingest sample text -> chat retrieves it with a citation, using a
   fake LLMProvider so it runs WITHOUT ollama; plus one ollama-gated test skipped if unreachable.
4. Run the FULL suite 'uv run pytest -q' and fix failures. Run 'uv run openbird preflight' and capture output.
Return JSON {cli_ok:bool, swift_capture_builds:bool, swift_audio_builds:bool, full_suite_passed:bool,
test_summary:str, remaining_issues:str}.`,
  { phase:'Integration', schema:{ type:'object', additionalProperties:false,
    required:['cli_ok','full_suite_passed','test_summary'], properties:{
      cli_ok:{type:'boolean'}, swift_capture_builds:{type:'boolean'}, swift_audio_builds:{type:'boolean'},
      full_suite_passed:{type:'boolean'}, test_summary:{type:'string'}, remaining_issues:{type:'string'} } } })

return {
  foundation_tests: foundation?.tests_passed,
  subsystems: built.map(b => ({ name:b.subsystem, ok:b.tests_passed })),
  codex_high_critical: actionable.length,
  integration: integ,
}
