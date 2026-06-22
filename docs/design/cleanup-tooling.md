# Cleanup tooling — uninstall + dev Launch-Services hygiene

## Motivation (root causes observed)

A tester's machine reached a broken state where OpenBird vanished from the menu bar
and the Keychain prompt read `python3.13` instead of `OpenBird`. Investigation found
**two** distinct problems:

1. **Self-corrupting code signature (fixed in the DMG seal guard).** The embedded Python writes
   `__pycache__/*.pyc` into the *signed* app bundle on first run
   (`Resources/python/lib/python3.13/encodings/__pycache__/aliases.cpython-313.pyc`
   was modified post-signing), breaking the Developer ID seal. Because the macOS
   Keychain ACL is keyed to the app's code signature, a broken seal makes the app
   unrecognizable → `crypto.py` falls back to `keyring` → the `python3.13` prompt;
   it also fails Gatekeeper (`spctl`) and destabilises launch/`MenuBarExtra`.
   The DMG packager now rebuilds hash-based `.pyc` files before signing, disables
   runtime bytecode writes in the bundled CLI wrapper, and fails the relocation
   smoke test if a bundled CLI launch adds or mutates Python bytecode. The Swift
   app also overlays `PYTHONDONTWRITEBYTECODE=1` for CLI children as defense in
   depth.

2. **Launch Services pollution (this work).** The machine had 8+ registrations of
   bundle id `ai.openbird.OpenBird` — five *ghost* `dist/OpenBird.app` from build
   worktrees, a mounted-dmg copy, and a zombie launchd job for a deleted bundle.
   macOS resolves a bundle id to *any* registered copy, so launching could fire a
   stale dev build. There is also no clean way to remove OpenBird's system state
   when a user trashes the app: the Keychain key, the routines LaunchAgent, and the
   LS registration all linger.

This work delivers **cleanup tooling** for both the dev loop and end-user uninstall.

## Deliverable 1 — `openbird uninstall` (system state only by default)

New top-level Typer command. Removes OpenBird's **system state** but **preserves
captured data** (`~/.openbird`) unless `--purge-data` is given.

Flags:
- `--purge-data` — also delete the data dir. Irreversible.
- `--yes` / `-y` — skip the confirmation prompt (mirrors `data purge`).
- `--dry-run` — print exactly what would be removed; touch nothing.

### Non-creating path resolution (Codex #2)

`get_settings()` MUST NOT be used: `Settings.__post_init__` calls
`data_dir.mkdir(...)` + `os.chmod(...)` (`config.py:94`), so a `--dry-run` (or any
uninstall on a clean machine) would *create* `~/.openbird`. Add a side-effect-free
resolver mirroring `config._default_data_dir()` precedence
(`OPENBIRD_DATA_DIR` → `~/.openbird`, expanduser, **no mkdir/chmod**) and a DB-path
resolver (`OPENBIRD_DB_PATH` → `<data_dir>/openbird.db`). Uninstall uses only these.

### The single key-safety rule (Codex #1, round-2 #1)

The Keychain key is deleted **iff no encrypted DB depends on it**, expressed in
terms of the **resolved DB path** (not the data dir) so a custom `OPENBIRD_DB_PATH`
outside the data dir (`config.py:59`, honored independently of `data_dir`) cannot be
stranded. Evaluated **after** all data removal:

1. Perform data removal first: `--purge-data` → `rmtree(data_dir)`. Note this
   removes the *default* DB (`<data_dir>/openbird.db`) but **not** a custom DB that
   resolves outside `data_dir` — we never delete a user-pointed external path.
2. Re-resolve the DB path and inspect it on disk. Delete the key **only if** the DB
   file is **absent/empty** OR its header **==** `SQLite format 3\0` (plaintext).
3. Otherwise — any **non-empty** DB **lacking** the plaintext magic looks
   SQLCipher-encrypted — **retain the key** and report that an encrypted DB still
   depends on it (default purge that partially failed, or a surviving custom-path
   DB). Deleting it would strand that DB.

This one rule covers every case: no-purge, `--purge-data`, partial `rmtree` failure,
and custom `OPENBIRD_DB_PATH`.

### Steps (each best-effort; collect a result line per step, report at end)

1. **Routines launchd job + LaunchAgent (Codex #3).** A job can be *loaded* under
   label `ai.openbird.routines` (`launchd.py:24` `AGENT_LABEL`) even after its
   plist is gone (the observed zombie). So: best-effort `launchctl bootout
   gui/<uid>/ai.openbird.routines` (fallback `launchctl remove ai.openbird.routines`),
   tolerating "not loaded"; **then** unlink `agent_plist_path()` if present. This
   clears an orphaned job the plist-only path in `routine uninstall` cannot.
2. **Launch Services** — unregister OpenBird app registrations, but **validate
   `CFBundleIdentifier == ai.openbird.OpenBird`** on each bundle before
   `lsregister -u` (Codex #5) so an unrelated `OpenBird.app` is never touched.
   Ghost paths (registration whose bundle no longer exists) are unregistered
   directly. macOS-only; skipped elsewhere.
3. **Pause sidecar** — remove `capture.paused` in the data dir (the only runtime
   sidecar today; `_SIDECAR_NAMES` is a tuple so a future lock/pid file slots in).
4. **Key + data** — apply the key-safety rule above (data removal, then conditional
   key delete via a new `crypto.delete_key()`, tolerant of not-found).

Confirmation: unless `--yes`, prompt summarising what will be removed (and whether
data + key are preserved). `--dry-run` implies no prompt and no mutation.

Privacy/safety: print only paths and reason codes, never captured content. Never
delete outside the resolved data dir, the known plist path, and validated app bundles.

## Deliverable 2 — `script/dev_cleanup.sh` (dev cache hygiene)

Bash script (`set -euo pipefail`, shellcheck-clean) to run after a dev build/test
cycle so stale `dist/OpenBird.app` builds stop polluting Launch Services.

**Targeting (Codex #4, #5) — positively match dev/build paths; never real installs.**
For each `OpenBird.app` path in `lsregister -dump`, unregister it **only** when:
- the path **does not exist** on disk (a ghost registration — points nowhere), OR
- the path matches a dev/build pattern (`*/dist/OpenBird.app` or
  `*/dist/dmg-stage/OpenBird.app`) **AND** its `CFBundleIdentifier` is
  `ai.openbird.OpenBird` (read from `Contents/Info.plist`).

`/Applications/OpenBird.app` is kept by default and unregistered **only with
`--all`** (after the same `CFBundleIdentifier` validation). **Never** unregister —
even with `--all` — anything under the Homebrew prefix (`brew --prefix`,
Cellar/opt/**`libexec/OpenBird.app`** — the formula installs there and launches via
`openbird-app`, `Formula/openbird.rb`), `/opt/homebrew`, or `/usr/local`.

Behavior:
1. Quit a running OpenBird via `osascript -e 'quit app "OpenBird"'` (best-effort).
2. Apply the targeting rule above; `lsregister -u <path>` each match. `--all`
   additionally unregisters a validated `/Applications/OpenBird.app` (still never
   Homebrew/system roots).
3. `--dist` — also `rm -rf` `dist/OpenBird.app` build artifacts under the repo.
4. Print a before/after registration count.

Wiring: callable standalone and documented in CLAUDE.md "Handy commands". It is
**not** auto-invoked from `script/build_and_run.sh` — that script builds *and may
launch* `dist/OpenBird.app`, so cleaning on its exit would unregister the very app
it just launched. The script's job is to purge ghosts/stale dev builds *between*
cycles; run it manually (or before a fresh build), and only ghost + dev-build paths
are ever touched.

**Recommendation (noted, not implemented here):** give dev builds a distinct bundle
id (e.g. `ai.openbird.OpenBird.dev`) so they can never shadow the release id in LS.
Cleaner long-term fix; deferred to keep this PR minimal.

## Testing

- `openbird uninstall --dry-run` and `--purge-data` paths unit-tested with a temp
  `OPENBIRD_DATA_DIR`, a faked keyring, and the stranding-guard branch (encrypted vs
  plaintext header) — no real Keychain or launchctl calls (monkeypatched).
- `shellcheck script/dev_cleanup.sh`.
- `uv run python -m pytest -q` green; CLI `--help` still loads without macOS deps.

## Non-goals

- The bytecode/seal fix (separate PR — the actual root cause of the prompt).
- An automatic on-delete trigger: macOS runs no hook when an app is trashed; the
  uninstall command + a dmg-bundled "Uninstall" affordance is the supported path.
