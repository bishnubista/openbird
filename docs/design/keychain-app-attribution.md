# Design: Keychain attribution fix (app-owned DB key)

## Problem — Keychain prompt says "python3.13" and re-prompts forever

The macOS Keychain dialog reads *"python3.13 wants to use your confidential
information stored in openbird"* and reappears on every run; "Always Allow"
never sticks.

**Root cause (verified):** `OpenBird.app` shells out to `openbird-cli`, which
runs Python; Python calls `keyring.get_password("openbird", "db-encryption-key")`
in `openbird/storage/crypto.py:197`. The keychain ACL binds to the **Designated
Requirement** of the *requesting* process. The accessor is an ad-hoc / linker-
signed interpreter (`.venv/bin/python`, `Identifier=-`, `Signature=adhoc`), whose
DR is just a cdhash that changes every run → re-prompt forever, and a bare Mach-O
with no `Info.plist` → dialog shows the file name `python3.13`. Even the notarized
DMG's embedded Developer-ID python would still display `python3.13` (no app name)
— only persistence improves. Current encryption state: OFF (`plaintext-0600`,
sqlcipher not in env), so these prompts currently buy nothing.

## Fix — Swift `OpenBird.app` owns the key; Python reads it from env

`crypto.py:185` already prefers `OPENBIRD_DB_KEY` (resolution path #1) over the
keychain, explicitly so the signed app can supply it. The app has a stable
Developer-ID DR and an `Info.plist` (display name "OpenBird").

1. New `KeychainKeyProvider` (Security framework) in the Swift app:
   - operate on a generic-password item, **service `openbird`, account
     `db-encryption-key`** (must match `_KEYRING_SERVICE` / `_KEYRING_USER`).
   - value = 64-char lowercase hex of 32 random bytes (`SecRandomCopyBytes`),
     matching Python's `secrets.token_hex(32)` so an existing encrypted DB stays
     decryptable. **Confirmed by Codex:** `crypto.py` consumes `OPENBIRD_DB_KEY`
     verbatim into the same SQLCipher `x'...'` key path as `token_hex(32)`.
   - accessibility: `kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly` (readable
     by a background capture daemon after first unlock; never iCloud-synced).
2. **Migration — read-only, never delete (Codex finding #1).** Try
   `SecItemCopyMatching` first. If an item exists, **reuse the value as-is** — do
   NOT delete-and-re-add (a delete that succeeds before a failed add would lose
   the only key and strand the encrypted DB). A python-owned item prompts **once**,
   now attributed to *OpenBird*; "Always Allow" adds the app (stable Developer-ID
   DR) to the item's ACL, so every later read is silent. The only copy is never at
   risk because we never delete it.
3. **Generate only when provably safe (Codex finding #2).** On
   `errSecItemNotFound`, before generating a new key, inspect the **resolved DB
   file** mirroring Python `Settings` precedence (`config.py`): `OPENBIRD_DB_PATH`
   if set, else `$OPENBIRD_DATA_DIR/openbird.db`, else `~/.openbird/openbird.db`.
   Read the first 16 bytes of *that exact file only*, not the directory — checking
   only the default could strand a custom-path encrypted DB. If the file is absent/empty, or those
   bytes are the plaintext magic `SQLite format 3\0`, it is safe to generate +
   `SecItemAdd` (app owns it → no prompt ever). If a **non-empty DB file lacks the
   plaintext magic** (i.e. looks encrypted) but no key item exists, **fail
   closed**: log `stranded-db`, skip injection, surface a recovery state — never
   mint a fresh key that can't open the existing DB.
4. **Injection — explicit overlay, not just inherited env (Codex finding #3).**
   Store the resolved key once; build every CLI child's environment by explicitly
   overlaying `OPENBIRD_DB_KEY` (centralized helper used by `askChat`, `preflight`,
   `memoryStats`, **and `startCapture`** — which today rebuilds env from a
   `ProcessInfo` snapshot that a post-launch `setenv` might miss). Keep a
   `setenv` at launch as a belt-and-suspenders default. Python's path #1 then
   short-circuits keyring in every child.
5. **Read denied:** if an item exists but the read is **denied**, do NOT generate a
   new key — log `denied`, skip injection, let Python fall back to its existing
   behavior.
6. **Observability:** log reason codes (`created` / `loaded` / `denied` /
   `stranded-db` / `error`); never log the key.

**Tradeoff:** the key now lives in the app's environment, inherited by all
descendants (incl. helpers). Accepted — it is the documented `OPENBIRD_DB_KEY`
channel; argv is still never used for secrets.

## Out of scope
- Standalone Spotlight-style quick-chat panel — **deferred** to a later PR.
- Enabling encryption (sqlcipher install); changing activation policy.
