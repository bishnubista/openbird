# Meeting recording MVP

## Outcome

Make the installed Apple-Silicon macOS app capable of explicitly recording a
meeting, transcribing system and microphone audio locally, and saving a durable,
searchable transcript that Ask can cite. Today owns the recording control and
status; widening Today briefing/timeline aggregation beyond `source="capture"` is
not part of this MVP.

## Product contract

- Recording is always manually started and stopped. Meeting-app detection never
  auto-records.
- Today and the native menu-bar menu both expose Start/Stop Meeting Recording.
- First use asks the user to download the named approximately 2.51 GB Parakeet model from
  Hugging Face into OpenBird's local model cache. The dialog names the destination,
  says no user data is uploaded, reports downloaded/total bytes, and supports
  Cancel. Partial downloads remain resumable; offline failure returns to idle with
  a retry message. The pre-network estimate, README, and tests use one package
  constant pinned to the public model manifest; progress switches to the manifest's
  resolved total after consent.
- The UI stays `preparing` until the model is cached and loadable. Only the CLI's
  metadata-only `recording_started` event moves the UI to `recording`.
- Recording shows an unambiguous persistent red state and elapsed time. Start copy
  tells the user to record only after everyone agrees.
- Raw PCM stays in process memory and private pipes. It is never written to disk,
  argv, environment, stdout/stderr logs, or crash messages.
- The saved transcript uses `source="meeting"`, stores the meeting UUID in the
  existing `session_id`, and encodes relative timing literally as
  `[seconds-seconds] me|others: text` in the observation body.
- Empty/no-transcribed-speech meetings create no observation. The energy-VAD seam
  is accepted only after the real-fixture gate below; the UI says "No speech was
  transcribed," not that silence was perfectly classified.
- Early helper exit, ASR queue overflow, per-window/backend transcription failure,
  finalization timeout, and persistence failure are distinct metadata-only outcomes.
  Partial results are labeled and never silently presented as complete.

## Python recording controller

Add `openbird/meetings/record.py` as the orchestration seam.

### Owned processes and supervision

1. Resolve the signed helper only from `OPENBIRD_AUDIO_HELPER` (or an explicit
   hidden test override), validate it is executable, and keep it unopened until
   supervision and storage are ready.
2. The app launches `openbird meeting record --jsonl` with a meeting-specific,
   token-authenticated death pipe (`OPENBIRD_SUPERVISOR_TOKEN` on stdin). Unlike
   capture's intentionally fail-open watcher, the meeting CLI must read the exact
   token within five seconds **before opening the database or launching the audio
   helper**. Wrong/short token, early EOF, over-long input, timeout, or read error is
   `supervisor_not_armed` and exits fail-closed without activating a microphone.
   Once armed, EOF stops the owned helper immediately and begins bounded finalization.
3. SIGINT/SIGTERM and a valid supervisor EOF all mark `stop_requested`, terminate
   only the owned helper PID, and drain a complete trailing frame.
4. Extend the IPC decoder with a strict recording mode that distinguishes clean EOF
   from truncated/corrupt records. Helper EOF before a requested stop becomes
   `helper_ended_early`; any completed transcript pieces are checkpointed/persisted as
   partial.
5. Launch the owned helper with `--max-seconds 14400` (four hours) as a hard privacy
   backstop. Normal meeting stop operates only on the owned CLI/helper process tree.
   Split the Settings action into capture cleanup and a clearly labeled emergency
   **Force Stop Meeting Audio** action that terminates the owned process and can
   exact-name kill OpenBird's `audio-helper` if ownership IPC is unavailable. Thus
   capture cleanup never kills an active meeting, while a user retains a reachable
   hot-mic escape hatch.

### Streaming ASR and backpressure

1. Decode `BinaryFrameAudioSource` and feed frames continuously into the existing
   per-track `MeetingPipeline`.
2. Send closed speech windows, each with a monotonic index, to one dedicated ASR
   worker thread through a private bounded `queue.Queue`. The worker owns the model
   and returns transcript results through a second bounded in-process queue. This
   deliberately avoids introducing an unproved `multiprocessing.spawn`/resource-
   tracker mechanism inside the relocated hardened-runtime bundle; the completed
   spike covers MLX inference in this process. Fake-blocking and real-backend gates
   below prove the reader continues draining while inference runs.
3. Queue capacity is eight 25-second windows. The reader never blocks the Swift
   helper's synchronous audio callbacks: when full it drops the newest closed window,
   increments `dropped_windows`, and keeps draining PCM. The final result and UI
   explicitly say the transcript is partial when this counter is non-zero.
4. Results are reordered by window index before existing stitch logic runs.
5. A failed window increments `failed_windows` and records only a closed backend
   reason; the worker continues with later windows when safe. A model-level fatal
   failure drains without blocking, fails outstanding windows, and yields
   `transcription_failed` rather than disappearing. On Stop, emit JSONL `finalizing`
   progress with completed/remaining/dropped/failed window
   counts. The target is Parakeet RTF <= 0.20 and faster-whisper RTF <= 1.0 on the
   supported validation Mac. Finalization has a 180-second deadline. At the deadline,
   stop accepting worker results, checkpoint/persist completed text, count outstanding windows
   as dropped, and report a partial result.
6. The result contains only meeting id, duration, segment/window counts, observation
   id, actual backend, clock-event count, dropped/failed-window counts, partial flag,
   and a closed reason code. No transcript text is emitted by the CLI.

### Durable SQLCipher-gated checkpoint and persistence

Add a schema migration for `pending_meetings`, keyed by meeting UUID, inside the
existing memory database. It contains the versioned transcript checkpoint, wall-clock
start/end, relative segment metadata, partial reason, and resulting observation id.
It is protected by the exact same SQLCipher/release-policy gate as observations—no
new plaintext content store or release carve-out exists. `meeting record` opens and
validates the store before starting audio, inserts the initial pending row, and commits
each ordered ASR result/checkpoint in a short transaction. The row is deleted only in
the transaction that commits the observation/indexes (or after idempotent recovery
proves that matching `source + session_id` observation already exists).

- Cap one checkpoint/transcript at 8 MiB UTF-8. If reached, stop accepting further text,
  increment a truncation counter, and report partial; continue draining audio so the
  helper never blocks.
- Change `MemoryStore.add_observation` to embed new chunks in bounded batches (32
  chunks per provider call) outside the write transaction instead of one unbounded
  document-sized request.
- Persist one observation with `app="OpenBird Meetings"`, `window="Recorded meeting"`,
  `source="meeting"`, wall-clock start `ts`, and meeting UUID `session_id`.
- If the store cannot open or pass the packaged-app encryption policy, recording never
  starts. If later embedding/commit fails, retain the encrypted pending row and return
  `persistence_pending`; `openbird meeting recover --json` retries rows idempotently
  and deletes only those committed successfully.
- `openbird data purge --all` removes all pending rows. Purge `--since` and prune
  remove rows selected by their start timestamp using the same strict boundaries
  as observations. Existing source-agnostic retention therefore applies to meeting
  observations and pending rows; user-initiated meetings are not exempt.

## CLI and model preparation

Turn `meeting` into a Typer group while preserving the current `openbird meeting`
status callback:

- A shared model-cache resolver sets `HF_HOME` to
  `~/.openbird/models/huggingface` for prepare, record, and the ASR thread before any
  backend import/load. `record` additionally forces Hugging Face offline/local-only
  resolution; a cache miss is `model_not_prepared` and never triggers network I/O.
- `openbird meeting prepare --jsonl`: download/load the selected backend, and emit
  metadata-only download progress plus the safe backend name. SIGTERM cancels the
  process; Hugging Face's incomplete cache remains resumable.
- `openbird meeting record --jsonl`: run until a requested stop, then emit progress
  and the terminal metadata result.
- `openbird meeting recover --json`: retry one/all encrypted pending rows.

Extend `Transcriber` with `prepare()` and an `active_backend` property. Auto mode
continues to fall back only on the existing typed backend errors. The Homebrew formula
remains CLI-only with no ASR extra; its status/record command must explain that meeting
recording requires the Apple-Silicon cask or a source install with `meetings-mlx`.

## macOS lifecycle and UI

`OpenBirdService` owns at most one preparation process and one meeting process. It
parses only metadata JSONL, exposes cancel/start/stop, and owns the supervisor write
end. It never discovers or kills meeting processes by name.

`AppModel` owns:

`idle -> consent/download -> preparing -> recording -> finalizing -> idle`

It gates start on audio-helper presence, Screen Recording, Microphone, backend package
readiness, encrypted-store release policy, and the embedding/model-route policy
surfaced by preflight. It
shows download bytes, finalization windows remaining, partial/pending recovery, and a
concise completion result.

Use one narrow AppKit hook: `AppDelegate.applicationShouldTerminate` delegates to the
model. If no meeting is active it returns `.terminateNow`. During prepare/record/finalize
it returns `.terminateLater`, stops the helper immediately, allows up to the same
180-second finalization deadline, then calls `NSApp.reply(toApplicationShouldTerminate:)`.
The death pipe remains the crash/SIGKILL backstop. A Swift quit-path test pins this
behavior; the existing `willTerminate` cleanup remains only a redundant final guard.

SwiftUI stays the source of truth:

- A focused Today glass card contains consent/download copy, progress + Cancel,
  Start, red recording elapsed time, Stop, finalization progress, and the last result.
- The menu-bar menu mirrors Start/Stop; when recording the menu-bar symbol becomes
  `waveform.circle.fill`.
- No dedicated Meetings destination or transcript editor is added.

## Privacy routes and deletion

Update `docs/privacy-routes.yaml` as executable architecture:

- `meetings.audio`: `status: implemented`; local raw-audio route, encrypted transcript checkpoint,
  SQLite observation/blob/chunk/FTS/vector storage, explicit no-raw-audio-persistence,
  ASR worker/private-queue enforcement, partial-result truth, and deletion through
  purge/prune plus uninstall/zap.
- `meetings.audio.egress.inherited_by` explicitly lists `embedding.remote`,
  `chat.remote`, and `routines.summary`, matching the existing captured-data model.
  Saving may send meeting chunks to a configured remote embedder; Ask and scheduled
  routines may send retrieved meeting text to a configured remote model only under
  their existing `OPENBIRD_ALLOW_CLOUD`/provider gates and warnings. Preflight and the
  recording consent/readiness surface state those inherited routes; the feature does
  not claim meeting transcript egress is always local.
- `meetings.model_download`: third-party network route to Hugging Face containing only
  the public model id and normal HTTP metadata, never captured/user content. It requires
  the explicit approximately 2.51 GB consent surface and supports cancel/resume/offline failure.
- Truth surfaces are the real contracts: `preflight.meetings.transcription`,
  `cli.meeting_status`, `app.meeting_recording_state`, and
  `app.meeting_model_download_consent`.
- Add tests pinning route status, forbidden fields, enforcement, storage, deletion,
  and truth-surface references. Add purge/prune tests proving meeting observations and
  pending rows are removed across observation/blob/chunk/FTS/vector storage.

Meeting transcripts remain excluded from capture-scoped background block summaries and
the current capture-scoped Deep Brain query. Scheduled routines are source-agnostic and
therefore may consume meeting rows; that behavior is declared through
`routines.summary` inheritance above. Ask retrieval is source-agnostic and is the
primary supported consumption path in this MVP. Today briefing/timeline capture-only
call sites and cache keys remain unchanged.

## Distribution and documentation

- Include `meetings-mlx` in the notarized DMG defaults on Apple Silicon and make the
  source/dev app wrapper use the same extra.
- After installing extras, `package_dmg.sh` must execute the staged Python and import
  both `parakeet_mlx` and `mlx.core`; a false platform marker, Rosetta build, or missing
  backend fails packaging before claims/codesigning.
- Record the hardened-runtime spike result below before implementation approval. The
  spike must cover dependency install, relocation audit, nested Mach-O signing,
  notarization, stapling, relocated CLI smoke, resulting DMG size, and whether MLX
  requires new Python entitlements.
- The currently published app/audio-helper binaries are arm64. This MVP is explicitly
  Apple-Silicon-only; cask and README copy must say so. Intel remains supported only by
  the formula/portable CLI path and gets no app meeting-recording claim.
- Update README, `Casks/openbird.rb` caveats, `script/package_dmg.sh` comments/defaults,
  `script/release.env.example`, privacy data-flow docs, and CLI help. Remove every
  statement that the notarized app cannot capture meetings.

### Hardened-runtime spike result

Completed on 2026-07-22 with
`OPENBIRD_DMG_EXTRAS=encryption,integrations,meetings-mlx ./script/package_dmg.sh`
as the dependency/build baseline and a manually completed relocation/sign/notary
pass over that staged artifact:

- The first relocation audit correctly rejected four absolute install IDs embedded
  by wheels: scikit-learn's `libomp.dylib` and SciPy's `libgfortran.5.dylib`,
  `libquadmath.0.dylib`, and `libgcc_s.1.1.dylib`. Rewriting each self-ID to
  `@loader_path/<basename>` made the full nested Mach-O audit pass. The production
  packaging change must perform and verify this normalization before signing.
- The copied/relocated bundle passed both direct CLI and symlink-invocation smokes.
  Every nested Mach-O and the outer app passed strict Developer ID verification.
- The initial import/matrix smoke was not sufficient: a real signed Parakeet model
  load was killed by the kernel with an invalid executable-page code-signing fault.
  The embedded Python therefore carries `disable-library-validation`, `allow-jit`,
  and `allow-unsigned-executable-memory`. These broader runtime permissions are
  scoped to the embedded interpreter only; OpenBird and both helpers retain their
  narrower entitlements. Packaging now executes a signed MLX kernel after signing,
  so losing this requirement fails the build instead of shipping an exit-137 app.
- The staged app measured 876,592 KiB on disk. The compressed DMG measured
  417,902,200 bytes (399 MiB).
- Apple accepted app submission `2e16457b-6c7f-48ef-89cd-95a22057e21a` and DMG
  submission `d901fe29-4dfc-4424-8ded-e2b667daedc0`. Both artifacts stapled and
  validated successfully; Gatekeeper reported `accepted`, `Notarized Developer ID`.

The model weights remain the separate, explicitly consented approximately 2.51 GB first-use
download described above; they are not embedded in the 399 MiB DMG.

## Validation

- Python: strict IPC truncation/EOF reason, helper killed mid-recording with partial
  save, death-pipe token/EOF, bounded queue overflow, ordered results, empty meeting,
  fail-closed pre-launch supervision, four-hour helper cap, transcription failure,
  finalization deadline, transcript cap, encrypted pending-row atomicity/recovery idempotency,
  `add_observation` failure retention, batched embeddings, and metadata-only errors.
- CLI: backward-compatible `openbird meeting`, prepare progress/cancel/offline/cache,
  record JSONL, recover, missing helper/backend, and formula guidance.
- Swift: state/readiness transitions, download progress/cancel, JSONL parsing, owned
  process start/stop, early/partial/pending results, recording symbol, and delayed quit.
- Privacy/deletion: route inheritance/release-policy contract plus full/since/prune
  deletion of SQLite artifacts and pending rows.
- Local gates: full pytest, `swift test`, `swift build`, shellcheck, signed helper smoke,
  and the hardened-runtime DMG gate.
- Throughput/VAD fixture gate on the supported Apple-Silicon Mac: use locally generated
  speech plus silence/music/notification fixtures through the real helper/backend;
  require Parakeet RTF <= 0.20, no dropped windows in a 10-minute run, zero transcript
  observations for a silence-only run, and document music/notification false positives.
  If the energy VAD fails that gate, replace `MeetingPipeline.is_speech` behind its
  existing seam before merge.
- Manual app pass: consent -> download/progress/cancel/resume -> record -> stop ->
  progress -> Ask a grounded question that cites the meeting observation. Repeat with
  helper termination and embedding failure to verify explicit partial/pending recovery.

### Validation evidence (2026-07-22)

- The real Parakeet backend transcribed a locally generated 13.1-second speech WAV
  in 1.093 seconds (RTF 0.0834) and produced 233 transcript bytes. Silence, a
  five-second 440 Hz music tone, and an 880 Hz notification tone produced zero
  transcript bytes.
- App-use testing of the signed staged bundle captured the same speech through the
  actual ScreenCaptureKit helper and saved an encrypted meeting observation. This
  exposed and fixed an MLX thread-affinity bug: model preparation and inference now
  run on one persistent ASR worker, and a regression test pins that invariant.
- A continuous 10 minute 17 second signed-app recording, driven by 43 repetitions of
  the known speech fixture, stopped and saved with the `completed` reason. By contract
  that terminal reason requires zero dropped windows, zero failed windows, no
  transcript truncation, and no early-helper/finalization error.
- The isolated encrypted store reported one observation/blob/chunk/vector after the
  short app pass. Ask initially remained gated by its cached zero-observation count;
  the completion handler now refreshes memory stats immediately when a meeting
  observation is committed.

## Non-goals

- Automatic recording, participant diarization beyond mic=`me`/system=`others`, raw
  audio retention/export/playback, cloud STT, Today briefing/timeline widening, a
  Meetings navigation destination, transcript editing, and synchronous action-item
  generation.
