# Design: Liquid Glass quick-chat "Spotlight" Ask panel

**Status:** APPROVED — Codex consensus (3 rounds: 5 findings → 2 → 0; final
VERDICT approve). Cleared to implement.
**Branch:** `feat/quick-chat-glass-panel` (off `origin/main` @ca7fbea).
**Source of truth:** `docs/design/handoff/design_handoff_openbird_liquid_glass/README.md`
(Claude Design "Mac work activity tracker" handoff). The `.dc.html`/`support.js`
are reference prototypes only — **not** ported.

## Goal

Ship **Direction A ("Spotlight")** from the handoff: a global-hotkey (⌥Space),
floating, centered glass panel that asks a grounded question over captured memory
and renders the answer + citations — wired to the **existing** `openbird chat
--json` pipeline (`AppModel.ask` → `OpenBirdService.askChat`). Plus the **shared
Liquid Glass foundation** (design tokens + a reusable surface material) that every
later surface (menu dropdown, Today view, Onboarding) will reuse.

Scope decided with the user:
- Direction A only (not B "Sources rail" / C "Timeline"). Pick-one-to-ship.
- macOS **13+ retained**: build the look on `.ultraThinMaterial`/`.regularMaterial`
  + the spec's gradient/shadow recipe; opt into real `.glassEffect()` only behind
  `if #available(macOS 26, *)`. No deployment-target bump → no tester dropped.

Out of scope for this PR (later, tracked in Plan.md Phase 2): menu-dropdown reskin,
Today/day-view window, Onboarding reskin, Directions B/C.

## Files (all under `mac-app/Sources/OpenBirdApp/`)

New:
- `DesignSystem/DesignTokens.swift` — colors, radii, spacing, type ramp, source
  identities, transcribed from the handoff "Design Tokens" section. Pure values.
- `DesignSystem/GlassSurface.swift` — `View.glassSurface(_:)` modifier. The
  macOS 13 material path is the **unconditional default**; native `.glassEffect`
  is reached only inside a **compile-time** gate (Codex #1):
  `#if compiler(>=6.2)` → `if #available(macOS 26, *)` → `.glassEffect(.regular,
  in:)`, with the material path in every other branch. So on an SDK predating
  macOS 26 the symbol is never referenced and the build still links on 13+.
- `DesignSystem/BirdLogo.swift` — the hummingbird as a SwiftUI `Shape`/`Path`
  (handoff ships it as inline SVG `viewBox 0 0 48 48`). Used at 13–54px.
- `Ask/AskPanelModel.swift` — `@MainActor ObservableObject` that **owns the panel's
  own ask flow** (Codex #5): `thread: [AskTurn]`, `busy`, `error`. Calls
  `OpenBirdService.askChat` off-main directly (sharing only the service instance),
  so it never observes the shared `AppModel.chatResult` — no cross-surface or
  duplicate appends. Holds a read-only ref to `AppModel` only for display
  (`askUnavailableReason`, `localModelStatusSummary`).
- `Views/AskPanelView.swift` — the Spotlight card: input row (bird + `esc` chip),
  divider, answer block (grounded status line, prose, citation source chips),
  suggestion chips, follow-up thread. Binds to `AskPanelModel` (asks) + `AppModel`
  (read-only status). Uses `@FocusState` for the text field.
- `Ask/AskPanel.swift` — `final class AskPanel: NSPanel` overriding
  `canBecomeKey = true` (Codex #2 — a borderless/nonactivating panel will not take
  key/first-responder otherwise) and `canBecomeMain = false`.
- `Ask/AskPanelController.swift` — `@MainActor` controller owning the `AskPanel`
  (styleMask `[.borderless, .nonactivatingPanel]`, **level `.statusBar`**,
  `collectionBehavior [.canJoinAllSpaces, .fullScreenAuxiliary, .transient]`
  (Codex #3 — `.floating` alone is unreliable over full-screen spaces), centered)
  hosting `AskPanelView` via `NSHostingView`. `toggle()`/`show()`/`hide()`. `show()`
  does `makeKeyAndOrderFront`, then focuses the field on the next main-runloop turn
  (`DispatchQueue.main.async`). Esc and `windowDidResignKey` dismiss. Idempotent
  `installHotKeyIfNeeded()`.
- `Ask/GlobalHotKey.swift` — Carbon `RegisterEventHotKey` wrapper for ⌥Space
  (works macOS 13+, no Accessibility/TCC needed; one process-wide
  `InstallEventHandler`). Calls a closure on press; unregisters + removes the
  handler on deinit.

Changed:
- `App/OpenBirdApp.swift` — create `service`, `model`, and `AskPanelController`
  **together in `init()`** sharing the one `AppModel`/`OpenBirdService` (Codex #4 —
  no second model, no configure-after-launch race):
  `let service = OpenBirdService(); let model = AppModel(service: service);
  _model = StateObject(wrappedValue: model);
  _askPanel = StateObject(wrappedValue: AskPanelController(model: model, service: service))`.
  A `.task` calls the idempotent `askPanel.installHotKeyIfNeeded()` (hotkey →
  `askPanel.toggle()`); idempotence removes any lifecycle-timing race.
- `Views/MenuBarView.swift` — "Ask OpenBird…" calls an injected `openAskPanel`
  closure (→ `askPanel.show()`), shows the ⌥Space shortcut; keep "Open OpenBird"
  for the window.
- `Models/AppModel.swift` — expose the shared `OpenBirdService` (or accept it being
  passed alongside) so the panel model reuses the same instance. Replace the
  `private static describeChatError` body with a call to the new shared
  `ChatErrorPresenter.describe(_:)` (behavior-preserving). No change to the
  existing `ask/chatResult` window-chat flow.
- `Services/ChatErrorPresenter.swift` (new) — `enum ChatErrorPresenter { static
  func describe(_ error: Error) -> String }`, lifted verbatim from
  `AppModel.describeChatError`, used by both `AppModel.ask` and `AskPanelModel.ask`
  so error wording is single-sourced and unit-testable.

## Liquid Glass foundation (the shared recipe)

`glassSurface(cornerRadius:)` — compile-time gate first, runtime guard inside
(Codex #1). The material path is the unconditional fallback so the symbol
`.glassEffect` is only ever compiled against an SDK that defines it:
```swift
@ViewBuilder func glassSurface(_ r: CGFloat) -> some View {
  #if compiler(>=6.2)            // toolchain new enough to KNOW about glassEffect
  if #available(macOS 26, *) {
    self.glassEffect(.regular, in: RoundedRectangle(cornerRadius: r))
  } else {
    self.materialSurface(r)      // <- shared fallback below
  }
  #else
  self.materialSurface(r)        // older SDK: never references glassEffect
  #endif
}
// materialSurface(r):
//   .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: r))
//   .overlay(topSheenGradient)            // linear-gradient sheen from spec
//   .overlay(RoundedRectangle(...).stroke(borderColor, lineWidth: 0.5))
//   .shadow(...) // float + contact; inner specular via overlay strokes
```
- Sheen/fill/border/shadow values come verbatim from the handoff "Liquid Glass —
  the core material" section, light/dark variants via `@Environment(\.colorScheme)`.
- Color refraction "orbs": in native we rely on real translucency picking up
  desktop/app content — do NOT paint fake orbs behind a window panel (that would
  defeat translucency). Keep the panel genuinely translucent.

## AskPanelView (Direction A) — layout per handoff §2-A

- 620pt wide, radius 17, single centered glass card, `obPop` entrance.
- Input row: 18pt; bird icon (tint when active) + `TextField` ("Ask about your
  work…") + `esc` chip. Enter submits via `model.ask`; clears draft.
- Divider, then answer area (only after first ask):
  - status line: green dot + "grounded in N sources" (from `result.grounded` +
    `result.citations.count`); amber "ungrounded — no verified source" when
    `!grounded` (reuse the existing ChatView wording).
  - prose: `result.answer` (`textSelection(.enabled)`).
  - source chips: one per citation — app glyph tile + "`app` · `time`", using the
    spec's source identities; falls back to window/"unknown" + abbreviated time
    (reuse ChatView's `sourceLabel`/`timeLabel` logic; lift to a shared helper).
  - busy: 3-dot "thinking" (`obDot`) while `askModel.busy` (panel-owned, NOT
    `AppModel.chatBusy`).
  - error: the current turn's `AskTurn.error` row (panel-owned, NOT
    `AppModel.chatError`).
- Suggestion chips (pills, radius 999): static, query-seeding ("Summarize my
  meetings", "What did I work on yesterday?", "Draft my standup") — tapping sets
  the draft and asks. (The handoff's demo keyword-matching is prototype-only; real
  answers come from RAG.)
- Respects `model.askUnavailableReason` (no memory yet) — show it inline, disable ask.

## State

- The panel does **not** read the shared `AppModel.chatResult` (Codex #5). Its
  ask flow lives in `AskPanelModel`:
  - `thread: [AskTurn]` where `AskTurn = {question: String, result: ChatResult?,
    error: String?}`; `busy: Bool`.
  - `ask(_:)` appends a pending turn, then `await Task.detached { service.askChat }`,
    and fills in *that* turn's `result`/`error` on completion — append/update is
    keyed to the panel-initiated request, so there is no duplicate or cross-surface
    append even if some other surface republishes its own state.
  - read-only display values (`askUnavailableReason`, `localModelStatusSummary`)
    are read from the shared `AppModel` so the "no memory yet" guidance matches the
    rest of the app.
- The existing `AppModel.ask/chatResult` window-chat path is untouched.
- Theme: follow system `colorScheme` (no in-app toggle — that was prototype-only).

## Failure modes → handling (working-agreement requirement)

1. Hotkey registration fails (already taken by another app) → log reason code
   `hotkey-register-failed`; the menu item "Ask OpenBird…" still opens the panel,
   so the feature degrades, never breaks.
2. Panel shown with no memory yet → `askUnavailableReason` inline; ask disabled.
3. Chat CLI missing / model down / timeout → friendly text. The existing
   `AppModel.describeChatError` is `private static`; extract it to a shared
   internal `ChatErrorPresenter.describe(_:) -> String` (or `ChatError.userMessage`)
   that BOTH `AppModel.ask` and `AskPanelModel.ask` call, so the panel and the
   window render identical, tested error wording. Store the string on the turn's
   `AskTurn.error`.
4. macOS < 26 → material path; no `glassEffect` symbol referenced at runtime
   (guarded by `#available`), so it links and runs on 13+.
5. Panel never steals focus from capture target inappropriately:
   `.nonactivatingPanel` + only activate when explicitly shown for input.
6. Privacy: the panel shows only the user's own typed question + the grounded
   answer the CLI already returns. No captured text is logged by the app; chat
   text continues to flow via STDIN, never argv (unchanged).

## Tests (`mac-app/Tests/OpenBirdAppTests/`)

Swift logic that can be unit-tested without a window:
- `DesignTokens` source-identity lookup (VS Code/Zoom/Linear/Notion → glyph+color;
  unknown → fallback).
- shared `sourceLabel`/`timeLabel` helper (extracted from ChatView) — app/window
  precedence, empty → "unknown".
- suggestion list non-empty + each seeds a non-empty draft.
- `AskPanelModel.ask` against a stubbed/seamed chat call: a single ask appends
  exactly one thread turn and fills its result; an empty/whitespace question is a
  no-op; an error populates that turn's `error` not a phantom result (guards
  Codex #5 — no duplicate/cross-surface append). The service call is injected via
  a closure seam so the test never spawns a process.
NSPanel/hotkey/`canBecomeKey`/focus are AppKit-integration glue — covered by build
+ manual smoke, not unit tests (documented in PR "how tested").

## Definition of done

```text
Codex consensus on this plan → implement → `swift build` green +
`uv run python -m pytest -q` green (no Python change expected) → manual smoke
(⌥Space toggles panel; ask returns a cited answer; light/dark both legible;
runs on a pre-Tahoe target) → Codex diff-review clean → PR → CodeRabbit → merge.
```
