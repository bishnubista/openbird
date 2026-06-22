# Menu-bar dropdown: glass reskin deferral

## Decision

The Liquid Glass handoff (`docs/design/handoff/.../README.md` §1) specifies a **288px,
radius-12 glass dropdown** anchored under the menu-bar bird icon, with accent-on-hover
rows. We are **keeping the native `MenuBarExtra` `.menu`** and **deferring** the glass
dropdown to a separate, soak-tested change. This file records why, so the next person
doesn't re-litigate it from scratch.

## Why native menus can't just be reskinned

macOS native menus (what `MenuBarExtra` renders in its default `.menu` style) are not
style-customizable — no rounded glass container, no per-row hover-accent, no colored
status dots. The only way to render the handoff's custom chrome is to **drop
`MenuBarExtra` and hand-own an `NSStatusItem` + a custom `NSPanel`** to host SwiftUI.

## Why that rewrite is deferred (not a routine reskin)

1. **It reopens recently-stabilized territory.** History: **#51** added a glass
   window-style menu; **#57** removed `.menuBarExtraStyle(.window)` and **restored the
   native menu** because the window-style popover was un-openable and the status item
   parked off-screen when the menu bar was full. #55/#56/#59 further stabilized the
   icon. A manual `NSStatusItem` is another status-item rewrite of exactly this area.
2. **`MenuBarExtra(.window)` is a dead end.** It has no first-party API to dismiss the
   popover, access the `NSStatusItem`, or toggle presentation
   ([FB11984872](https://github.com/feedback-assistant/reports/issues/383)).
3. **Correctness is runtime-bound.** A hand-owned status item + panel must be verified
   against full menu bar (item off-screen), multiple/notched displays, capture-state
   icon updates, click-away + Escape dismissal, VoiceOver labels/roles, and not stealing
   focus — none of which static review or `swift build` can confirm. It needs hands-on
   GUI verification on a real menu bar.

A cross-family (Codex) design review reached the same conclusion: ship the native menu
and treat the glass dropdown as a separate risky platform change.

## What we did instead

Kept the native menu and aligned its **content** to the handoff where the native menu
allows: item order, and the **⌘P Pause Capture** shortcut. The current menu also keeps
the windows added since the handoff (Today, Timeline, Ask with Sources, About).

## If the glass dropdown is taken on later

Do it as its own PR with hands-on verification, preserving these invariants
(from the Codex review):

- One strong-owned `NSStatusItem` for the app's lifetime; no duplicate `MenuBarExtra`.
- `statusItem(withLength: .variableLength)`; icon from `AppModel.menuBarSymbol`
  (`bird` / `bird.fill` / `pause.circle`), `isTemplate = true`, updated on every
  capture/pause change on the main actor.
- Keep `NSApp.setActivationPolicy(.regular)` and the launch/Keychain bootstrap ordering.
- Anchor from `statusItem.button.window?.convertToScreen(button.frame)` (never guessed
  coordinates); clamp the 288px panel to the active screen's `visibleFrame`.
- No-op + privacy-safe log if the status item is off-screen/unavailable.
- Hide the panel before any action that opens a window/folder/settings or quits.
- Click-away dismiss via `windowDidResignKey` + Escape handling; honest shortcuts only.
- Manual verification: launch, relaunch, full menu bar, multiple displays, capture-state
  icon changes, click-away, shortcuts, opening each row.
