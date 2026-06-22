# Claude Code — Implementation & Validation Prompt: OpenBird (Liquid Glass)

You are implementing the **OpenBird** macOS app UI from a design handoff. Read `README.md` in this
folder first — it is the source of truth for layout, tokens, copy, and the Liquid Glass material recipe.
This file tells you **how to validate** your implementation against the reference screenshots.

## Target environment
OpenBird is a native macOS menu-bar app. Implement in **SwiftUI on macOS 26+** and use the system
**Liquid Glass** APIs (`.glassEffect(.regular)` / `Glass` materials, `.regularMaterial`) rather than
re-deriving the blur/highlight math. If a different stack is already established in the repo, match it and
recreate the look with that stack's idioms. The HTML in this bundle (`OpenBird.dc.html`) is a **reference
prototype**, not code to port.

## About the reference screenshots
The PNGs in `screenshots/` were captured from the prototype with the live backdrop-blur **flattened to a
solid fill** (the capture renderer can't rasterize `backdrop-filter`). So judge **layout, hierarchy,
spacing, color, type, and copy** against them — but for the **true glass translucency, blur, and the way
background color refracts through panels**, open `OpenBird.dc.html` in a browser. Your native build should
match the *live HTML* for material feel and the *screenshots* for everything else.

## Validation protocol
For each screen below: build it, run the app (or SwiftUI preview), capture the same state, and compare
side-by-side to the named reference file. Tick every checklist item. Treat a miss as a bug to fix, not a
variation to keep. There are **8 reference files** — every one must have a matching implemented state.

---

### `screenshots/01-menu-dropdown-dark.png` — Menu bar + dropdown (dark)
The 28px menu bar with the OpenBird item open into its dropdown.
- [ ] Menu bar is 28px tall; left shows Apple logo, **OpenBird** (semibold), File / Capture / View / Help.
- [ ] Right shows wifi + battery glyphs, the **accent-filled bird button**, and tabular clock `Sat Jun 20  9:41 AM`.
- [ ] Dropdown is 288px wide, anchored under the bird button, radius ~12.
- [ ] Row order exactly: Open OpenBird · Ask OpenBird… (`⌥Space`) · divider · **CAPTURING** (red dot, `1,284 today`) · Pause Capture (`⌘P`) · Today's Activity (`›`) · divider · Capture helper `OK` · Audio helper `OK` · Encryption at rest `on` (all green dots) · divider · Re-check Setup · Data Folder · Settings… (`⌘,`) · Quit OpenBird (`⌘Q`).
- [ ] Hovering a row fills it with the accent color and white text.
- [ ] Shortcut hints are right-aligned and dimmed; the CAPTURING eyebrow is uppercase/tracked.

### `screenshots/02-ask-spotlight-dark.png` — Ask · Direction A "Spotlight" (dark)
Single centered command-palette card (~620px wide).
- [ ] Header row: bird icon + 18px input `Ask about your work…` + `esc` chip; divider below.
- [ ] Status line `Answer · grounded in 4 sources from yesterday` with a green dot.
- [ ] Prose summary paragraph, then a 4-row time→description list (`9:12–11:40`, `11:00 AM`, `2:15 PM`, `3:40 PM`) with bold app/file names.
- [ ] Source chip row: VS Code `{}` `9:12 AM`, Zoom `Z` `11:00 AM`, Linear `L` `2:15 PM`, Notion `N` `3:40 PM` — each a small colored glyph tile + label.
- [ ] Suggestion chips at the bottom (pill, 999px radius): "Summarize the Memory sync", "What's left on OB-142", "Draft my standup".
- [ ] Glyph tile colors: VS Code `#2f6be0`, Zoom `#2d8cff`, Linear `#5e6ad2`, Notion `#4b4b52`.

### `screenshots/03-ask-sources-rail-dark.png` — Ask · Direction B "Sources rail" (dark)
800×540 window: chat left, sources panel right.
- [ ] Titlebar: traffic lights left, centered `Ask OpenBird`, right `● grounded`.
- [ ] User bubble (accent) `What did I work on yesterday?` right-aligned.
- [ ] Assistant answer = intro line + 4 bullets, each ending in a **superscript citation number** ¹²³⁴ in accent.
- [ ] Bottom: rounded input `Ask a follow-up…` + circular accent send button.
- [ ] Right **262px Sources panel**, header `SOURCES · 4`; 4 numbered cards: `1 rag.py — openbird` (VS Code · 9:12 AM), `2 Memory sync · 4 people` (Zoom · 11:00 AM), `3 OB-142 · Citation validation` (Linear · 2:15 PM), `4 Storage growth & retention` (Notion · 3:40 PM) — each with glyph tile + one-line excerpt.
- [ ] Citation numbers in the answer correspond to the numbered source cards.

### `screenshots/04-ask-timeline-dark.png` — Ask · Direction C "Timeline-grounded" (dark)
840×560 window: day timeline rail left (~312px), narrative chat right.
- [ ] Titlebar centered `Yesterday · Ask`.
- [ ] Left rail header `FRIDAY, JUNE 19`; 4 sessions on a connector with colored nodes: rag.py (VS Code · 2h 28m · 142 captures), Memory sync · 4 people (Zoom · 35m · transcript), OB-142 · Citation validation (Linear · 35m · 8 captures), Storage growth & retention (Notion · 50m · 64 captures) — each with start/end times.
- [ ] Right side: user bubble, then a narrative answer that reads **down the day** with time-stamped colored-dot bullets (`9:12 …`, `11:00 …`, `2:15 …`, `3:40 …`).
- [ ] Bottom follow-up input + send button.

### `screenshots/05-today-dayview-dark.png` — Today / day view (dark)
940×600 main window.
- [ ] 222px sidebar: traffic lights, bird + **OpenBird** wordmark, nav **Today** (active/accent) · Timeline · Meetings · Routines · Search · Settings, footer `Capturing · 3 apps allowed · on-device · encrypted` with a pulsing dot.
- [ ] Header: `Yesterday`, `Friday, June 19 · 6 sessions across 6 apps`, and an `Ask about this day` button (with bird icon).
- [ ] **Daily briefing card**: accent eyebrow `DAILY BRIEFING`, `generated 7:00 AM`, summary paragraph, and stat chips `6 apps` / `1 meeting · 35m` / `1,284 captures` / `5h 53m active`.
- [ ] `TIMELINE` section with session cards on a vertical connector rail (app tile, title, app · duration, capture count, one-line description). First card = `rag.py — openbird`, VS Code · 2h 28m, 142 captures.

### `screenshots/06-setup-onboarding-dark.png` — Onboarding / permissions (dark)
540px centered sheet.
- [ ] Centered 54px bird, `Welcome to OpenBird`, privacy subhead emphasizing it reads the **text** of the active window — never screenshots.
- [ ] Four permission rows (icon tile in faint-accent square, title + subtitle, trailing status): Screen text capture **✓ Granted** · Accessibility **✓ Granted** · System audio for meetings **[Enable]** button · Local model · Ollama **● Connected** (`llama3.2 + nomic-embed-text`).
- [ ] Primary full-width `Start capturing` button (accent). Footer lock line "Nothing leaves your device…" (may sit below the fold — verify it exists).

### `screenshots/07-ask-spotlight-light.png` — Ask · Spotlight (light)
Same as `02` but **light appearance**.
- [ ] Light glass: bright frosted white card, dark text, accent unchanged.
- [ ] All Spotlight content from `02` present and identical in layout.
- [ ] Verify your light-mode token swap matches README's light palette (text `rgba(0,0,0,0.85/0.5/0.34)`, separators `rgba(0,0,0,0.1)`, etc.).

### `screenshots/08-today-dayview-light.png` — Today (light)
Same as `05` but **light appearance**.
- [ ] Light sidebar + window; active **Today** nav still accent-filled.
- [ ] Daily briefing card + timeline render with light fills, dark text, readable contrast.
- [ ] Full light/dark parity with `05` — no missing elements, same spacing.

---

## Cross-cutting checks (all screens)
- [ ] **Liquid Glass material** (validate against the live `OpenBird.dc.html`, not the flattened PNGs): translucent fill, real backdrop blur, a bright **top specular edge**, a faint full-perimeter rim, and a soft lower inner glow. Surfaces float above content; background color subtly refracts through. Prefer native `.glassEffect`.
- [ ] **Accent** `#2f7ff2` used only for primary actions, active nav, links, citation numbers, focus.
- [ ] **Typography**: SF Pro / system font; sizes per README (21/700 titles, 18 spotlight input, 14/1.6 body, 13 menu rows, 10.5/700 uppercase eyebrows); tabular-nums for all times/counts.
- [ ] **Radii**: 17 spotlight / 13 windows / 12 dropdown / 11 cards / 999 pills; **traffic lights** `#ff5f57 #febc2e #28c840`.
- [ ] **Light & dark** both fully themed; capturing/status dot is `#ff453a` and pulses; health dots green.
- [ ] **App glyph identity** consistent everywhere: VS Code `#2f6be0`, Zoom `#2d8cff`, Linear `#5e6ad2`, Notion `#4b4b52` (swap letter tiles for real app icons where possible).
- [ ] **Copy** matches the prototype exactly (the seed answer to "What did I work on yesterday?" and all labels).
- [ ] The bottom-center **prototype control pill is NOT part of the product** — do not implement it.

## Report back
After validating, produce a short report: for each of the 8 files, `PASS` or the specific deltas you found
and fixed. Flag anything in the design that conflicts with macOS HIG or the repo's existing patterns so a
human can decide.
