# Handoff: OpenBird — Liquid Glass UI

> Historical prototype note: this handoff predates Phase C2 deep capture. Treat
> the root `README.md`, `docs/design/privacy-data-flow.md`, and
> `docs/privacy-routes.yaml` as authoritative for current privacy copy and
> Screen Recording scope.

## Overview
OpenBird is a local-first macOS work-memory app. It runs in the menu bar, stores text from your active window by default, optionally uses per-app transient window stills for on-device OCR, transcribes meetings locally, and answers natural-language questions about what you did ("What did I work on yesterday?") with grounded, citation-backed answers drawn from on-device memory.

This handoff covers the full prototype: the **menu bar + dropdown**, a **standalone Ask chat** (three explorable directions), a **Today / day-view window**, and an **Onboarding / permissions** screen. The visual language is Apple's **Liquid Glass** (macOS 26 Tahoe): translucent materials with specular edge highlights, a top sheen, lensing rim light, and colorful content refracting through the glass.

## About the Design Files
The files in this bundle (`OpenBird.dc.html`, `support.js`) are **design references created in HTML** — a prototype showing the intended look and behavior. They are **not production code to copy directly**. `*.dc.html` is a self-contained prototype format; `support.js` is its runtime and is **not** part of the product.

The task is to **recreate these designs in OpenBird's actual environment**. Given this is a native macOS menu-bar app, the natural target is **SwiftUI on macOS 26+**, where Liquid Glass is a first-class system material (`.glassEffect()`, `Glass` backgrounds, `.regularMaterial`/`.ultraThinMaterial`). If a different stack is already in use (e.g. Electron + React), recreate the look with that stack's patterns. Use real system materials wherever possible rather than re-implementing the glass math by hand — the CSS below is a faithful *description* of the target, not a prescription.

## Fidelity
**High-fidelity (hifi).** Final colors, typography, spacing, radii, and interactions are specified. Recreate pixel-faithfully, but prefer native Liquid Glass materials over literal CSS translation.

---

## Liquid Glass — the core material
Every floating surface (menu bar, dropdown, chat panel, windows, control pills) is built from one shared recipe. In SwiftUI, reach for `.glassEffect(.regular)` / a `Glass` background on the container and let the system do the highlights. The CSS recipe below documents the intended result so you can match it if you're not on a native glass API.

**Backdrop:** `blur(50px) saturate(180%) brightness(1.08)` (large surfaces); `blur(36px) saturate(170%) brightness(1.05)` (menu bar).

**Material fill** (sheen baked into the fill as a top-down gradient over a translucent base):
- Dark panel: `linear-gradient(180deg, rgba(255,255,255,0.14), rgba(255,255,255,0.02) 36%, rgba(255,255,255,0) 62%), rgba(46,46,55,0.6)`
- Dark window: `linear-gradient(180deg, rgba(255,255,255,0.11), rgba(255,255,255,0) 32%), rgba(34,34,42,0.64)`
- Light panel: `linear-gradient(180deg, rgba(255,255,255,0.85), rgba(255,255,255,0.3) 40%), rgba(250,250,252,0.55)`
- Light window: `linear-gradient(180deg, rgba(255,255,255,0.8), rgba(255,255,255,0.25) 34%), rgba(252,252,253,0.62)`

**Glass shadow stack** (the signature — top specular line, lensing rim, lower inner glow, float + contact shadows):
- Dark: `inset 0 1px 0 rgba(255,255,255,0.5), inset 0 0 0 0.5px rgba(255,255,255,0.16), inset 0 -12px 22px -14px rgba(255,255,255,0.22), 0 30px 70px rgba(0,0,0,0.55), 0 4px 14px rgba(0,0,0,0.3)`
- Light: `inset 0 1px 0 rgba(255,255,255,0.95), inset 0 0 0 0.5px rgba(255,255,255,0.6), inset 0 -12px 22px -14px rgba(255,255,255,0.7), 0 30px 70px rgba(0,0,0,0.18), 0 3px 10px rgba(0,0,0,0.12)`

**Border:** `0.5px solid rgba(255,255,255,0.10)` (dark) / `rgba(0,0,0,0.08)` (light).

**Color refraction:** three soft, blurred color orbs sit *behind* the glass so the translucency picks up color (purple `rgba(120,90,255,…)`, blue `rgba(47,127,242,…)`, pink `rgba(255,120,170,…)`), each `~420–520px`, `filter:blur(20px)`, radial-gradient fading to transparent at 70%. In native, this is whatever app/desktop content sits behind the glass — keep surfaces genuinely translucent so it shows.

---

## Screens / Views

### 1. Menu bar + dropdown
- **Purpose:** Always-present status & quick actions. The bird icon button lives at the right of the system menu bar.
- **Layout:** Fixed bar, height **28px**, space-between. Left: Apple logo, "OpenBird" (600 weight), File / Capture / View / Help (opacity .82). Right: wifi + battery glyphs, the bird button (30×20, radius 6; accent-filled when active), tabular clock "Sat Jun 20  9:41 AM".
- **Dropdown:** 288px wide, radius 12, glass panel, anchored `top:31px; right:80px`, 6px padding. Rows are 5px/10px, radius 6, **hover = accent background, white text**. Contents: Open OpenBird · Ask OpenBird… (⌥Space) · divider · **Capturing** status (pulsing red dot, "1,284 today") · Pause Capture (⌘P) · Today's Activity (›) · divider · three green-dot health rows (Capture helper OK / Audio helper OK / Encryption at rest on) · divider · Re-check Setup · Data Folder · Settings… (⌘,) · Quit OpenBird (⌘Q). A full-screen invisible click-catcher dismisses it.

### 2. Ask — standalone chat (THREE directions to compare)
All three answer the same seed query **"What did I work on yesterday?"** with the same realistic answer, grounded in 4 sources (VS Code 9:12, Zoom 11:00, Linear 2:15, Notion 3:40). Pick one to ship; they explore different ways to surface citations.

- **Direction A — "Spotlight":** 620px, radius 17, single centered glass card. Big 18px input row with bird icon + `esc` chip, divider, then the answer: a "grounded in 4 sources" status line, a prose summary, a 4-row time/description list, a row of source chips (app glyph tile + "App · time"), follow-up message thread, thinking dots, and a row of suggestion chips (pill, 999px). Fastest / lightest.
- **Direction B — "Sources rail":** 800×540 window (traffic-light titlebar "Ask OpenBird"). Left = chat column (user bubble accent, assistant prose with bulleted points carrying superscript citation numbers ¹²³⁴, bottom input pill + round send button). Right = **262px Sources panel**: 4 numbered cards, each app glyph tile + title + "App · time" + a one-line quote/excerpt. Best for verifiable citations.
- **Direction C — "Timeline-grounded":** 840×560 window ("Yesterday · Ask"). Left = **312px timeline rail** for Fri Jun 19 (dotted connector, colored nodes, per-session app · duration · capture count). Right = chat that reads *down the day* as a narrative with time-stamped colored-dot bullets. Best for context.

**Source identity (consistent everywhere):** VS Code `{}` `#2f6be0`; Zoom `Z` `#2d8cff`; Linear `L` `#5e6ad2`; Notion `N` `#4b4b52`.

### 3. Today / day view
- **Purpose:** Main window — review a day's captured activity.
- **Layout:** 940×600 window. **222px sidebar** (traffic lights, OpenBird wordmark, nav: Today (active, accent) / Timeline / Meetings / Routines / Search / Settings, footer "Capturing · 3 apps allowed · on-device · encrypted" with pulsing dot). Main: header ("Yesterday", "Friday, June 19 · 6 sessions across 6 apps", "Ask about this day" button) → **Daily briefing card** (accent eyebrow, "generated 7:00 AM", prose summary, stat chips: 6 apps / 1 meeting · 35m / 1,284 captures / 5h 53m active) → **Timeline** of 6 session cards on a connector rail (app tile, title, "app · duration", capture count, one-line description).

### 4. Onboarding / Setup
- **Purpose:** First-run permissions.
- **Layout:** 540px glass sheet (traffic lights). Centered: 54px bird, "Welcome to OpenBird", privacy subhead ("stores text, not screenshots or images"). Permission rows (icon tile in `rgba(47,127,242,0.14)`, title + subtitle, status): Screen text capture **Granted** · Accessibility **Granted** · system-audio / Screen Recording **Enable** button · Meeting transcription status · active model route status. Primary "Start capturing" button. Footer lock line: "Local by default. Pause anytime from the menu bar."

> The bottom-center floating pill (Menu/Ask/Today/Setup + direction switcher + light/dark toggle) is a **prototype-only control** — do not ship it.

---

## Interactions & Behavior
- **Surface switching:** menu ↔ ask ↔ today ↔ setup. Bird button toggles menu/ask; dropdown items navigate.
- **Chat send:** Enter (no shift) submits; input clears; user bubble appends; ~850ms "thinking" dots; then a keyword-matched assistant answer with its own source chips. Auto-scroll to bottom on send and on answer.
- **Answer matching (demo logic):** meeting/sync/call → Memory-sync recap (Zoom + Linear sources); 142/ticket/review/citation → OB-142 status (Linear + VS Code); standup/draft/summary → drafted standup (Linear + Notion); else → generic memory fallback (VS Code + Notion). In production, replace with the real RAG pipeline.
- **Hover:** menu/nav rows → accent (or `--hover`) bg; chips/buttons → lighten; send button → brightness 1.1.
- **Animations:** `obPop` (panel entrance, opacity+translateY+scale), `obFade`, `obPulse` (capturing dot), `obDot` (3-dot thinking, staggered .15s). Keep transitions subtle/system-like.
- **Light/Dark:** full parity — every token swaps; glass goes frosted-bright in light, glowing in dark.

## State Management
- `surface`: `menu | ask | today | setup`
- `dir`: `A | B | C` (which Ask direction; ship one)
- `theme`: `dark | light`
- Per-direction: `draft` (input text), `follow[]` (message list: `{role:'user'|'assistant', text, sources?[]}`), `thinking` (bool)
- Real app additionally needs: capture on/off, today's capture count, session/timeline data, meeting transcripts, permission grants, model-connection status.

## Design Tokens
**Accent:** `#2f7ff2` (tweakable: `#7a5cff`, `#e0533d`, `#1fa463`). **Green/OK:** `#32d74b` dark / `#1e9e3a` light. **Traffic lights:** `#ff5f57 / #febc2e / #28c840`. **Capturing dot:** `#ff453a`.

**Text (dark):** `rgba(255,255,255,0.92)` / `.56` / `.34`. **Text (light):** `rgba(0,0,0,0.85)` / `.5` / `.34`. **Separator:** `rgba(255,255,255,0.12)` dark / `rgba(0,0,0,0.1)` light. **Field/Card fills:** `rgba(255,255,255,0.08)` / `0.06` dark; `rgba(0,0,0,0.05)` / `0.04` light.

**Type:** `-apple-system, BlinkMacSystemFont, 'SF Pro Text'`. Scale used: 21px/700 (window title), 18px (spotlight input), 15px/700 (wordmark), 14.5px, 14px/1.6 (body), 13.5px, 13px (menu rows), 12.5px, 11.5px, 11px, 10.5px/700/uppercase/.05em (eyebrows). Tabular-nums for times/counts.

**Radii:** 17 (spotlight) / 14 (setup) / 13 (windows) / 12 (dropdown) / 11 (cards) / 9–6 (buttons/tiles) / 999 (pills). **Spacing:** 4/6/8/12/14/18/22px rhythm.

**Glass:** see the "Liquid Glass — the core material" section above (fills, backdrop, shadow stacks, orbs).

## Assets
- **OpenBird bird logo:** inline SVG (a stylized hummingbird), `viewBox="0 0 48 48"`, single-path + ellipse, `fill:currentColor`. Recreate as an asset/SF Symbol-style vector. Used at 13–54px.
- **App glyphs** (VS Code `{}`, Zoom `Z`, Linear `L`, Notion `N`, Slack `#`, Chrome `◉`) are simple letter/symbol tiles — swap for real app icons in production.
- All other icons (wifi, battery, nav, permissions, send arrow) are inline SVG strokes — replace with SF Symbols on macOS.
- No raster images; nothing external.

## Reference Screenshots (`screenshots/`)
Per-screen reference captures. Filenames match the validation checklist in `CLAUDE_CODE_PROMPT.md`.
Note: the live **backdrop-blur was flattened to a solid fill** for these captures (the renderer can't
rasterize `backdrop-filter`) — judge layout/color/type/copy from these, but open `OpenBird.dc.html` for
the true glass translucency, blur, and color refraction.

- `01-menu-dropdown-dark.png` — Menu bar + dropdown (dark)
- `02-ask-spotlight-dark.png` — Ask · Direction A "Spotlight" (dark)
- `03-ask-sources-rail-dark.png` — Ask · Direction B "Sources rail" (dark)
- `04-ask-timeline-dark.png` — Ask · Direction C "Timeline-grounded" (dark)
- `05-today-dayview-dark.png` — Today / day view (dark)
- `06-setup-onboarding-dark.png` — Onboarding / permissions (dark)
- `07-ask-spotlight-light.png` — Ask · Spotlight (light)
- `08-today-dayview-light.png` — Today (light)

## Files
- `OpenBird.dc.html` — the full prototype (all four surfaces + three Ask directions, light/dark, demo chat logic). Open in a browser to interact.
- `support.js` — prototype runtime only. **Not** part of the product; do not port.
- `CLAUDE_CODE_PROMPT.md` — implementation + per-screenshot validation protocol. Start here for the build/validate loop.
- `screenshots/` — the 8 reference captures listed above.
